#!/usr/bin/env python3
"""Collect action-conditioned world-model training data from tournament games.

Runs pretrained agents against each other fully headless (EGL) with agent
inference on the GPU, rendering a small RGB frame every control step, and
writes one .npz per episode plus one .mp4 per game for human verification.
Replaces the two-phase eval_2023.py -> replay_2023.py flow for data
collection: no dataset.pkl, no re-rendering, no per-step logging overhead.

Episode file layout (DreamerV3 obs-first convention; T actions, T+1 states):
    image         uint8   (T+1, H, W, 3)  image[k] is the frame of state s_k;
                                          image[0] is the post-reset frame
    action        float32 (T+1, 2, 2)     commanded mallet x,y in WORLD frame
                                          (same frame the camera sees), from
                                          forward kinematics of the commanded
                                          joint positions. action[k] led into
                                          s_k, i.e. the action taken *at*
                                          image[k] is action[k+1]; action[0]
                                          is zeros. Axes: (agent, xy)
    action_joints float32 (T+1, 2, 2, 7)  the raw joint-space command behind
                                          action[k]. Axes: (agent, [pos|vel],
                                          joint)
    obs           float32 (T+1, 46)       raw low-dim env observation of s_k
    is_first    bool    (T+1,)          True only at index 0
    is_last     bool    (T+1,)          True only at index -1
    is_terminal bool    (T+1,)          legacy split_on_absorbing mode only;
                                        fixed_length games are always false
    score       int32   (T+1, 2)        running score at s_k (agent1, agent2)
    faults      int32   (T+1, 2)        running fault count at s_k

Examples:
    python scripts/collect_2023.py --model1 tournament_aggressive \
        --model2 tournament_aggressive --games 2 --steps 5000
    # Throughput scales with CPU workers (physics is the bottleneck):
    python scripts/collect_2023.py --workers 16 --platform cpu --games 16
"""
import os
import subprocess
import sys

# Must be set before mujoco / drl_air_hockey imports. EGL renders headless on
# the GPU; the DRL_AIR_HOCKEY_* variables opt agent inference into the GPU
# (read by config_dreamerv3 at agent construction). setdefault so the shell
# can override, e.g. CUDA_VISIBLE_DEVICES=1 or DRL_AIR_HOCKEY_JAX_PLATFORM=cpu.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# --platform must take effect before jax initializes, i.e. before argparse.
# An EXPLICIT cpu request (flag or env var) additionally moves rendering off
# the GPUs (see below); the automatic fallback for unsupported GPUs does not.
_requested_platform = os.environ.get("DRL_AIR_HOCKEY_JAX_PLATFORM")
if "--platform" in sys.argv[:-1]:
    _requested_platform = _requested_platform or sys.argv[sys.argv.index("--platform") + 1]
    os.environ.setdefault("DRL_AIR_HOCKEY_JAX_PLATFORM", _requested_platform)


def _default_jax_platform():
    """GPU when jaxlib can actually target it, else CPU.

    The pinned jaxlib 0.4.23 cannot generate code for GPUs newer than
    Hopper (e.g. RTX 5090 / Blackwell, compute capability 12.x) — XLA
    falls back to sm_90a PTX and ptxas aborts the process. EGL rendering
    is unaffected either way.
    """
    try:
        caps = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.split()
        if caps and all(float(cap) <= 9.0 for cap in caps):
            return "gpu"
        print("collect_2023: GPU compute capability unsupported by the pinned "
              "jaxlib; running inference on CPU (rendering stays on GPU)")
    except Exception:
        pass
    return "cpu"


os.environ.setdefault("DRL_AIR_HOCKEY_JAX_PLATFORM", _default_jax_platform())
if os.environ["DRL_AIR_HOCKEY_JAX_PLATFORM"] == "cpu":
    # Keep jax from even initializing its CUDA backend: routing computation
    # to the CPU (jax_platform_name) alone still creates a CUDA context on
    # every visible GPU — ~500 MB of VRAM per worker per GPU for nothing.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Two dreamerv3 Agent instances share this process; preallocating 75% of
# VRAM per XLA client is unnecessary and hostile to a shared GPU.
os.environ.setdefault("DRL_AIR_HOCKEY_JAX_PREALLOC", "false")


def _egl_cuda_devices():
    """List (egl_index, cuda_index) for every GPU-backed EGL device.

    EGL enumerates devices in its own order (reversed relative to CUDA on
    some hosts, plus non-CUDA software devices), so MUJOCO_EGL_DEVICE_ID
    cannot reuse the CUDA index. NVIDIA's driver exposes each EGL device's
    CUDA index through the EGL_CUDA_DEVICE_NV attribute.
    """
    import ctypes

    from OpenGL import EGL

    device_t = ctypes.c_void_p
    attrib_t = ctypes.c_ssize_t
    EGL_CUDA_DEVICE_NV = 0x323A
    query_devices = ctypes.CFUNCTYPE(
        ctypes.c_uint, ctypes.c_int, ctypes.POINTER(device_t),
        ctypes.POINTER(ctypes.c_int),
    )(EGL.eglGetProcAddress("eglQueryDevicesEXT"))
    query_attrib = ctypes.CFUNCTYPE(
        ctypes.c_uint, device_t, ctypes.c_int, ctypes.POINTER(attrib_t)
    )(EGL.eglGetProcAddress("eglQueryDeviceAttribEXT"))
    count = ctypes.c_int()
    query_devices(0, None, ctypes.byref(count))
    devices = (device_t * count.value)()
    query_devices(count.value, devices, ctypes.byref(count))
    found = []
    for i in range(count.value):
        attrib = attrib_t()
        if query_attrib(devices[i], EGL_CUDA_DEVICE_NV, ctypes.byref(attrib)):
            found.append((i, attrib.value))
    return found


def _egl_device_for_cuda(cuda_index):
    for egl_index, cuda in _egl_cuda_devices():
        if cuda == cuda_index:
            return egl_index
    raise RuntimeError(f"No EGL device maps to CUDA device {cuda_index}")


def _egl_software_device():
    """EGL index of Mesa's software rasterizer (llvmpipe), or None."""
    import ctypes

    from OpenGL import EGL

    device_t = ctypes.c_void_p
    EGL_EXTENSIONS = 0x3055
    query_devices = ctypes.CFUNCTYPE(
        ctypes.c_uint, ctypes.c_int, ctypes.POINTER(device_t),
        ctypes.POINTER(ctypes.c_int),
    )(EGL.eglGetProcAddress("eglQueryDevicesEXT"))
    query_string = ctypes.CFUNCTYPE(
        ctypes.c_char_p, device_t, ctypes.c_int
    )(EGL.eglGetProcAddress("eglQueryDeviceStringEXT"))
    count = ctypes.c_int()
    query_devices(0, None, ctypes.byref(count))
    devices = (device_t * count.value)()
    query_devices(count.value, devices, ctypes.byref(count))
    for i in range(count.value):
        extensions = query_string(devices[i], EGL_EXTENSIONS)
        if extensions and b"software" in extensions:
            return i
    return None


# --gpu pins BOTH inference (CUDA) and rendering (EGL) to one nvidia-smi
# device. It must take effect before jax/mujoco initialize, hence this early
# argv scan instead of argparse (which runs after the imports below).
if "--gpu" in sys.argv[:-1]:
    _gpu = sys.argv[sys.argv.index("--gpu") + 1]
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    # Query the EGL mapping BEFORE restricting CUDA visibility: the driver
    # reports EGL_CUDA_DEVICE_NV in terms of the currently visible devices,
    # so setting CUDA_VISIBLE_DEVICES first would renumber them.
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(_egl_device_for_cuda(int(_gpu))))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", _gpu)

# An EXPLICIT --platform cpu means: leave the GPUs alone entirely. Inference
# runs on CPU via the env var above; rendering goes to Mesa's software EGL
# device (llvmpipe). The automatic CPU fallback for jaxlib-unsupported GPUs
# keeps rendering on the GPU, which works fine there.
if (_requested_platform == "cpu" and "--gpu" not in sys.argv
        and "MUJOCO_EGL_DEVICE_ID" not in os.environ):
    _soft = _egl_software_device()
    if _soft is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(_soft)
        # llvmpipe saturates around 2-4 threads at 256px with shadows off.
        os.environ.setdefault("LP_NUM_THREADS", "2")
    else:
        print("collect_2023: no software EGL device found; rendering stays on GPU")

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from air_hockey_challenge.utils.kinematics import forward_kinematics
from air_hockey_challenge.utils.tournament_agent_wrapper import (
    SimpleTournamentAgentWrapper,
)
from tqdm import tqdm

from eval_2023 import MODELS, make_agent


# Keep the stock total free width around the puck.  In other words, the goal
# opening is always ``2 * puck_radius + GOAL_CLEARANCE``.  The value is derived
# from the unmodified tournament table (0.25 m opening, 31.65 mm puck radius),
# so default collection remains exactly the stock geometry.
DEFAULT_PUCK_RADIUS = 0.03165
DEFAULT_GOAL_WIDTH = 0.25
GOAL_CLEARANCE = DEFAULT_GOAL_WIDTH - 2.0 * DEFAULT_PUCK_RADIUS

# The table model already contains two unused world sites (``puck_vis`` and
# ``puck_vis_rot``).  The collector repurposes them as an asymmetric, dark-red
# orientation cross.  These are absolute metre dimensions, deliberately not a
# fraction of the puck radius: a 10 cm puck and a stock 3.165 cm puck get the
# same readable marker in the recorded image.
DEFAULT_ORIENTATION_MARKER_ARM_LENGTH = 0.080
DEFAULT_ORIENTATION_MARKER_STROKE_WIDTH = 0.024
DEFAULT_ORIENTATION_MARKER_HEIGHT = 0.0015
# The puck visual site's top surface is at z=8 mm.  Keep the thin marker just
# above it so it cannot depth-fight with the puck when rendered top-down.
ORIENTATION_MARKER_Z = 0.010
ORIENTATION_MARKER_RGBA = (0.12, 0.0, 0.0, 1.0)


class EpisodeBuffer:
    """Accumulates one episode; index 0 holds the post-reset state."""

    def __init__(self, frame, obs, score, faults):
        self.images = [frame]
        self.obs = [obs]
        self.actions_xy = [np.zeros((2, 2), dtype=np.float32)]
        self.actions_joints = [np.zeros((2, 2, 7), dtype=np.float32)]
        self.scores = [score]
        self.faults = [faults]

    def append(self, frame, obs, action_xy, action_joints, score, faults):
        self.images.append(frame)
        self.obs.append(obs)
        self.actions_xy.append(action_xy)
        self.actions_joints.append(action_joints)
        self.scores.append(score)
        self.faults.append(faults)

    @property
    def n_actions(self):
        return len(self.actions_xy) - 1

    def save(self, path, terminal):
        n = len(self.images)
        is_first = np.zeros(n, dtype=bool)
        is_first[0] = True
        is_last = np.zeros(n, dtype=bool)
        is_last[-1] = True
        is_terminal = np.zeros(n, dtype=bool)
        is_terminal[-1] = terminal
        np.savez_compressed(
            path,
            image=np.stack(self.images),
            action=np.stack(self.actions_xy).astype(np.float32),
            action_joints=np.stack(self.actions_joints).astype(np.float32),
            obs=np.stack(self.obs).astype(np.float32),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
            score=np.asarray(self.scores, dtype=np.int32),
            faults=np.asarray(self.faults, dtype=np.int32),
        )


def make_command_to_xy(env_info):
    """Map a joint-position command to the commanded mallet x,y (world frame).

    Forward kinematics of the commanded joints gives the mallet target in the
    robot's base frame; the base transform brings both agents into the one
    world frame the camera sees (agent 2's own frame is rotated 180 degrees).
    """
    robot_model = env_info["robot"]["robot_model"]
    robot_data = env_info["robot"]["robot_data"]
    base_frames = env_info["robot"]["base_frame"]

    def command_to_xy(joint_pos_cmd, agent_idx):
        pos, _ = forward_kinematics(robot_model, robot_data, joint_pos_cmd)
        return (base_frames[agent_idx] @ np.append(pos, 1.0))[:2]

    return command_to_xy


def update_tournament_hit_range(mdp):
    """Keep tournament puck reset positions clear of the real colliders."""
    base_env = mdp.base_env
    if not hasattr(base_env, "hit_range"):
        return

    model = base_env._model
    puck_radius = float(model.geom("puck").size[0])
    mallet_radii = np.array(
        [model.geom(f"iiwa_{agent_idx}/ee").size[0] for agent_idx in (1, 2)]
    )
    if not np.isclose(mallet_radii[0], mallet_radii[1]):
        raise RuntimeError("tournament mallet radii must match")
    hit_width = mdp.env_info["table"]["width"] / 2 - puck_radius - 2 * mallet_radii[0]
    if hit_width <= 0:
        raise ValueError(
            "puck and mallet radii leave no valid tournament puck-reset width: "
            f"puck={puck_radius}, mallet={mallet_radii[0]}"
        )
    base_env.hit_range[1] = (-hit_width, hit_width)


def set_puck_radius(mdp, radius):
    """Resize the puck in the compiled model (collision geom + visual sites).

    Mass and inertia scale as for a same-material, same-thickness disc
    (mass ~ r^2, Izz = m r^2 / 2), keeping the dynamics self-consistent.
    env_info is deliberately left at the trained constants: the policy's
    obs normalization is part of its training contract. Large deviations
    from the trained 0.03165 m remain out-of-distribution physics for the
    pretrained policies, so expect degraded play, not identical games.
    """
    model = mdp.base_env._model
    puck_geom = model.geom("puck")
    scale = radius / puck_geom.size[0]
    puck_geom.size[0] = radius
    # Collision culling uses the precompiled bounding radius AND per-geom
    # AABB; without updating both, wall contacts engage only after deep
    # penetration and the solver ejects the puck at hundreds of m/s.
    model.geom_rbound[puck_geom.id] = float(np.hypot(radius, puck_geom.size[1]))
    model.geom_aabb[puck_geom.id][:3] = 0.0
    model.geom_aabb[puck_geom.id][3:] = (radius, radius, puck_geom.size[1])
    body = model.body("puck")
    # MuJoCo >= 2.3.6 also prunes collisions with a static BVH whose leaf
    # AABBs are compile-time copies of the geom AABBs. Mesh (short-end rim)
    # and mallet contact pairs go through it; without this update they
    # engage (r - default_r) late and the solver ejects the puck at
    # hundreds of m/s (box side rims use the analytic collider and were
    # unaffected — hence "only the players' sides misbehave").
    leaf = model.body_bvhadr[body.id]
    model.bvh_aabb[leaf][:3] = model.geom_pos[puck_geom.id]
    model.bvh_aabb[leaf][3:] = (radius, radius, puck_geom.size[1])
    height = 2.0 * puck_geom.size[1]
    body.mass[0] *= scale**2
    body.inertia[0] = body.inertia[1] = body.mass[0] * (3 * radius**2 + height**2) / 12
    body.inertia[2] = 0.5 * body.mass[0] * radius**2
    # Recompute mass-derived solver constants (invweight0 etc.); after this
    # the mutated model matches a recompiled model with the new size exactly.
    mujoco.mj_setConst(model, mdp.base_env._data)
    update_tournament_hit_range(mdp)
    puck_site = model.site("puck_site")
    puck_site.size[0] = radius
    body_id = body.id
    for site_id in range(model.nsite):
        # The unnamed rotation-indicator dot: keep it inside the disc.
        if model.site_bodyid[site_id] == body_id and site_id != puck_site.id:
            model.site_pos[site_id][0] *= scale


def configure_puck_orientation_marker(mdp, arm_length, stroke_width):
    """Configure a fixed-world-size, asymmetric red cross over the puck.

    The puck itself is a bright-red visual site, so the marker uses dark red
    for contrast.  A long fore/aft stroke plus a shorter crossbar shifted
    toward the forward end conveys yaw; a centred symmetric ``+`` would be
    ambiguous.  Both sites live in the world body and are updated from the
    puck's x/y/yaw before every rendered frame.
    """
    arm_length = float(arm_length)
    stroke_width = float(stroke_width)
    if arm_length <= 0 or stroke_width <= 0:
        raise ValueError(
            "orientation marker dimensions must be positive, got "
            f"arm_length={arm_length}, stroke_width={stroke_width}"
        )
    if stroke_width >= arm_length:
        raise ValueError(
            "orientation marker stroke width must be smaller than its arm length"
        )

    model = mdp.base_env._model
    marker_ids = (model.site("puck_vis").id, model.site("puck_vis_rot").id)
    for site_id in marker_ids:
        model.site_type[site_id] = mujoco.mjtGeom.mjGEOM_BOX
        # Avoid an XML material overriding the marker's deliberately dark-red
        # RGBA. The two reserved sites currently have no material, but this
        # makes the runtime mutation robust to a future asset update.
        model.site_matid[site_id] = -1
        model.site_rgba[site_id] = ORIENTATION_MARKER_RGBA

    mdp._puck_orientation_marker = {
        "puck_body_id": model.body("puck").id,
        "puck_yaw_qposadr": int(model.jnt_qposadr[model.joint("puck_yaw").id]),
        "long_bar_site": marker_ids[0],
        "cross_bar_site": marker_ids[1],
        "arm_length": arm_length,
        "stroke_width": stroke_width,
        "hidden": False,
    }
    update_puck_orientation_marker(mdp)


def update_puck_orientation_marker(mdp):
    """Place the fixed-size marker at the puck's current world pose."""
    marker = getattr(mdp, "_puck_orientation_marker", None)
    if marker is None or marker["hidden"]:
        return

    model = mdp.base_env._model
    data = mdp.base_env._data
    puck_pos = data.xpos[marker["puck_body_id"]]
    yaw = float(data.qpos[marker["puck_yaw_qposadr"]])
    forward = np.array((np.cos(yaw), np.sin(yaw)))
    rotation = np.array(
        (
            (np.cos(yaw), -np.sin(yaw), 0.0),
            (np.sin(yaw), np.cos(yaw), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    arm_length = marker["arm_length"]
    stroke_width = marker["stroke_width"]
    half_height = DEFAULT_ORIENTATION_MARKER_HEIGHT / 2.0

    # The first bar is centred. The second, shorter bar sits forward of
    # centre, so the otherwise cross-shaped cue has a unique heading.  It is
    # intentionally long enough to remain a readable cross at 128 px.
    long_bar = marker["long_bar_site"]
    cross_bar = marker["cross_bar_site"]
    data.site_xpos[long_bar] = (puck_pos[0], puck_pos[1], ORIENTATION_MARKER_Z)
    data.site_xmat[long_bar] = rotation.ravel()
    model.site_size[long_bar] = (arm_length / 2.0, stroke_width / 2.0, half_height)
    data.site_xpos[cross_bar] = (
        puck_pos[0] + forward[0] * arm_length * 0.13,
        puck_pos[1] + forward[1] * arm_length * 0.13,
        ORIENTATION_MARKER_Z,
    )
    data.site_xmat[cross_bar] = rotation.ravel()
    model.site_size[cross_bar] = (
        stroke_width / 2.0,
        arm_length * 0.65,
        half_height,
    )


def hide_puck_visuals(mdp):
    """Make all rendered puck elements disappear without removing its state."""
    model = mdp.base_env._model
    puck_body_id = model.body("puck").id
    for site_id in range(model.nsite):
        if model.site_bodyid[site_id] == puck_body_id:
            # Size zero is robust even for sites whose XML material overrides
            # their RGBA field (the puck's bright-red visual disc does).
            model.site_size[site_id] = 0.0

    marker = getattr(mdp, "_puck_orientation_marker", None)
    if marker is not None:
        marker["hidden"] = True
        for site_id in (marker["long_bar_site"], marker["cross_bar_site"]):
            model.site_size[site_id] = 0.0


class FixedLengthEventController:
    """Collector-local, nonterminating tournament event controller.

    ``AirHockeyTournament.is_absorbing`` mutates scores/faults when it returns
    ``True``. This controller replaces that method only for collection so a
    game always reaches its fixed requested length. A *goal* is scored once,
    then its puck is kept at the goal coordinate but hidden and deactivated.
    Stuck, timeout, and edge/escape conditions stay visible and keep
    simulating; they are merely recorded as metadata events. The benchmark
    environment itself remains unchanged.
    """

    def __init__(self, mdp):
        self.mdp = mdp
        self.base_env = mdp.base_env
        self.events = []
        self.pending_goal = None
        self.active_non_goal_event = None
        self.puck_hidden = False
        self.step_index = -1
        self._original_is_absorbing = self.base_env.is_absorbing

    def install(self):
        self.base_env.is_absorbing = self.is_absorbing

    def restore(self):
        self.base_env.is_absorbing = self._original_is_absorbing

    def _event(self, kind):
        event = {"kind": kind, "step": int(self.step_index)}
        self.events.append(event)
        if kind.startswith("goal_"):
            self.pending_goal = event

    def _classify_event(self, obs):
        """Classify one event while preserving goal/fault bookkeeping."""
        base = self.base_env
        puck_pos, puck_vel = base.get_puck(obs)

        # Goals take priority over an expired side timer. A real scored puck
        # must be the only condition that triggers the hidden/inert tail.
        goal_width = base.env_info["table"]["goal_width"]
        if abs(puck_pos[1]) <= goal_width / 2.0:
            if puck_pos[0] > base.env_info["table"]["length"] / 2.0:
                base.score[0] += 1
                base.start_side = -1
                return "goal_player_1"
            if puck_pos[0] < -base.env_info["table"]["length"] / 2.0:
                base.score[1] += 1
                base.start_side = 1
                return "goal_player_2"

        # Keep the tournament's side-stall timer/accounting semantics, but
        # reset its timer after a fault because this collection does not reset
        # the environment. That prevents the same stationary puck from
        # accruing a fault on every later transition.
        if np.sign(puck_pos[0]) == base.prev_side:
            base.timer += base.dt
        else:
            base.prev_side *= -1
            base.timer = 0.0

        if base.timer > 15.0 and abs(puck_pos[0]) >= 0.15:
            if base.prev_side == -1:
                base.faults[0] += 1
                base.start_side = -1
                if base.faults[0] % 3 == 0:
                    base.score[1] += 1
                kind = "timeout_player_1"
            else:
                base.faults[1] += 1
                base.start_side = 1
                if base.faults[1] % 3 == 0:
                    base.score[0] += 1
                kind = "timeout_player_2"
            base.timer = 0.0
            return kind

        # A puck at centre with no meaningful planar motion cannot be hit by
        # either policy. Unlike the original code, inspect both x/y velocity
        # components; yaw spin alone should not keep a dead puck alive.
        if abs(puck_pos[0]) < 0.15 and np.linalg.norm(puck_vel[:2]) < 0.025:
            return "center_stuck"

        boundary = np.array(
            (base.env_info["table"]["length"], base.env_info["table"]["width"])
        ) / 2.0
        if (
            np.any(np.abs(puck_pos[:2]) > boundary + 0.01)
            or np.linalg.norm(puck_vel[:2]) > 100.0
        ):
            return "escape_or_invalid_speed"
        return None

    def is_absorbing(self, obs):
        """Record an event but never terminate the fixed-length game."""
        # Some framework paths may query absorption more than once around a
        # transition. A pending *goal* must not score twice before the frame
        # is recorded and the puck is hidden.
        if self.puck_hidden or self.pending_goal is not None:
            return False
        kind = self._classify_event(obs)
        if kind is None:
            self.active_non_goal_event = None
        elif kind.startswith("goal_"):
            self._event(kind)
        elif kind != self.active_non_goal_event:
            # Edge, speed, and stuck conditions remain in the image and may
            # naturally resolve. Record the onset once, then allow a future
            # onset after the condition clears.
            self._event(kind)
            self.active_non_goal_event = kind
        return False

    def hide_pending_puck(self):
        """Hide only a scored puck after its final visible frame is saved."""
        if self.pending_goal is None or self.puck_hidden:
            return False

        model = self.base_env._model
        data = self.base_env._data
        puck_geom = model.geom("puck")
        puck_geom.contype = 0
        puck_geom.conaffinity = 0
        for joint_name in ("puck_x", "puck_y", "puck_yaw"):
            data.joint(joint_name).qvel = 0.0
        hide_puck_visuals(self.mdp)
        # Keep the puck at its scored coordinate, but make it physically inert
        # and invisible for the remaining fixed-length tail.
        mujoco.mj_forward(model, data)
        self.puck_hidden = True
        self.pending_goal = None
        return True

    @property
    def event_counts(self):
        counts = {}
        for event in self.events:
            counts[event["kind"]] = counts.get(event["kind"], 0) + 1
        return counts


def set_goal_opening_for_puck(mdp):
    """Make the physical, scored, and rendered goal mouth track the puck.

    The tournament scorer uses ``env_info['table']['goal_width']``, while the
    short end rims are four compiled mesh geoms.  Changing only the former
    creates a visual/scoring goal that the puck cannot actually enter.  Rather
    than mutating shared mesh vertices and their compiled triangle BVH, replace
    those four end segments with equivalent box colliders at runtime.  Their
    static-body BVH is then made conservative so MuJoCo cannot prune a real
    puck--rim contact after the geometry changes.

    ``GOAL_CLEARANCE`` is the *total* extra opening beyond the puck diameter:
    the stock 0.25 m mouth is preserved for the stock 0.03165 m-radius puck.
    """
    base_env = mdp.base_env
    model = base_env._model
    puck_radius = float(model.geom("puck").size[0])
    goal_width = 2.0 * puck_radius + GOAL_CLEARANCE

    if puck_radius <= 0:
        raise ValueError(f"puck radius must be positive, got {puck_radius}")
    if goal_width <= 2.0 * puck_radius:
        raise ValueError(
            f"goal width {goal_width} leaves no clearance for puck radius {puck_radius}"
        )

    rim_names = ("rim_home_l", "rim_home_r", "rim_away_l", "rim_away_r")
    rim_ids = [model.geom(name).id for name in rim_names]
    # The mesh rims reach from their centre to the table's outside edge.  All
    # four must agree: an asset change should fail loudly instead of producing
    # an asymmetric goal.
    rim_types = {int(model.geom_type[geom_id]) for geom_id in rim_ids}
    if rim_types == {int(mujoco.mjtGeom.mjGEOM_MESH)}:
        # The authored mesh's local-z axis maps to table-y.
        outer_edges = [
            abs(float(model.geom_pos[geom_id, 1])) + float(model.geom_aabb[geom_id, 5])
            for geom_id in rim_ids
        ]
    elif rim_types == {int(mujoco.mjtGeom.mjGEOM_BOX)}:
        # Subsequent calls operate on the runtime boxes, whose local-y axis
        # is table-y.
        outer_edges = [
            abs(float(model.geom_pos[geom_id, 1])) + float(model.geom_size[geom_id, 1])
            for geom_id in rim_ids
        ]
    else:
        raise RuntimeError(f"end rims have unexpected mixed geom types: {rim_types}")
    outer_edge = float(np.mean(outer_edges))
    if not np.allclose(outer_edges, outer_edge, rtol=0.0, atol=1e-6):
        raise RuntimeError(f"end-rim outer edges disagree: {outer_edges}")

    # Each end has two equal rail segments outside the new mouth.  Keep a
    # nonzero segment on each side; otherwise a malformed large puck could
    # remove the physical end wall entirely.
    segment_half_length = (outer_edge - goal_width / 2.0) / 2.0
    if segment_half_length <= 0.005:
        raise ValueError(
            f"goal width {goal_width:.4f} m is too wide for the {2 * outer_edge:.4f} m "
            "table end"
        )

    current_goal_width = float(mdp.env_info["table"]["goal_width"])
    mdp.env_info["table"]["goal_width"] = goal_width
    # With stock geometry, leave the authored meshes untouched.  This avoids
    # changing default collection in the common no-override case.
    if np.isclose(goal_width, current_goal_width, rtol=0.0, atol=1e-9):
        return goal_width

    # The mesh's world-space bounds are a 45 mm x (2*0.197 m) x 20 mm box.
    # Use the same dimensions and material/contact settings already assigned
    # to each geom, only replacing its shape and placement along table-y.
    rim_size = np.array((0.045, segment_half_length, 0.01), dtype=float)
    for geom_id in rim_ids:
        side = np.sign(model.geom_pos[geom_id, 1])
        if side == 0:
            raise RuntimeError(f"end rim {model.geom(geom_id).name} has no y-side")
        model.geom_type[geom_id] = mujoco.mjtGeom.mjGEOM_BOX
        model.geom_dataid[geom_id] = -1
        model.geom_size[geom_id] = rim_size
        model.geom_pos[geom_id, 1] = side * (goal_width / 2.0 + segment_half_length)
        model.geom_pos[geom_id, 2] = 0.01
        model.geom_quat[geom_id] = (1.0, 0.0, 0.0, 0.0)

    for marker_name in ("goal_marker_home", "goal_marker_away"):
        marker = model.geom(marker_name)
        marker.size[1] = goal_width / 2.0

    # Refresh derived constants for the new analytic boxes, then explicitly
    # refresh their culling bounds. mj_setConst does not rebuild a static
    # body's BVH after runtime geometry edits.
    mujoco.mj_setConst(model, base_env._data)
    for geom_id in rim_ids:
        model.geom_aabb[geom_id, :3] = 0.0
        model.geom_aabb[geom_id, 3:] = rim_size
        model.geom_rbound[geom_id] = float(np.linalg.norm(rim_size))
    for marker_name in ("goal_marker_home", "goal_marker_away"):
        marker = model.geom(marker_name)
        model.geom_aabb[marker.id, :3] = 0.0
        model.geom_aabb[marker.id, 3:] = marker.size
        model.geom_rbound[marker.id] = float(np.linalg.norm(marker.size))

    rim_body = model.body("rim")
    bvh_start = int(model.body_bvhadr[rim_body.id])
    bvh_num = int(model.body_bvhnum[rim_body.id])
    if bvh_start < 0 or not bvh_num:
        raise RuntimeError("table rim has no static-body BVH to refresh")
    body_geom_ids = np.flatnonzero(model.geom_bodyid == rim_body.id)
    # A conservative per-node extent is intentional: it prevents false
    # negatives without relying on MuJoCo's internal mesh-BVH layout.  This is
    # only the table's handful of static rim geoms, so the extra broad-phase
    # work is negligible next to rendering and physics.
    local_centres = model.geom_pos[body_geom_ids] - rim_body.ipos
    extent = np.max(
        np.abs(local_centres) + model.geom_rbound[body_geom_ids, None], axis=0
    )
    model.bvh_aabb[bvh_start:bvh_start + bvh_num, :3] = 0.0
    model.bvh_aabb[bvh_start:bvh_start + bvh_num, 3:] = extent
    mujoco.mj_forward(model, base_env._data)
    return goal_width


def _scale_mallet_mesh_radially(vertices, scale, axial_direction):
    """Increase a foam mallet's radial footprint without changing its height."""
    center = vertices.mean(axis=0)
    axis = np.asarray(axial_direction, dtype=float)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("axial_direction must be one finite 3-vector")
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("axial_direction must be nonzero")
    axis = axis / norm
    # The foam asset uses a different authored axis convention than the
    # cylinder. Preserve displacement along the cylinder axis and scale only
    # the plane perpendicular to it, irrespective of mesh coordinates.
    radial_scale = scale * np.eye(3) + (1.0 - scale) * np.outer(axis, axis)
    vertices[:] = center + (vertices - center) @ radial_scale.T


def set_mallet_radius(mdp, radius):
    """Resize the physical IIWA mallets and keep policy-facing bounds aligned.

    The ``iiwa_*/ee`` cylinders are the active puck-contact colliders.  Their
    visual foam meshes are separate, so both are resized together.  The mallet
    bodies are position-controlled and have explicitly authored inertias; those
    inertias deliberately stay unchanged to preserve the pretrained policy's
    closed-loop behavior while making the contact footprint genuinely larger.
    """
    if radius <= 0:
        raise ValueError(f"mallet radius must be positive, got {radius}")

    base_env = mdp.base_env
    model = base_env._model
    old_radius = float(mdp.env_info["mallet"]["radius"])
    if np.isclose(radius, old_radius):
        return
    scale = radius / old_radius
    mallet_mesh_axes = {}
    # Resolve the cylinder axis in each foam mesh's own coordinates. The
    # model's dynamic poses are valid here and the relative direction is
    # invariant under later robot motion.
    mujoco.mj_fwdPosition(model, base_env._data)

    for agent_idx in (1, 2):
        geom = model.geom(f"iiwa_{agent_idx}/ee")
        old_geom_radius = float(geom.size[0])
        geom.size[0] = radius
        model.geom_rbound[geom.id] = float(np.hypot(radius, geom.size[1]))
        # ``geom_aabb`` is expressed in the geom's own frame.  The geom
        # offset is applied separately by MuJoCo, so its centre must stay at
        # zero (not at ``geom_pos``).
        model.geom_aabb[geom.id][:3] = 0.0
        model.geom_aabb[geom.id][3:] = (radius, radius, geom.size[1])

        body = model.body(f"iiwa_{agent_idx}/striker_mallet")
        # A striker has a two-node per-body BVH (root + geom leaf).  Both
        # nodes need the new extent or the broad phase can discard a real
        # puck--mallet collision.  BVHs are expressed relative to the body's
        # inertial frame, hence the subtraction of ``body.ipos``.
        bvh_start = int(model.body_bvhadr[body.id])
        bvh_num = int(model.body_bvhnum[body.id])
        if bvh_start >= 0 and bvh_num:
            bvh_geom_ids = model.bvh_geomid[bvh_start:bvh_start + bvh_num]
            if np.any((bvh_geom_ids >= 0) & (bvh_geom_ids != geom.id)):
                raise RuntimeError(
                    f"iiwa_{agent_idx} mallet BVH contains an unexpected collider"
                )
            bvh_center = model.geom_pos[geom.id] - body.ipos
            model.bvh_aabb[bvh_start:bvh_start + bvh_num, :3] = bvh_center
            model.bvh_aabb[bvh_start:bvh_start + bvh_num, 3:] = (
                radius,
                radius,
                geom.size[1],
            )

        # Find the visual foam mesh on the same body and scale its horizontal
        # footprint to match the real collider. It is shared by both robots,
        # hence the set.
        collider_axis_world = base_env._data.geom_xmat[geom.id].reshape(3, 3)[:, 2]
        for geom_id in range(model.ngeom):
            if (
                model.geom_bodyid[geom_id] == body.id
                and model.geom_dataid[geom_id] >= 0
                and model.geom_contype[geom_id] == 0
                and model.geom_conaffinity[geom_id] == 0
            ):
                mesh_id = model.geom_dataid[geom_id]
                mesh_rotation = base_env._data.geom_xmat[geom_id].reshape(3, 3)
                axis_in_mesh = mesh_rotation.T @ collider_axis_world
                previous_axis = mallet_mesh_axes.setdefault(mesh_id, axis_in_mesh)
                if not np.isclose(abs(np.dot(previous_axis, axis_in_mesh)), 1.0, atol=1e-6):
                    raise RuntimeError(
                        "shared mallet foam mesh has incompatible cylinder axes"
                    )

        # A defensive assertion catches a future XML change where the visual
        # and physical mallet no longer share their intended nominal radius.
        if not np.isclose(old_geom_radius, old_radius):
            raise RuntimeError(
                f"iiwa_{agent_idx}/ee radius {old_geom_radius} does not match "
                f"env_info mallet radius {old_radius}"
            )

    for mesh_id, axis_in_mesh in mallet_mesh_axes.items():
        start = model.mesh_vertadr[mesh_id]
        end = start + model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[start:end]
        # The physical cylinder keeps its authored height when only its
        # radius changes. Match that geometry: enlarge the foam footprint in
        # its local XY plane, but preserve Z so the visual mallet stays flush
        # with the table instead of visibly growing through it.
        _scale_mallet_mesh_radially(vertices, scale, axis_in_mesh)

    # The policy constructors run after build_mdp(), so this makes their
    # operating boxes match the enlarged real collider.  Keep the tournament
    # puck-reset range and existing end-effector constraint in sync as well.
    mdp.env_info["mallet"]["radius"] = radius
    update_tournament_hit_range(mdp)
    ee_constraint = mdp.env_info.get("constraints", {}).get("ee_constr")
    if ee_constraint is not None:
        ee_constraint.x_lb = -mdp.env_info["robot"]["base_frame"][0][0, 3] - (
            mdp.env_info["table"]["length"] / 2 - radius
        )
        ee_constraint.y_lb = -(mdp.env_info["table"]["width"] / 2 - radius)
        ee_constraint.y_ub = mdp.env_info["table"]["width"] / 2 - radius

    # Refresh mass-derived solver constants after changing collision geometry.
    # Explicit mallet inertias remain intact (see docstring).
    mujoco.mj_setConst(model, base_env._data)


def set_robot_visual_scale(mdp, scale):
    """Scale only the rendered IIWA meshes, leaving simulation state intact.

    The robot link meshes are visual-only (``contype=conaffinity=0``)
    in the tournament model.  Enlarging their vertices before the viewer is
    created increases their pixel footprint without changing collisions,
    inertias, kinematics, policies, or the action labels.  Mesh assets are
    shared by both robots, so each asset is transformed once around its own
    centroid.
    """
    if scale <= 0:
        raise ValueError(f"robot visual scale must be positive, got {scale}")
    if np.isclose(scale, 1.0):
        return

    model = mdp.base_env._model
    mesh_ids = set()
    for geom_id in range(model.ngeom):
        body_name = model.body(model.geom_bodyid[geom_id]).name or ""
        mesh_id = model.geom_dataid[geom_id]
        if (
            body_name.startswith("iiwa_")
            and "/striker_mallet" not in body_name
            and mesh_id >= 0
            and model.geom_contype[geom_id] == 0
            and model.geom_conaffinity[geom_id] == 0
        ):
            mesh_ids.add(mesh_id)

    if not mesh_ids:
        raise RuntimeError("no visual-only IIWA meshes found to scale")
    for mesh_id in mesh_ids:
        start = model.mesh_vertadr[mesh_id]
        end = start + model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[start:end]
        center = vertices.mean(axis=0)
        vertices[:] = center + scale * (vertices - center)


MALLET_DOWN_WORLD = np.array((0.0, 0.0, -1.0))


def _solve_universal_level_angles(down_in_link_frame):
    """Solve the two passive striker joints that make the mallet level.

    The XML declares the joints in local-y then local-x order.  For a desired
    downward mallet normal ``d`` expressed in the striker-link frame,
    ``R_y(q1) R_x(q2) [0, 0, 1] == d``.  The tournament's stock plugin only
    PD-controls an approximation to this target; direct projection is needed
    because the mallet/table collision pair is intentionally disabled.
    """
    down = np.asarray(down_in_link_frame, dtype=float)
    if down.shape != (3,) or not np.all(np.isfinite(down)):
        raise ValueError("down_in_link_frame must be one finite 3-vector")
    norm = np.linalg.norm(down)
    if norm == 0.0:
        raise ValueError("down_in_link_frame must be nonzero")
    down = down / norm
    return np.array(
        (
            np.arctan2(down[0], down[2]),
            -np.arcsin(np.clip(down[1], -1.0, 1.0)),
        )
    )


class HardMalletLevelGuard:
    """Keep the passive IIWA striker joints exactly level during collection.

    Air Hockey Challenge normally drives these two universal joints with a
    weak PD plugin.  Under abrupt commands it can lag, letting a visual (and
    puck-contact) mallet tilt through the table because that collision pair is
    disabled.  This collector-local guard projects both joints at every 1 ms
    simulation boundary without touching the seven controlled arm joints or
    their recorded action labels.
    """

    def __init__(self, mdp):
        self.mdp = mdp
        self.base_env = mdp.base_env
        self.model = self.base_env._model
        self.data = self.base_env._data
        self._link_body_ids = []
        self._base_body_ids = []
        self._qpos_addrs = []
        self._dof_addrs = []
        self._actuator_ids = []
        self._joint_ranges = []
        self._arm_qpos_addrs = []
        self._arm_dof_addrs = []
        self._arm_joint_ranges = []
        for agent_idx in (1, 2):
            self._link_body_ids.append(
                self.model.body(f"iiwa_{agent_idx}/striker_joint_link").id
            )
            self._base_body_ids.append(self.model.body(f"iiwa_{agent_idx}/base").id)
            arm_qpos_addrs, arm_dof_addrs, arm_joint_ranges = [], [], []
            for joint_idx in range(1, 8):
                joint = self.model.joint(f"iiwa_{agent_idx}/joint_{joint_idx}")
                arm_qpos_addrs.append(int(self.model.jnt_qposadr[joint.id]))
                arm_dof_addrs.append(int(self.model.jnt_dofadr[joint.id]))
                arm_joint_ranges.append(self.model.jnt_range[joint.id].copy())
            self._arm_qpos_addrs.append(arm_qpos_addrs)
            self._arm_dof_addrs.append(arm_dof_addrs)
            self._arm_joint_ranges.append(arm_joint_ranges)
            for joint_idx in (1, 2):
                joint = self.model.joint(
                    f"iiwa_{agent_idx}/striker_joint_{joint_idx}"
                )
                self._qpos_addrs.append(int(self.model.jnt_qposadr[joint.id]))
                self._dof_addrs.append(int(self.model.jnt_dofadr[joint.id]))
                self._joint_ranges.append(self.model.jnt_range[joint.id].copy())
                self._actuator_ids.append(
                    self.model.actuator(
                        f"iiwa_{agent_idx}/striker_joint_{joint_idx}"
                    ).id
                )
        self._qpos_addrs = np.asarray(self._qpos_addrs, dtype=int)
        self._dof_addrs = np.asarray(self._dof_addrs, dtype=int)
        self._actuator_ids = np.asarray(self._actuator_ids, dtype=int)
        self._joint_ranges = np.asarray(self._joint_ranges, dtype=float)
        self._arm_qpos_addrs = np.asarray(self._arm_qpos_addrs, dtype=int)
        self._arm_dof_addrs = np.asarray(self._arm_dof_addrs, dtype=int)
        self._arm_joint_ranges = np.asarray(self._arm_joint_ranges, dtype=float)
        self._ee_desired_height = float(mdp.env_info["robot"]["ee_desired_height"])
        self._installed = False

    def _project_arm_height(self):
        """Restore each striker-link height while preserving its live XY."""
        jac_pos = np.empty((3, self.model.nv))
        jac_rot = np.empty((3, self.model.nv))
        for agent_idx, (base_body_id, link_body_id) in enumerate(
            zip(self._base_body_ids, self._link_body_ids)
        ):
            target = self.data.xpos[link_body_id].copy()
            target[2] = self.data.xpos[base_body_id, 2] + self._ee_desired_height
            qpos_addrs = self._arm_qpos_addrs[agent_idx]
            dof_addrs = self._arm_dof_addrs[agent_idx]
            lower = self._arm_joint_ranges[agent_idx, :, 0]
            upper = self._arm_joint_ranges[agent_idx, :, 1]
            # A small damped least-squares projection avoids changing the
            # current XY endpoint while recovering the policy's fixed height.
            for _ in range(6):
                current = self.data.xpos[link_body_id]
                error = target - current
                if np.linalg.norm(error) <= 1e-5:
                    break
                mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, link_body_id)
                jacobian = jac_pos[:, dof_addrs]
                delta = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + 1e-6 * np.eye(3), error
                )
                delta_norm = np.linalg.norm(delta)
                if delta_norm > 0.1:
                    delta *= 0.1 / delta_norm
                current_qpos = self.data.qpos[qpos_addrs]
                next_qpos = np.clip(current_qpos + delta, lower, upper)
                if np.array_equal(next_qpos, current_qpos):
                    break
                self.data.qpos[qpos_addrs] = next_qpos
                mujoco.mj_fwdPosition(self.model, self.data)

            height_error = target[2] - self.data.xpos[link_body_id, 2]
            if abs(height_error) > 1e-4:
                raise RuntimeError(
                    f"cannot restore iiwa_{agent_idx + 1} mallet height: "
                    f"error={height_error:.6f} m"
                )
            # Remove only the arm velocity component that would immediately
            # reintroduce vertical motion. This leaves horizontal play intact.
            mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, link_body_id)
            jacobian_z = jac_pos[2, dof_addrs]
            denominator = jacobian_z @ jacobian_z + 1e-6
            arm_velocity = self.data.qvel[dof_addrs]
            self.data.qvel[dof_addrs] = (
                arm_velocity
                - jacobian_z * (jacobian_z @ arm_velocity) / denominator
            )

    def project(self, refresh_kinematics=True, project_height=True):
        """Hard-set safe arm height and passive mallet orientation."""
        # ``mj_step`` can leave derived body transforms from the preceding
        # integration phase. Refresh before solving so the parent rod pose is
        # current; otherwise one projection can lag a moving arm slightly.
        mujoco.mj_fwdPosition(self.model, self.data)
        if project_height:
            self._project_arm_height()
            # Height projection changes the striker-link pose that anchors
            # the universal joints, so refresh before solving their angles.
            mujoco.mj_fwdPosition(self.model, self.data)
        angles = []
        for body_id in self._link_body_ids:
            link_rotation = self.data.xmat[body_id].reshape(3, 3)
            down_in_link = link_rotation.T @ MALLET_DOWN_WORLD
            angles.extend(_solve_universal_level_angles(down_in_link))
        angles = np.asarray(angles)
        lower, upper = self._joint_ranges[:, 0], self._joint_ranges[:, 1]
        if np.any(angles < lower - 1e-6) or np.any(angles > upper + 1e-6):
            raise RuntimeError(
                "cannot level a mallet without exceeding the universal-joint limits: "
                f"angles={angles.tolist()}, ranges={self._joint_ranges.tolist()}"
            )
        self.data.qpos[self._qpos_addrs] = np.clip(angles, lower, upper)
        self.data.qvel[self._dof_addrs] = 0.0
        # Disable the stock weak PD torque after it has updated.  The next
        # boundary projects again, so these joints cannot accumulate tilt.
        self.data.ctrl[self._actuator_ids] = 0.0
        if refresh_kinematics:
            mujoco.mj_fwdPosition(self.model, self.data)

    def install(self):
        """Wrap reset and 1 ms simulator hooks; safe to call once."""
        if self._installed:
            return
        self._original_pre_step = self.base_env._simulation_pre_step
        self._original_post_step = self.base_env._simulation_post_step
        self._original_setup = self.base_env.setup

        def pre_step():
            self._original_pre_step()
            # mj_step recomputes position before integration. Project now so
            # each 1 ms physics interval starts level; its own forward pass
            # will apply the new universal-joint coordinates. The previous
            # post-step already restored arm height.
            self.project(refresh_kinematics=False, project_height=False)

        def post_step():
            self._original_post_step()
            # Correct the arm's height and mallet level after every 1 ms
            # integration interval; the public observation is then built
            # from this same safe state after the final sub-step.
            self.project(refresh_kinematics=True, project_height=True)

        def setup(*args, **kwargs):
            result = self._original_setup(*args, **kwargs)
            self.project(refresh_kinematics=True, project_height=True)
            return result

        self.base_env._simulation_pre_step = pre_step
        self.base_env._simulation_post_step = post_step
        self.base_env.setup = setup
        self._installed = True

    def restore(self):
        """Restore the environment hooks before disposing the simulator."""
        if not self._installed:
            return
        self.base_env._simulation_pre_step = self._original_pre_step
        self.base_env._simulation_post_step = self._original_post_step
        self.base_env.setup = self._original_setup
        self._installed = False


def install_hard_mallet_level_guard(mdp):
    """Attach the mandatory collector safety guard and return it."""
    guard = HardMalletLevelGuard(mdp)
    guard.install()
    mdp._hard_mallet_level_guard = guard
    return guard


def build_mdp(args):
    mdp = AirHockeyChallengeWrapper(
        "tournament",
        interpolation_order=(
            3 if args.model1 == "baseline" else -1,
            3 if args.model2 == "baseline" else -1,
        ),
        viewer_params={
            "width": args.width,
            "height": args.height,
            "headless": True,
            "hide_menu_on_startup": True,
            # Top-down view with the table's long axis along the image width;
            # distance tuned so the table length (2.128 m incl. rims) nearly
            # fills the frame width at the default square 256x256.
            "camera_params": {
                "static": dict(
                    distance=2.66, elevation=-90.0, azimuth=90.0, lookat=(0.0, 0.0, 0.0)
                )
            },
            "default_camera_mode": "static",
        },
    )
    if not args.keep_scoreboard:
        # AirHockeyTournament injects a scoreboard-overlay render callback;
        # drop it so no text is baked into the training frames. The viewer is
        # created lazily on the first render(), so this is early enough.
        mdp.base_env._viewer_params["custom_render_callback"] = None
    if args.puck_radius is not None:
        set_puck_radius(mdp, args.puck_radius)
    set_goal_opening_for_puck(mdp)
    if args.mallet_radius is not None:
        set_mallet_radius(mdp, args.mallet_radius)
    set_robot_visual_scale(mdp, args.robot_visual_scale)
    configure_puck_orientation_marker(
        mdp,
        getattr(args, "orientation_marker_arm_length", DEFAULT_ORIENTATION_MARKER_ARM_LENGTH),
        getattr(args, "orientation_marker_stroke_width", DEFAULT_ORIENTATION_MARKER_STROKE_WIDTH),
    )
    install_hard_mallet_level_guard(mdp)
    return mdp


def close_mdp(mdp):
    """Stop the env and free its EGL context deterministically.

    mujoco.egl terminates the EGL display via atexit; MujocoViewer.stop()
    never frees headless GL contexts, so without this they get finalized
    after the display is gone and __del__ raises EGL_NOT_INITIALIZED
    ("Exception ignored" noise at interpreter shutdown).
    """
    guard = getattr(mdp, "_hard_mallet_level_guard", None)
    if guard is not None:
        guard.restore()
    viewer = mdp.base_env._viewer
    mdp.base_env.stop()
    if viewer is not None and getattr(viewer, "_opengl_context", None) is not None:
        viewer._opengl_context.free()


def open_video_writer(path, width, height, fps):
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(fps), "-i", "-",
         # One encoder thread is ample at this resolution and keeps N
         # parallel workers from spawning N * cores x264 threads.
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-preset", "fast", "-threads", "1", str(path)],
        stdin=subprocess.PIPE,
    )


def render_frame(mdp):
    """Render one collection frame after synchronizing the puck marker."""
    update_puck_orientation_marker(mdp)
    return mdp.render(record=True)


def reset_episode(mdp, agent, video, shadows):
    """Core.reset() semantics: episode_start before env reset."""
    agent.episode_start()
    state = mdp.reset()
    frame = render_frame(mdp)
    viewer = mdp.base_env._viewer
    if not shadows and viewer._scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]:
        # Off by default: llvmpipe (--platform cpu rendering) is ~6x slower
        # with shadow maps, and frames must look the same regardless of
        # which machine/renderer produced them. The viewer is created
        # lazily by the render() above, hence flag-flip + one re-render.
        viewer._scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        viewer._scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        frame = render_frame(mdp)
    video.stdin.write(frame.tobytes())
    buffer = EpisodeBuffer(
        frame, state, list(mdp.base_env.score), list(mdp.base_env.faults)
    )
    return state, buffer


def make_post_goal_random_agent(mdp, args):
    """Build fresh smooth-random players for the tail after a scored goal."""
    return SimpleTournamentAgentWrapper(
        mdp.env_info,
        make_agent(
            mdp.env_info,
            1,
            "smooth_random",
            idle_probability=args.idle_prob1,
            idle_min_steps=args.idle_min_steps,
            idle_max_steps=args.idle_max_steps,
        ),
        make_agent(
            mdp.env_info,
            2,
            "smooth_random",
            idle_probability=args.idle_prob2,
            idle_min_steps=args.idle_min_steps,
            idle_max_steps=args.idle_max_steps,
        ),
    )


def pause_statistics(agent):
    """Return one serializable pause summary per tournament player."""
    return [
        dict(getattr(agent.agent_1, "pause_stats", {})),
        dict(getattr(agent.agent_2, "pause_stats", {})),
    ]


def collect_game(game_dir, mdp, agent, args):
    game_dir.mkdir(parents=True)
    video = open_video_writer(game_dir / "video.mp4", args.width, args.height, args.fps)
    episode_lengths = []
    # npz compression is slow enough to stall the loop for seconds per
    # episode; zlib releases the GIL, so a single worker thread overlaps
    # saving with collection. Buffers are never touched after submission.
    writer = ThreadPoolExecutor(max_workers=1)
    saves = []
    start = time.time()

    command_to_xy = make_command_to_xy(mdp.env_info)
    episode_mode = getattr(args, "episode_mode", "fixed_length")
    fixed_length = episode_mode == "fixed_length"
    controller = FixedLengthEventController(mdp) if fixed_length else None
    active_agent = agent
    post_goal_agent = None
    post_goal_step = None
    if controller is not None:
        controller.install()

    # Both agent types have empty preprocessor lists, so Core._preprocess is
    # skipped here; revisit if agents ever register preprocessors.
    try:
        state, buffer = reset_episode(mdp, agent, video, args.shadows)
        for step in tqdm(range(args.steps), desc=game_dir.name, unit="step",
                         disable=args.no_progress):
            if controller is not None:
                controller.step_index = step
            action_1, action_2, _, _ = active_agent.draw_action(state)
            obs, _, absorbing, info = mdp.step((action_1, action_2))
            # Every event frame remains visible. Only a scored goal hides the
            # puck after this frame; edge and stuck states keep rendering.
            frame = render_frame(mdp)
            video.stdin.write(frame.tobytes())
            buffer.append(
                frame,
                obs,
                np.stack([command_to_xy(action_1[0], 0), command_to_xy(action_2[0], 1)]),
                np.stack([action_1, action_2]),
                list(info["score"]),
                list(info["faults"]),
            )
            if controller is not None and controller.hide_pending_puck():
                if getattr(args, "post_goal_policy", "smooth_random") == "smooth_random":
                    # Do not mutate ``agent``: it is reused for the next game
                    # and its wrapper caches episode-start bound methods.
                    post_goal_agent = make_post_goal_random_agent(mdp, args)
                    post_goal_agent.episode_start()
                    active_agent = post_goal_agent
                    post_goal_step = step + 1
            state = obs

            if not fixed_length and absorbing:
                saves.append(writer.submit(
                    buffer.save,
                    game_dir / f"episode_{len(episode_lengths):03d}.npz",
                    True,
                ))
                episode_lengths.append(buffer.n_actions)
                state, buffer = reset_episode(mdp, agent, video, args.shadows)

        # A fixed-length game always yields one episode with T actions and
        # T+1 state frames. It is a truncation, never an environment terminal.
        if buffer.n_actions > 0:
            saves.append(writer.submit(
                buffer.save,
                game_dir / f"episode_{len(episode_lengths):03d}.npz",
                False,
            ))
            episode_lengths.append(buffer.n_actions)
    finally:
        if controller is not None:
            controller.restore()

    video.stdin.close()
    video.wait()
    for save in saves:
        save.result()
    writer.shutdown()
    elapsed = time.time() - start

    import jax

    meta = {
        "model1": args.model1,
        "model2": args.model2,
        # ``--seed`` is a base. Persist the actual per-game seed so a raw
        # directory remains reproducible after multi-game or worker launches.
        "seed": int(getattr(args, "effective_seed", args.seed)),
        "seed_base": int(args.seed),
        "game_index": int(getattr(args, "game_index", 0)),
        "steps": args.steps,
        "episode_mode": episode_mode,
        "n_episodes": len(episode_lengths),
        "episode_lengths": episode_lengths,
        "final_score": list(mdp.base_env.score),
        "final_faults": list(mdp.base_env.faults),
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "puck_radius": float(mdp.base_env._model.geom("puck").size[0]),
        "goal_width": float(mdp.env_info["table"]["goal_width"]),
        "goal_clearance": GOAL_CLEARANCE,
        "mallet_radius": mdp.env_info["mallet"]["radius"],
        "robot_visual_scale": args.robot_visual_scale,
        "mallet_level_lock": args.mallet_level_lock,
        "orientation_marker": {
            "style": "asymmetric_forward_cross",
            "arm_length_m": args.orientation_marker_arm_length,
            "stroke_width_m": args.orientation_marker_stroke_width,
            "fixed_world_scale": True,
        },
        "pause_probabilities": [args.idle_prob1, args.idle_prob2],
        "pause_duration_steps": [args.idle_min_steps, args.idle_max_steps],
        "pause_statistics": {
            "initial_policy": pause_statistics(agent),
            "post_goal_random": (
                None if post_goal_agent is None else pause_statistics(post_goal_agent)
            ),
        },
        # This is the configured tail policy; ``post_goal_step`` records
        # whether a real goal activated it during this game.
        "post_goal_policy": getattr(args, "post_goal_policy", "smooth_random"),
        "post_goal_step": post_goal_step,
        "terminal_events": [] if controller is None else controller.events,
        "event_counts": {} if controller is None else controller.event_counts,
        "puck_hidden": False if controller is None else controller.puck_hidden,
        "shadows": args.shadows,
        "jax_backend": jax.default_backend(),
        "wall_time_s": round(elapsed, 1),
        "steps_per_s": round(args.steps / elapsed, 1),
    }
    # meta.json is the completion marker consumed by the YAML launcher and
    # converter. Publish it atomically only after video and episode archives
    # have finished writing, so interrupted games cannot enter the dataset.
    meta_path = game_dir / "meta.json"
    meta_tmp_path = meta_path.with_name(meta_path.name + ".tmp")
    meta_tmp_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    meta_tmp_path.replace(meta_path)
    return meta


def spawn_workers(args, run_dir):
    """Split games across worker subprocesses writing into one run dir."""
    # Spread the render contexts across all GPUs (unless the user pinned
    # one): dozens of EGL contexts on a single card lose real throughput
    # to GL context switching.
    egl_devices = []
    if args.gpu is None and "MUJOCO_EGL_DEVICE_ID" not in os.environ:
        try:
            egl_devices = [str(egl) for egl, _ in _egl_cuda_devices()]
        except Exception:
            pass

    base, extra = divmod(args.games, args.workers)
    procs = []
    game_start = 0
    for worker in range(args.workers):
        n_games = base + (1 if worker < extra else 0)
        if n_games == 0:
            break
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--model1", args.model1, "--model2", args.model2,
            "--games", str(n_games), "--steps", str(args.steps),
            "--width", str(args.width), "--height", str(args.height),
            "--fps", str(args.fps), "--seed", str(args.seed),
            "--episode-mode", args.episode_mode,
            "--idle-prob1", str(args.idle_prob1),
            "--idle-prob2", str(args.idle_prob2),
            "--idle-min-steps", str(args.idle_min_steps),
            "--idle-max-steps", str(args.idle_max_steps),
            "--post-goal-policy", args.post_goal_policy,
            "--mallet-level-lock", args.mallet_level_lock,
            "--orientation-marker-arm-length", str(args.orientation_marker_arm_length),
            "--orientation-marker-stroke-width", str(args.orientation_marker_stroke_width),
            "--run-dir", str(run_dir), "--game-start", str(game_start),
            "--no-progress",
        ]
        if args.keep_scoreboard:
            cmd.append("--keep-scoreboard")
        if args.shadows:
            cmd.append("--shadows")
        if args.puck_radius is not None:
            cmd += ["--puck-radius", str(args.puck_radius)]
        if args.mallet_radius is not None:
            cmd += ["--mallet-radius", str(args.mallet_radius)]
        if args.robot_visual_scale != 1.0:
            cmd += ["--robot-visual-scale", str(args.robot_visual_scale)]
        if args.platform:
            cmd += ["--platform", args.platform]
        if args.gpu is not None:
            cmd += ["--gpu", str(args.gpu)]
        env = dict(os.environ)
        # Keep each worker single-threaded so N workers don't oversubscribe
        # the box; MuJoCo physics dominates and is single-threaded anyway.
        # XLA's CPU backend otherwise spins up an all-cores Eigen pool in
        # every worker, and those pools thrash each other.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
        if egl_devices:
            env["MUJOCO_EGL_DEVICE_ID"] = egl_devices[worker % len(egl_devices)]
        procs.append(subprocess.Popen(cmd, env=env))
        game_start += n_games
    return all(proc.wait() == 0 for proc in procs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", default="tournament_balanced", choices=MODELS)
    parser.add_argument("--model2", default="tournament_balanced", choices=MODELS)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--steps", type=int, default=45000,
                        help="Actions per game; fixed_length writes one T-action episode")
    parser.add_argument(
        "--episode-mode",
        choices=("fixed_length", "split_on_absorbing"),
        default="fixed_length",
        help="fixed_length keeps one episode per game and hides only a scored puck; "
             "split_on_absorbing preserves the legacy reset-on-terminal behavior",
    )
    parser.add_argument("--out", default="data_2023")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fps", type=int, default=50, help="Video fps (env runs at 50 Hz)")
    parser.add_argument("--keep-scoreboard", action="store_true",
                        help="Keep the scoreboard overlay baked into the frames")
    parser.add_argument("--shadows", action="store_true",
                        help="Render shadows and reflections (off by default "
                             "so frames match across GPU and CPU renderers)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--puck-radius", type=float, default=None,
                        help="Puck radius in meters (model default: 0.03165); "
                        "resizes collision and visuals at runtime")
    parser.add_argument("--mallet-radius", type=float, default=None,
                        help="Physical IIWA mallet radius in meters; updates collision, "
                        "visual mallet, policy bounds, and tournament reset range")
    parser.add_argument("--robot-visual-scale", type=float, default=1.0,
                        help="Visual-only scale for IIWA arm meshes (not mallets); "
                             "does not change physics, actions, or kinematics")
    parser.add_argument(
        "--orientation-marker-arm-length",
        type=float,
        default=DEFAULT_ORIENTATION_MARKER_ARM_LENGTH,
        help="Absolute metre length of the puck's forward-cross marker; never "
             "scaled with puck radius",
    )
    parser.add_argument(
        "--orientation-marker-stroke-width",
        type=float,
        default=DEFAULT_ORIENTATION_MARKER_STROKE_WIDTH,
        help="Absolute metre stroke width of the puck orientation marker",
    )
    parser.add_argument(
        "--idle-prob1",
        type=float,
        default=0.0,
        help="Per-unpaused-step probability player 1 starts a short random pause",
    )
    parser.add_argument(
        "--idle-prob2",
        type=float,
        default=0.0,
        help="Per-unpaused-step probability player 2 starts a short random pause",
    )
    parser.add_argument(
        "--idle-min-steps",
        type=int,
        default=50,
        help="Minimum inclusive random pause length (50 = 1 s at 50 Hz)",
    )
    parser.add_argument(
        "--idle-max-steps",
        type=int,
        default=250,
        help="Maximum inclusive random pause length (250 = 5 s at 50 Hz)",
    )
    parser.add_argument(
        "--post-goal-policy",
        choices=("smooth_random", "keep"),
        default="smooth_random",
        help="Policy for the remaining tail after a real goal (default: smooth_random)",
    )
    parser.add_argument(
        "--mallet-level-lock",
        choices=("hard_level_height_projection",),
        default="hard_level_height_projection",
        help="Mandatory hard mallet level/height safety projection",
    )
    parser.add_argument("--gpu", type=int, default=None,
                        help="nvidia-smi GPU index to pin both inference (CUDA) "
                             "and rendering (EGL) to; default lets each library "
                             "pick its own device")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel collector processes; games are split "
                             "among them (rendering still uses the GPU)")
    parser.add_argument("--platform", choices=("gpu", "cpu"), default=None,
                        help="Device for agent inference (default: gpu)")
    # Internal args used by spawn_workers for its children.
    parser.add_argument("--run-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--game-start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--no-progress", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    for flag, probability in (("--idle-prob1", args.idle_prob1),
                              ("--idle-prob2", args.idle_prob2)):
        if not 0.0 <= probability <= 1.0:
            parser.error(f"{flag} must be in [0, 1]")
    if args.idle_min_steps <= 0 or args.idle_max_steps < args.idle_min_steps:
        parser.error("idle pause lengths must satisfy 0 < --idle-min-steps <= --idle-max-steps")

    run_dir = (
        Path(args.run_dir) if args.run_dir
        else Path(args.out) / f"collect-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    if args.workers > 1:
        run_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        ok = spawn_workers(args, run_dir)
        elapsed = time.time() - start
        total = args.games * args.steps
        print(
            f"{args.workers} workers: {total} steps in {elapsed:.0f}s "
            f"-> {total / elapsed:.1f} steps/s aggregate"
        )
        print(f"Data written to: {run_dir}")
        sys.exit(0 if ok else 1)

    mdp = build_mdp(args)
    # Agents are built once (checkpoint load + JIT warmup is expensive);
    # episode_start() fully resets their recurrent state between episodes.
    agent = SimpleTournamentAgentWrapper(
        mdp.env_info,
        make_agent(
            mdp.env_info, 1, args.model1, idle_probability=args.idle_prob1,
            idle_min_steps=args.idle_min_steps, idle_max_steps=args.idle_max_steps,
        ),
        make_agent(
            mdp.env_info, 2, args.model2, idle_probability=args.idle_prob2,
            idle_min_steps=args.idle_min_steps, idle_max_steps=args.idle_max_steps,
        ),
    )

    for i in range(args.games):
        game = args.game_start + i
        args.game_index = game
        args.effective_seed = args.seed + game
        np.random.seed(args.effective_seed)
        if i > 0:
            # score/faults persist on the env instance; rebuild per game.
            close_mdp(mdp)
            mdp = build_mdp(args)
        meta = collect_game(run_dir / f"game_{game:03d}", mdp, agent, args)
        print(
            f"game_{game:03d}: {meta['n_episodes']} episodes, "
            f"score {meta['final_score']}, {meta['steps_per_s']} steps/s",
            flush=True,
        )
    close_mdp(mdp)
    if not args.run_dir:
        print(f"Data written to: {run_dir}")


if __name__ == "__main__":
    main()
