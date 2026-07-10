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
    is_terminal bool    (T+1,)          is_last AND the episode ended in an
                                        absorbing state (goal/fault/stuck);
                                        False when truncated by --steps
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
if "--platform" in sys.argv[:-1]:
    os.environ.setdefault(
        "DRL_AIR_HOCKEY_JAX_PLATFORM", sys.argv[sys.argv.index("--platform") + 1]
    )


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
    puck_site = model.site("puck_site")
    puck_site.size[0] = radius
    body_id = body.id
    for site_id in range(model.nsite):
        # The unnamed rotation-indicator dot: keep it inside the disc.
        if model.site_bodyid[site_id] == body_id and site_id != puck_site.id:
            model.site_pos[site_id][0] *= scale


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
    return mdp


def close_mdp(mdp):
    """Stop the env and free its EGL context deterministically.

    mujoco.egl terminates the EGL display via atexit; MujocoViewer.stop()
    never frees headless GL contexts, so without this they get finalized
    after the display is gone and __del__ raises EGL_NOT_INITIALIZED
    ("Exception ignored" noise at interpreter shutdown).
    """
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


def reset_episode(mdp, agent, video):
    """Core.reset() semantics: episode_start before env reset."""
    agent.episode_start()
    state = mdp.reset()
    frame = mdp.render(record=True)
    video.stdin.write(frame.tobytes())
    buffer = EpisodeBuffer(
        frame, state, list(mdp.base_env.score), list(mdp.base_env.faults)
    )
    return state, buffer


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

    # Both agent types have empty preprocessor lists, so Core._preprocess is
    # skipped here; revisit if agents ever register preprocessors.
    state, buffer = reset_episode(mdp, agent, video)
    for _ in tqdm(range(args.steps), desc=game_dir.name, unit="step",
                  disable=args.no_progress):
        action_1, action_2, _, _ = agent.draw_action(state)
        obs, _, absorbing, info = mdp.step((action_1, action_2))
        frame = mdp.render(record=True)
        video.stdin.write(frame.tobytes())
        buffer.append(
            frame,
            obs,
            np.stack([command_to_xy(action_1[0], 0), command_to_xy(action_2[0], 1)]),
            np.stack([action_1, action_2]),
            list(info["score"]),
            list(info["faults"]),
        )
        state = obs
        if absorbing:
            saves.append(writer.submit(
                buffer.save, game_dir / f"episode_{len(episode_lengths):03d}.npz", True
            ))
            episode_lengths.append(buffer.n_actions)
            state, buffer = reset_episode(mdp, agent, video)

    # Budget exhausted: save the truncated tail unless the last step was
    # absorbing, which leaves only a fresh post-reset state in the buffer.
    if buffer.n_actions > 0:
        saves.append(writer.submit(
            buffer.save, game_dir / f"episode_{len(episode_lengths):03d}.npz", False
        ))
        episode_lengths.append(buffer.n_actions)

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
        "seed": args.seed,
        "steps": args.steps,
        "n_episodes": len(episode_lengths),
        "episode_lengths": episode_lengths,
        "final_score": list(mdp.base_env.score),
        "final_faults": list(mdp.base_env.faults),
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "puck_radius": args.puck_radius,
        "jax_backend": jax.default_backend(),
        "wall_time_s": round(elapsed, 1),
        "steps_per_s": round(args.steps / elapsed, 1),
    }
    (game_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
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
            "--run-dir", str(run_dir), "--game-start", str(game_start),
            "--no-progress",
        ]
        if args.keep_scoreboard:
            cmd.append("--keep-scoreboard")
        if args.puck_radius is not None:
            cmd += ["--puck-radius", str(args.puck_radius)]
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
                        help="Steps per game (45000 = full 15 min game)")
    parser.add_argument("--out", default="data_2023")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fps", type=int, default=50, help="Video fps (env runs at 50 Hz)")
    parser.add_argument("--keep-scoreboard", action="store_true",
                        help="Keep the scoreboard overlay baked into the frames")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--puck-radius", type=float, default=None,
                        help="Puck radius in meters (model default: 0.03165); "
                             "resizes collision and visuals at runtime")
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
        make_agent(mdp.env_info, 1, args.model1),
        make_agent(mdp.env_info, 2, args.model2),
    )

    for i in range(args.games):
        game = args.game_start + i
        np.random.seed(args.seed + game)
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
