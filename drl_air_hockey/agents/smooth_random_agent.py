"""Scripted smooth-random mallet policy for world-model data diversity.

Wanders the same operating box the RL agents use: random waypoints in
end-effector xy space (robot base frame) tracked by a critically damped
spring, with per-episode randomized speeds, pauses, and occasional fast
strokes. Joint commands come from warm-started inverse kinematics on the
commanded xy — the same pattern as the env's built-in scripted opponent
(IiwaPositionHit._default_opponent_action_gen) — so the world-frame
`action` recorded by the collector (FK of the joint command) is exactly
this smooth trajectory.
"""
import numpy as np
from air_hockey_challenge.framework.agent_base import AgentBase
from air_hockey_challenge.utils.kinematics import (
    forward_kinematics,
    inverse_kinematics,
)

MOTION_PARAMS = {
    # Per-episode uniform draws
    "omega_base": (2.0, 4.5),  # rad/s: spring frequency of casual segments
    "v_cap": (0.9, 1.5),  # m/s: speed cap of casual segments
    "p_hold": (0.2, 0.5),  # chance to pause after reaching a waypoint
    "p_stroke": (0.05, 0.25),  # chance a new segment is a fast stroke
    # Per-segment uniform draws
    "omega_segment_scale": (0.7, 1.3),  # spread around omega_base
    "omega_stroke": (8.0, 14.0),  # rad/s: peak ~1.5-3 m/s, like RL hits
    "hold_duration": (0.2, 1.5),  # s
    # Fixed
    "v_cap_stroke": 3.0,  # m/s: max 6 cm commanded step at 50 Hz
    "min_waypoint_dist": 0.15,  # m: redraw waypoints closer than this
    "waypoint_inset": 0.01,  # m: keep waypoints off the box edges
    "arrival_dist": 0.02,  # m
    "arrival_speed": 0.1,  # m/s
    "segment_timeout": 6.0,  # multiples of 1/omega before giving up
}


class SmoothRandomAgent(AgentBase):
    """Mallet wanderer that mostly ignores the puck; no checkpoint, no jax."""

    def __init__(self, env_info, agent_id=1, **kwargs):
        super().__init__(env_info, agent_id, **kwargs)
        self.dt = env_info["dt"]
        self.n_joints = env_info["robot"]["n_joints"]
        self.ee_height = env_info["robot"]["ee_desired_height"]
        # Match the env's safety layer so commands are never rewritten there.
        self.joint_vel_limit = 0.95 * env_info["robot"]["joint_vel_limit"][1]
        # Operating box in the robot's own base frame, identical to the RL
        # agents' (SingleStrategySpaceRAgent.compute_ee_table_minmax with the
        # balanced-strategy offsets), so the action support matches RL games.
        offset_table, offset_centre, offset_goal = 0.005, 0.15, 0.01
        base_x = np.abs(env_info["robot"]["base_frame"][0][0, 3])
        half_length = env_info["table"]["length"] / 2
        half_width = env_info["table"]["width"] / 2
        mallet_radius = env_info["mallet"]["radius"]
        self.box_min = np.array(
            [
                base_x - half_length + mallet_radius + offset_table + offset_goal,
                -half_width + mallet_radius + offset_table,
            ]
        )
        self.box_max = np.array(
            [
                base_x - offset_centre,
                half_width - mallet_radius - offset_table,
            ]
        )
        self.reset()

    def reset(self):
        # Agents are constructed once per worker and reused across games, and
        # the collector reseeds np.random per game *after* reset — so no RNG
        # draws here; everything is deferred to the first draw_action.
        self._first_step = True

    def draw_action(self, obs):
        if self._first_step:
            self._start_episode(obs)
        self._step_motion()
        return self._track()

    def _start_episode(self, obs):
        self._first_step = False
        # Start from the measured post-reset configuration so the first
        # command is continuous with where the mallet actually is.
        self._q_cmd = self.get_joint_pos(obs).copy()
        self._pos = self.get_ee_pose(obs)[0][:2].copy()
        self._vel = np.zeros(2)
        p = MOTION_PARAMS
        self._omega_base = np.random.uniform(*p["omega_base"])
        self._v_cap_episode = np.random.uniform(*p["v_cap"])
        self._p_hold = np.random.uniform(*p["p_hold"])
        self._p_stroke = np.random.uniform(*p["p_stroke"])
        self._hold_steps = 0
        self._start_segment()

    def _start_segment(self):
        p = MOTION_PARAMS
        if np.random.uniform() < self._p_stroke:
            self._omega = np.random.uniform(*p["omega_stroke"])
            self._v_cap = p["v_cap_stroke"]
        else:
            self._omega = self._omega_base * np.random.uniform(
                *p["omega_segment_scale"]
            )
            self._v_cap = self._v_cap_episode
        lo = self.box_min + p["waypoint_inset"]
        hi = self.box_max - p["waypoint_inset"]
        for _ in range(5):
            self._target = np.random.uniform(lo, hi)
            if np.linalg.norm(self._target - self._pos) >= p["min_waypoint_dist"]:
                break
        self._segment_steps_left = int(
            round(p["segment_timeout"] / self._omega / self.dt)
        )

    def _step_motion(self):
        p = MOTION_PARAMS
        if self._hold_steps > 0:
            self._hold_steps -= 1
            if self._hold_steps == 0:
                self._start_segment()
        else:
            self._segment_steps_left -= 1
            arrived = (
                np.linalg.norm(self._target - self._pos) < p["arrival_dist"]
                and np.linalg.norm(self._vel) < p["arrival_speed"]
            )
            if arrived or self._segment_steps_left <= 0:
                if np.random.uniform() < self._p_hold:
                    self._hold_steps = int(
                        round(np.random.uniform(*p["hold_duration"]) / self.dt)
                    )
                else:
                    self._start_segment()
        # Critically damped spring toward the waypoint (a hold keeps servoing
        # to the reached waypoint, settling to rest).
        accel = self._omega**2 * (self._target - self._pos)
        accel -= 2.0 * self._omega * self._vel
        self._vel += self.dt * accel
        speed = np.linalg.norm(self._vel)
        if speed > self._v_cap:
            self._vel *= self._v_cap / speed
        self._pos = self._pos + self.dt * self._vel
        clipped = np.clip(self._pos, self.box_min, self.box_max)
        out = clipped != self._pos
        if out.any():
            self._vel[out] = 0.0
            self._pos = clipped

    def _track(self):
        """IK-track the commanded xy; return the (2, n_joints) [pos, vel] command.

        Warm-started from the previously *commanded* joints (never the
        measured ones), so FK of the command reproduces the internal xy state
        and the recorded action channel stays clean.
        """
        success, q_new = inverse_kinematics(
            self.robot_model,
            self.robot_data,
            np.array([self._pos[0], self._pos[1], self.ee_height]),
            initial_q=self._q_cmd,
        )
        if not success:
            # Not expected inside the operating box: hold the previous
            # command for one step, resync, and head somewhere else.
            self._pos = forward_kinematics(
                self.robot_model, self.robot_data, self._q_cmd
            )[0][:2].copy()
            self._vel[:] = 0.0
            self._start_segment()
            return np.vstack([self._q_cmd, np.zeros(self.n_joints)])
        joint_vel = (q_new - self._q_cmd) / self.dt
        over = np.max(np.abs(joint_vel) / self.joint_vel_limit)
        if over > 1.0:
            # Downscale the step and resync the internal state to what is
            # actually commanded, so state and command never diverge.
            q_new = self._q_cmd + (q_new - self._q_cmd) / over
            joint_vel /= over
            self._pos = forward_kinematics(
                self.robot_model, self.robot_data, q_new
            )[0][:2].copy()
            self._vel /= over
        self._q_cmd = q_new
        return np.vstack([q_new, joint_vel])
