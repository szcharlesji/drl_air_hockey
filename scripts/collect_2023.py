#!/usr/bin/env python3
"""Collect action-conditioned world-model training data from tournament games.

Runs pretrained agents against each other fully headless (EGL) with agent
inference on the GPU, rendering a small RGB frame every control step, and
writes one .npz per episode plus one .mp4 per game for human verification.
Replaces the two-phase eval_2023.py -> replay_2023.py flow for data
collection: no dataset.pkl, no re-rendering, no per-step logging overhead.

Episode file layout (DreamerV3 obs-first convention; T actions, T+1 states):
    image       uint8   (T+1, H, W, 3)  image[k] is the frame of state s_k;
                                        image[0] is the post-reset frame
    action      float32 (T+1, 2, 2, 7)  action[k] led into s_k, i.e. the
                                        action taken *at* image[k] is
                                        action[k+1]; action[0] is zeros.
                                        Axes: (agent, [pos|vel], joint)
    obs         float32 (T+1, 46)       raw low-dim env observation of s_k
    is_first    bool    (T+1,)          True only at index 0
    is_last     bool    (T+1,)          True only at index -1
    is_terminal bool    (T+1,)          is_last AND the episode ended in an
                                        absorbing state (goal/fault/stuck);
                                        False when truncated by --steps
    score       int32   (T+1, 2)        running score at s_k (agent1, agent2)
    faults      int32   (T+1, 2)        running fault count at s_k

Example:
    python scripts/collect_2023.py --model1 tournament_aggressive \
        --model2 tournament_aggressive --games 2 --steps 5000
"""
import os

# Must be set before mujoco / drl_air_hockey imports. EGL renders headless on
# the GPU; the DRL_AIR_HOCKEY_* variables opt agent inference into the GPU
# (read by config_dreamerv3 at agent construction). setdefault so the shell
# can override, e.g. CUDA_VISIBLE_DEVICES=1 or DRL_AIR_HOCKEY_JAX_PLATFORM=cpu.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("DRL_AIR_HOCKEY_JAX_PLATFORM", "gpu")
# Two dreamerv3 Agent instances share this process; preallocating 75% of
# VRAM per XLA client is unnecessary and hostile to a shared GPU.
os.environ.setdefault("DRL_AIR_HOCKEY_JAX_PREALLOC", "false")

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
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
        self.actions = [np.zeros((2, 2, 7), dtype=np.float32)]
        self.scores = [score]
        self.faults = [faults]

    def append(self, frame, obs, action, score, faults):
        self.images.append(frame)
        self.obs.append(obs)
        self.actions.append(action)
        self.scores.append(score)
        self.faults.append(faults)

    @property
    def n_actions(self):
        return len(self.actions) - 1

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
            action=np.stack(self.actions).astype(np.float32),
            obs=np.stack(self.obs).astype(np.float32),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
            score=np.asarray(self.scores, dtype=np.int32),
            faults=np.asarray(self.faults, dtype=np.int32),
        )


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
            "camera_params": {
                "static": dict(
                    distance=3.0, elevation=-45.0, azimuth=90.0, lookat=(0.0, 0.0, 0.0)
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
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-preset", "fast", str(path)],
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

    # Both agent types have empty preprocessor lists, so Core._preprocess is
    # skipped here; revisit if agents ever register preprocessors.
    state, buffer = reset_episode(mdp, agent, video)
    for _ in tqdm(range(args.steps), desc=game_dir.name, unit="step"):
        action_1, action_2, _, _ = agent.draw_action(state)
        obs, _, absorbing, info = mdp.step((action_1, action_2))
        frame = mdp.render(record=True)
        video.stdin.write(frame.tobytes())
        buffer.append(
            frame,
            obs,
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
        "jax_backend": jax.default_backend(),
        "wall_time_s": round(elapsed, 1),
        "steps_per_s": round(args.steps / elapsed, 1),
    }
    (game_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", default="tournament_balanced", choices=MODELS)
    parser.add_argument("--model2", default="tournament_balanced", choices=MODELS)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--steps", type=int, default=45000,
                        help="Steps per game (45000 = full 15 min game)")
    parser.add_argument("--out", default="data_2023")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--fps", type=int, default=50, help="Video fps (env runs at 50 Hz)")
    parser.add_argument("--keep-scoreboard", action="store_true",
                        help="Keep the scoreboard overlay baked into the frames")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.out) / f"collect-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    mdp = build_mdp(args)
    # Agents are built once (checkpoint load + JIT warmup is expensive);
    # episode_start() fully resets their recurrent state between episodes.
    agent = SimpleTournamentAgentWrapper(
        mdp.env_info,
        make_agent(mdp.env_info, 1, args.model1),
        make_agent(mdp.env_info, 2, args.model2),
    )

    for game in range(args.games):
        np.random.seed(args.seed + game)
        if game > 0:
            # score/faults persist on the env instance; rebuild per game.
            close_mdp(mdp)
            mdp = build_mdp(args)
        meta = collect_game(run_dir / f"game_{game:03d}", mdp, agent, args)
        print(
            f"game_{game:03d}: {meta['n_episodes']} episodes, "
            f"score {meta['final_score']}, {meta['steps_per_s']} steps/s"
        )
    close_mdp(mdp)
    print(f"Data written to: {run_dir}")


if __name__ == "__main__":
    main()
