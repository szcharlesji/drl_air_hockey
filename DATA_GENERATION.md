# Data Generation for World-Model Training

Collect action-conditioned pixel data from 2023 Air Hockey Challenge tournament
games played by the pretrained DreamerV3 agents. One pass does everything:
headless GPU (EGL) rendering, per-episode `.npz` files, a per-game `.mp4` for
human verification, and a `meta.json` per game. No `dataset.pkl`, no replay step.

Everything lives on the `data-collection` branch of **two** repos:
- this repo (`drl_air_hockey_2023`) — the collector and pretrained-model glue;
- `air_hockey_challenge_2023` — the env (visual changes + absorbing-guard fixes).

## Setup (one-time)

```bash
conda activate airhockey2023
# CUDA jaxlib for GPU inference (A6000 = sm_86; driver >= 570 already installed)
pip install "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "nvidia-cudnn-cu12>=8.9,<9"   # pip resolves cuDNN 9 by default; jaxlib 0.4.23 needs 8.9
python -c "import jax; print(jax.default_backend())"   # -> gpu
```

Pretrained checkpoints go in `drl_air_hockey/agents/models/` (untracked):
`tournament_{balanced,aggressive,defensive,balanced_no_selfplay}.ckpt`.

## Commands

```bash
# Quick test: one game, 2000 steps, GPU inference (default)
python scripts/collect_2023.py --model1 tournament_aggressive --model2 tournament_aggressive \
    --games 1 --steps 2000

# Full game = 45000 steps = 15 min of play at 50 Hz (the default --steps)
python scripts/collect_2023.py --model1 tournament_aggressive --model2 tournament_defensive

# RECOMMENDED for bulk collection: parallel CPU workers (physics is the
# bottleneck, not inference — see Performance). Games are split across
# subprocesses writing into one run dir.
python scripts/collect_2023.py --workers 24 --platform cpu --games 24

# Pin both CUDA inference and EGL rendering to one nvidia-smi GPU
python scripts/collect_2023.py --gpu 1 --games 1

# Domain variation: resize the puck at runtime (no XML edits needed)
python scripts/collect_2023.py --puck-radius 0.04 --games 4
```

| Flag | Default | Meaning |
|---|---|---|
| `--model1/--model2` | `tournament_balanced` | agent checkpoints (or `baseline`) |
| `--games` | 1 | number of games |
| `--steps` | 45000 | env steps per game (50 Hz) |
| `--out` | `data_2023` | output root |
| `--width/--height` | 256/256 | frame size |
| `--workers` | 1 | parallel collector processes |
| `--platform` | `gpu` | inference device (`cpu` for worker fleets) |
| `--gpu` | — | pin CUDA + EGL to one nvidia-smi device |
| `--puck-radius` | 0.03165 | puck radius in m (collision + visuals + mass/inertia) |
| `--seed` | 0 | per-game seed = `seed + game_index` |
| `--keep-scoreboard` | off | bake the score overlay into frames |
| `--fps` | 50 | verification-video framerate |

## Output layout

```
data_2023/collect-<timestamp>/game_000/
    episode_000.npz ... episode_NNN.npz
    video.mp4        # exact collected frames, H.264, plays in VSCode
    meta.json        # models, seed, episode lengths, score, steps/s, jax backend
```

## Episode format (DreamerV3 obs-first convention: T actions, T+1 states)

| Key | Shape / dtype | Notes |
|---|---|---|
| `image` | (T+1, H, W, 3) uint8 | `image[0]` = post-reset frame |
| `action` | (T+1, 2, 2, 7) float32 | `action[k]` led into state k; `action[0]` = zeros. Axes: (agent, pos/vel, joint) |
| `obs` | (T+1, 46) float32 | raw low-dim env observation |
| `is_first / is_last / is_terminal` | (T+1,) bool | `is_terminal[-1]` False only when truncated by `--steps` |
| `score / faults` | (T+1, 2) int32 | running counters |

The action taken *at* `image[k]` is `action[k+1]`. Episodes end on goal, fault
(15 s one-side timer), or stuck-puck deadlock; episode lengths per game sum to `--steps`.

## Rendered scene (env repo, `data-collection` branch)

Top-down camera (table long axis = image width, distance 2.66 m), white surface,
**red rims**, **green goal-mouth markers** (visual-only geoms), bright-red puck with
blue heading dot, grey mallets. Scoreboard overlay stripped from training frames.

## Performance (measured on this box: 32-core/64-thread TR PRO 5975WX, 3x A6000)

- Single process: ~62 steps/s at 256x256 (~71 at 128x72). GPU inference beats CPU
  only ~1.25x — 69% of each step is single-threaded MuJoCo physics.
- Workers (at 128px): ~43 steps/s each up to ~8 workers, ~26 each at 24 workers
  (SMT + all-core clocks), sustained aggregate ~630 steps/s at 24 workers,
  plateau ~660 at 32. **Use ~24 workers**; each worker needs ~30 s startup.
- Storage: ~14 KB/step at 256px (~630 MB per full game); a 24-game batch ~15 GB.

## Puck resizing caveats

`set_puck_radius` keeps the sim self-consistent (collision size, bounding
radius + AABB + BVH leaf, mass ~ r², disc inertia, solver constants, visuals) —
verified: wall bounces conserve energy at r up to 0.08 with <1 mm penetration.
`env_info['puck']['radius']` deliberately stays 0.03165 (the policies' training
constant). The pretrained agents are out-of-distribution on odd pucks: expect
degraded play and shorter episodes beyond ~±30% of 0.03165 (0.025–0.042 is a
reasonable variation band).

## Related scripts

```bash
python scripts/eval_2023.py --model1 tournament_aggressive --model2 baseline --steps 500  # logs dataset.pkl
python scripts/replay_2023.py logs_2023/eval-*/Game_0/*/dataset.pkl                       # dataset.pkl -> video
python scripts/challenge/download_dataset.py    # official 2023 tournament datasets (same pkl format)
```
