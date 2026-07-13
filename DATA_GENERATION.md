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
# Quick CPU smoke test: one fixed-length 100-transition episode at 128 px / 50 Hz.
# It writes one .npz plus a 50 fps verification video under --out.
conda run -n airhockey2023 python scripts/collect_2023.py \
    --platform cpu --model1 smooth_random --model2 smooth_random \
    --games 1 --steps 100 --width 128 --height 128 --fps 50 \
    --puck-radius 0.10 --mallet-radius 0.10 --robot-visual-scale 2.5 \
    --idle-prob1 0.10 --idle-prob2 0.10 --out /tmp/airhockey-smoke

# Full game = 45000 transitions = 15 min at 50 Hz (the default --steps)
python scripts/collect_2023.py --model1 tournament_aggressive --model2 tournament_defensive

# RECOMMENDED for bulk collection: parallel CPU workers (physics is the
# bottleneck, not inference — see Performance). Games are split across
# subprocesses writing into one run dir.
python scripts/collect_2023.py --workers 24 --platform cpu --games 24

# Pin both CUDA inference and EGL rendering to one nvidia-smi GPU
python scripts/collect_2023.py --gpu 1 --games 1

# Domain variation: resize the puck at runtime (no XML edits needed)
python scripts/collect_2023.py --puck-radius 0.04 --games 4

# Scripted smooth-random mallet motion (no checkpoints, no policy inference):
# mallets wander their own halves and only incidentally touch the puck
python scripts/collect_2023.py --workers 24 --platform cpu --games 24 \
    --model1 smooth_random --model2 smooth_random
```

| Flag | Default | Meaning |
|---|---|---|
| `--model1/--model2` | `tournament_balanced` | agent checkpoints, `baseline`, or `smooth_random` |
| `--games` | 1 | number of games |
| `--steps` | 45000 | transitions per game (50 Hz); default mode writes one episode per game |
| `--episode-mode` | `fixed_length` | `fixed_length` records exactly `--steps`; legacy `split_on_absorbing` resets after events |
| `--out` | `data_2023` | output root |
| `--width/--height` | 256/256 | frame size |
| `--workers` | 1 | parallel collector processes |
| `--platform` | auto | `cpu` = fully GPU-free: CPU inference AND software (llvmpipe) rendering |
| `--gpu` | — | pin CUDA + EGL to one nvidia-smi device |
| `--shadows` | off | render shadows/reflections (off so frames match across GPU/CPU renderers) |
| `--puck-radius` | 0.03165 | puck radius in m (collision + visuals + mass/inertia); the physical/scored/rendered goal mouth stays 0.1867 m wider than the puck diameter |
| `--mallet-radius` | 0.04815 | physical IIWA mallet radius (collision, visual mallet, policy bounds, and tournament reset range) |
| `--robot-visual-scale` | 1.0 | visual-only IIWA arm-mesh scale; physics and actions are unchanged |
| `--orientation-marker-arm-length` / `--orientation-marker-stroke-width` | 0.08 / 0.024 m | absolute dimensions of the puck's asymmetric red cross, independent of puck size |
| `--idle-prob1` / `--idle-prob2` | 0 | per-game probability each player's mallet holds its reset pose for the entire episode |
| `--seed` | 0 | per-game seed = `seed + game_index` |
| `--keep-scoreboard` | off | bake the score overlay into frames |
| `--fps` | 50 | verification-video framerate |

## The `smooth_random` policy

`smooth_random` (`drl_air_hockey/agents/smooth_random_agent.py`) replaces the
RL checkpoint with a scripted wanderer: random waypoints inside the same
operating box the RL agents use, tracked by a critically damped spring in
mallet-xy space, with per-episode randomized speeds, pauses, and occasional
fast strokes (~1.5–3 m/s). Joint commands come from warm-started IK, so the
recorded `action` channel *is* this smooth xy trajectory. With both sides
random, the puck often idles. In the default fixed-length collector this
latches goal, fault, centre-stuck, and edge events without ending the game.
Only a real goal leaves the puck at its goal coordinate, disables its contacts,
and hides it for the remaining frames; faults, stuck pucks, and edge contacts
remain visible and keep simulating. The episode still reaches `--steps`.
Tunables live in `MOTION_PARAMS` at the top of the agent file.

`--idle-prob1` and `--idle-prob2` wrap any policy (including tournament
policies and `smooth_random`). The decision is sampled once per game; an idle
mallet holds its measured reset joint pose with zero joint velocity, so it is
real, physically stationary action data rather than a visual-only effect.

## Output layout

```
data_2023/collect-<timestamp>/game_000/
    episode_000.npz   # default: the only episode, exactly --steps actions
    video.mp4        # exact collected frames, H.264, plays in VSCode
    meta.json        # models, seed, episode lengths, score, steps/s, jax backend
```

## Episode format (DreamerV3 obs-first convention: T actions, T+1 states)

| Key | Shape / dtype | Notes |
|---|---|---|
| `image` | (T+1, H, W, 3) uint8 | `image[0]` = post-reset frame |
| `action` | (T+1, 2, 2) float32 | commanded mallet x,y in world frame (FK of the joint command); `action[k]` led into state k; `action[0]` = zeros. Axes: (agent, xy) |
| `action_joints` | (T+1, 2, 2, 7) float32 | raw joint-space command behind `action[k]`. Axes: (agent, pos/vel, joint) |
| `obs` | (T+1, 46) float32 | raw low-dim env observation |
| `is_first / is_last / is_terminal` | (T+1,) bool | default fixed-length games are truncations: `is_terminal` is all false |
| `score / faults` | (T+1, 2) int32 | running counters |

The action taken *at* `image[k]` is `action[k+1]`. With the default
`fixed_length` mode, every game is exactly one `T = --steps` action episode
and has `T+1` image/observation frames. A goal, 15 s side fault, centre-stuck
puck, or invalid escape is recorded in `meta.json` under `terminal_events`.
Only a goal's event frame is followed by a hidden, inert
puck tail; fault, stuck, and edge conditions remain visible. Use
`--episode-mode split_on_absorbing` only to recover the old reset-and-split
behavior. The raw `obs` retains the frozen in-goal puck coordinate; the SA
converter uses only images and actions.

## SA dataset pipeline (X/sa world model)

The versioned YAML config is the source of truth for reproducible SA data:
[`configs/data/airhockey-v6.yaml`](configs/data/airhockey-v6.yaml). It defines
5,000 games × 1,000 transitions = 5,000,000 transitions at 128 px / 50 Hz:
3,500 aggressive-vs-aggressive games on 17 CPU workers and 1,500
smooth-random-vs-smooth-random games on 7 workers. Both sources use a 10 cm
puck, 10 cm physical mallets, 2.5× arm visuals, a fixed-scale marker, and a
10% per-player idle-game probability.

```bash
# Validate the YAML and print the exact two collector and conversion commands.
conda run -n airhockey2023 python scripts/collect_sa_dataset.py \
    --config configs/data/airhockey-v6.yaml --dry-run

# Run both source collectors concurrently, then rebuild the H5 dataset.
conda run -n airhockey2023 python scripts/collect_sa_dataset.py \
    --config configs/data/airhockey-v6.yaml

# Compatibility wrapper for the same v6 config.
bash scripts/collect_sa_dataset.sh
```

- Raw 128 px masters accumulate in
  `~/data/raw_2023_puck100_mallet100_arm25_v6/{aggressive,random}`; the H5
  root is `~/data/airhockey_sa_h5_puck100_mallet100_arm25_v6_5m`. The YAML
  counts are target totals: re-running collects only the remaining completed
  games to reach 3,500 + 1,500, ignores incomplete game directories, chooses
  fresh seeds, then rebuilds H5 from completed raw games. It snapshots the
  YAML under `raw/.collection/` and holds a root lock, so run only one
  launcher per raw root. Use `--skip-convert` to collect raw data only; use
  `--break-lock` only after confirming a previous lock is stale.
- The H5 root contains `h5_manifest.json` (the loader never
  globs shards) + `train|valid/shard_*.h5` holding
  `/episodes/<key>/images (T,128,128,3) uint8` and `actions (T,4) float32` =
  `[a1x, a1y, a2x, a2y]` commanded mallet xy in world frame. Alignment is
  shifted from the npz convention to SA's: `images[t] = image[t]`,
  `actions[t] = action[t+1]` (drives frame t -> t+1), final frame dropped.
  The valid split holds out whole games per source. Converter:
  `scripts/convert_npz_to_sa_h5.py` (standalone; see `--help`).
- `X/sa/configs/recipe-airhockey-v1.yaml` deliberately remains pointed at the
  old v1 data. For this v6 physical-mallet variant, copy that recipe and set
  `args.data_dir` to
  `/home_shared/grail_charles/data/airhockey_sa_h5_puck100_mallet100_arm25_v6_5m` plus a
  new `args.ckpt_dir_name` before launching. (The recipe overrides CLI
  `--data_dir`.) It keeps the same sketchy-v4 hyperparameters, `action_dim: 4`,
  and seq_len-20 dense windows.

```bash
cd /home_shared/grail_charles/X/sa
# after copying/configuring the v6 recipe described above:
python main.py --config configs/sa-episodic.py --recipe configs/recipe-airhockey-puck100-mallet100-arm25-v6.yaml \
    --wandb_project sa-airhockey --wandb_entity charlesji --wandb_run_name airhockey_puck100_mallet100_arm25_v6
```

## Running on different machines

Device selection adapts per machine; the frames come out identical either way
(shadows/reflections are disabled by default for exactly this reason):

- **default (no flags)**: inference on GPU if the pinned jaxlib 0.4.23 can
  target it (Ampere/Ada/Hopper, compute capability <= 9.0 — e.g. A6000 yes,
  RTX 5090/Blackwell no; auto-detected via nvidia-smi, falls back to CPU
  inference with GPU rendering). Rendering always on GPU via EGL.
- **`--platform cpu`**: nothing touches the GPUs — CPU inference plus Mesa
  llvmpipe software rendering (~16 ms/frame at 256px, `LP_NUM_THREADS=2`).
- **`--gpu N`**: pin both CUDA inference and EGL rendering to nvidia-smi
  device N (EGL index resolved via `EGL_CUDA_DEVICE_NV`, since EGL device
  order is machine-specific).

## Rendered scene (env repo, `data-collection` branch)

Top-down camera (table long axis = image width, distance 2.66 m), white surface,
**red rims**, **green goal-mouth markers**, bright-red puck with an asymmetric
dark-red forward cross, grey mallets. The marker has a fixed world-space scale
when the puck radius changes. The physical end-rim gap, scoring width, and
green marker widen or narrow together while retaining the stock 0.1867 m total
clearance around the puck. Scoreboard overlay is stripped from training frames.

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
degraded play beyond ~±30% of 0.03165 (0.025–0.042 is a reasonable variation
band); fixed-length collection still keeps its requested duration.

## Related scripts

```bash
python scripts/eval_2023.py --model1 tournament_aggressive --model2 baseline --steps 500  # logs dataset.pkl
python scripts/replay_2023.py logs_2023/eval-*/Game_0/*/dataset.pkl                       # dataset.pkl -> video
python scripts/challenge/download_dataset.py    # official 2023 tournament datasets (same pkl format)
```
