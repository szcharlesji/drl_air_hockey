#!/bin/bash
# One-shot SA dataset build: collect air-hockey games with 30 parallel workers
# (70% aggressive-vs-aggressive, 30% smooth_random-vs-smooth_random, running
# concurrently), then convert everything collected so far into the
# langtable_action_h5_v1 shard format that X/sa trains on (dataset: languagetable).
#
#   bash scripts/collect_sa_dataset.sh
#
# Re-running the script ADDS more games (seeds auto-advance past what already
# exists in RAW_DIR) and rebuilds the H5 root from all raw data. Train with:
#   cd /home_shared/grail_charles/X/sa
#   python main.py --config configs/sa-episodic.py --recipe configs/recipe-airhockey-v1.yaml \
#       --wandb_project sa-airhockey --wandb_entity charlesji --wandb_run_name airhockey_v1
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV=airhockey2023

# ---------- knobs ----------
DATA_ROOT=/home_shared/grail_charles/data
RAW_DIR=$DATA_ROOT/raw_2023            # npz + mp4 masters (256px), kept for reuse
H5_DIR=$DATA_ROOT/airhockey_sa_h5_v1   # what the recipe's data_dir points at
GAMES_AGGR=${GAMES_AGGR:-105}          # 70% of games
GAMES_RAND=${GAMES_RAND:-45}           # 30% of games
WORKERS_AGGR=21                        # 70% of the 30 workers
WORKERS_RAND=9
STEPS=${STEPS:-2000}                   # env steps per game (50 Hz)
PUCK_RADIUS=${PUCK_RADIUS:-0.07}
PLATFORM=${PLATFORM:-cpu}              # cpu = GPU-free; on the A6000 box GPU rendering is ~3x faster
IMG_SIZE=128                           # H5 frame size (masters stay 256)
VAL_FRAC=0.05                          # held-out games per source

mkdir -p "$RAW_DIR/aggressive" "$RAW_DIR/random" "$RAW_DIR/logs" "$H5_DIR"

# Advance seeds past already-collected games so re-runs add fresh trajectories
# (per-game seed inside a run = seed + game_index). Random uses a disjoint base.
existing_aggr=$(find "$RAW_DIR/aggressive" -maxdepth 2 -type d -name 'game_*' 2>/dev/null | wc -l)
existing_rand=$(find "$RAW_DIR/random" -maxdepth 2 -type d -name 'game_*' 2>/dev/null | wc -l)
SEED_AGGR=$existing_aggr
SEED_RAND=$((100000 + existing_rand))

STAMP=$(date +%Y%m%d-%H%M%S)
echo "[collect] aggressive: $GAMES_AGGR games x $STEPS steps, $WORKERS_AGGR workers, seed base $SEED_AGGR"
echo "[collect] random:     $GAMES_RAND games x $STEPS steps, $WORKERS_RAND workers, seed base $SEED_RAND"
echo "[collect] logs: $RAW_DIR/logs/{aggressive,random}-$STAMP.log"

trap 'kill 0' INT TERM

conda run -n "$CONDA_ENV" python "$REPO/scripts/collect_2023.py" \
    --platform "$PLATFORM" --workers "$WORKERS_AGGR" \
    --model1 tournament_aggressive --model2 tournament_aggressive \
    --games "$GAMES_AGGR" --steps "$STEPS" --puck-radius "$PUCK_RADIUS" \
    --seed "$SEED_AGGR" --out "$RAW_DIR/aggressive" \
    > "$RAW_DIR/logs/aggressive-$STAMP.log" 2>&1 &
AGGR_PID=$!

conda run -n "$CONDA_ENV" python "$REPO/scripts/collect_2023.py" \
    --platform "$PLATFORM" --workers "$WORKERS_RAND" \
    --model1 smooth_random --model2 smooth_random \
    --games "$GAMES_RAND" --steps "$STEPS" --puck-radius "$PUCK_RADIUS" \
    --seed "$SEED_RAND" --out "$RAW_DIR/random" \
    > "$RAW_DIR/logs/random-$STAMP.log" 2>&1 &
RAND_PID=$!

wait "$AGGR_PID" || { echo "aggressive collection FAILED, see log"; exit 1; }
echo "[collect] aggressive done"
wait "$RAND_PID" || { echo "random collection FAILED, see log"; exit 1; }
echo "[collect] random done"

echo "[convert] rebuilding $H5_DIR from all raw data"
conda run -n "$CONDA_ENV" python "$REPO/scripts/convert_npz_to_sa_h5.py" \
    --source aggressive="$RAW_DIR/aggressive" \
    --source random="$RAW_DIR/random" \
    --out "$H5_DIR" --img-size "$IMG_SIZE" --val-frac "$VAL_FRAC" --overwrite

echo "[done] dataset at $H5_DIR (manifest: $H5_DIR/h5_manifest.json)"
