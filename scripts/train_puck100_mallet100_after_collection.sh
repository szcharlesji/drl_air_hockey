#!/bin/bash
# Start the dedicated 2x5090 DDP run only after the matching collection H5 is
# complete.  This is intentionally a fresh run: it never resumes airhockey v1.
set -euo pipefail

DATA_DIR=/home_shared/grail_charles/data/airhockey_sa_h5_puck100_mallet100_arm25_v4_500k
RAW_DIR=/home_shared/grail_charles/data/raw_2023_puck100_mallet100_arm25_v4_500k
SA_DIR=/home_shared/grail_charles/X/sa
RECIPE=/home_shared/grail_charles/drl_air_hockey_2023/configs/recipe-airhockey-puck100-mallet100-arm25-v4.yaml
LOG_DIR="$SA_DIR/output"
LOG_FILE="$LOG_DIR/airhockey_puck100_mallet100_arm25_v4_ddp2_bs8.log"

mkdir -p "$LOG_DIR"
while [[ ! -f "$DATA_DIR/h5_manifest.json" ]]; do
    sleep 30
done

python - "$RAW_DIR" "$DATA_DIR" <<'PY'
import json
import sys
from pathlib import Path

raw_root, h5_root = map(Path, sys.argv[1:])
metas = sorted(raw_root.glob("*/collect-*/game_*/meta.json"))
raw_steps = sum(json.loads(path.read_text())["steps"] for path in metas)
manifest = json.loads((h5_root / "h5_manifest.json").read_text())
frames = manifest["metadata"]["total_frames"]
if not manifest.get("complete") or manifest.get("action_dim") != 4:
    raise SystemExit("incomplete or incompatible H5 manifest")
if raw_steps != 500_000:
    raise SystemExit(f"expected 500000 raw steps, found {raw_steps}")
if frames <= 0:
    raise SystemExit("H5 manifest has no training frames")
print(f"[handoff] raw_steps={raw_steps}, h5_frames={frames}", flush=True)
PY

# Do not claim GPUs that became busy while CPU collection was running.  Small
# display-server allocations are expected; a training process is not.
mapfile -t GPU_STATE < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
if [[ ${#GPU_STATE[@]} -lt 2 ]]; then
    echo "[handoff] expected two Orion GPUs; refusing to launch" >&2
    exit 1
fi
for gpu in 0 1; do
    IFS=',' read -r memory utilization <<<"${GPU_STATE[$gpu]}"
    memory=${memory// /}
    utilization=${utilization// /}
    if (( memory > 500 || utilization > 5 )); then
        echo "[handoff] GPU $gpu is no longer free (${memory} MiB, ${utilization}%); refusing to launch" >&2
        exit 1
    fi
done

cd "$SA_DIR"
exec env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 WANDB_MODE=online \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    conda run -n sa torchrun --standalone --nproc_per_node=2 main.py \
        --ddp \
        --config configs/sa-episodic.py \
        --recipe "$RECIPE" \
        --wandb_project sa-airhockey \
        --wandb_entity charlesji \
        --wandb_run_name airhockey_puck100_mallet100_arm25_v4_ddp2_bs8 \
        --ckpt_every_k 10 \
        > "$LOG_FILE" 2>&1
