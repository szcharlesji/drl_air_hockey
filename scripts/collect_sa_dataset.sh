#!/usr/bin/env bash
# Compatibility entry point for the YAML-driven v6 dataset collector.
#
#   bash scripts/collect_sa_dataset.sh
#   bash scripts/collect_sa_dataset.sh --dry-run
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-airhockey2023}"

exec conda run -n "$CONDA_ENV" python "$REPO/scripts/collect_sa_dataset.py" \
    --config "$REPO/configs/data/airhockey-v6.yaml" "$@"
