#!/bin/bash
### One-shot setup of the `airhockey2023` conda env for this repo's data
### generation. Everything needed lives in the repo itself: the collector
### package (drl_air_hockey), the vendored 2023 challenge framework
### (third_party/air_hockey_challenge), and this pinned Dec-2023-era stack
### (the old requirements are unpinned and modern versions break the APIs).
### The pretrained tournament checkpoints are gitignored: copy
### drl_air_hockey/agents/models/*.ckpt from an existing machine or backup
### before running (the guard below refuses to continue without them).
set -euo pipefail
set -x

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-${HOME}/miniforge3}"

## 0. Gitignored RL checkpoints must already be in place
ls "${REPO}/drl_air_hockey/agents/models/"*.ckpt > /dev/null

## 1. Conda env (Python 3.10 for 2023-era wheel availability)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^airhockey2023 "; then
    conda create -y -n airhockey2023 python=3.10
fi
conda activate airhockey2023

## 2. Era-pinned scientific stack (CPU-only; old jax has no RTX 5090 support)
pip install setuptools==65.5.0 wheel==0.38.4
pip install \
    numpy==1.26.4 \
    jax==0.4.23 jaxlib==0.4.23 \
    tensorflow-cpu==2.15.0 tensorflow-probability==0.23.0 \
    optax==0.1.7 chex==0.1.85 \
    ruamel.yaml rich cloudpickle pyzmq PyYAML joblib scipy==1.11.4
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
# numpy and opencv pinned IN THE SAME resolver call: mushroom-rl's unpinned
# opencv dependency otherwise drags in numpy 2.x, breaking TF 2.15 and the
# mujoco 2.3.7 binary
pip install mujoco==2.3.7 mushroom-rl==1.10.1 dm_control==1.0.13 osqp nlopt \
    numpy==1.26.4 opencv-python==4.9.0.80

## Optional: GPU jax for Ampere-or-older GPUs (A6000/A4000 etc.; NOT RTX 50xx).
## Requires NVIDIA driver >= 525. Uncomment to replace the CPU jaxlib above:
# pip install "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

## 3. gym 0.19 (needed by old code). pip >= 24.1 rejects its malformed
##    metadata outright, so downgrade pip first (2023 Dockerfile did the same).
pip install "pip<24.1"
pip install --no-deps --no-build-isolation gym==0.19.0

## Sanity check before installing the repos
python -c "import numpy; assert numpy.__version__ == '1.26.4', numpy.__version__; import gym, mujoco, tensorflow, jax; print('2023 stack imports OK')"

## 4. This repo's two packages (no deps -- everything is already pinned above)
pip install --no-deps -e "${REPO}/third_party/air_hockey_challenge"
pip install --no-deps "dreamerv3 @ git+https://github.com/AndrejOrsula/dreamerv3.git@d4f47fcb18f52777314f2735389cd1f449513c9a"
pip install --no-deps -e "${REPO}"

echo "=== 2023 SETUP DONE ==="
echo "Run: conda activate airhockey2023 && cd ${REPO} && bash scripts/collect_sa_dataset.sh"
