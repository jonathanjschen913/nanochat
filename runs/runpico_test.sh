#!/bin/bash

# Local smoke test for Part 2 ablations (CPU/MPS)
# Tests all 4 configs at tiny scale to verify code runs before cloud training.
# Estimated time: ~3-5 min total on M3 Max
# Run as: bash runs/runpico_test.sh

set -e

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate

# Train tokenizer (skip if already done)
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer.json" ]; then
    echo "==> Training tokenizer..."
    python -m nanochat.dataset -n 4
    python -m scripts.tok_train --max-chars=500000000
fi

# Shared tiny config — just enough to verify code runs without errors
COMMON="
    --depth=4
    --aspect-ratio=64
    --head-dim=64
    --window-pattern=L
    --max-seq-len=256
    --device-batch-size=4
    --total-batch-size=1024
    --num-iterations=50
    --eval-every=25
    --eval-tokens=4096
    --core-metric-every=-1
    --sample-every=-1
    --run=dummy
"

echo ""
echo "==> [1/4] Smoke test: pico_baseline"
python -m scripts.base_train \
    $COMMON \
    --model-tag="test_baseline"

echo ""
echo "==> [2/4] Smoke test: pico_diff_attn (Differential Attention)"
python -m scripts.base_train \
    $COMMON \
    --differential-attn \
    --model-tag="test_diff_attn"

echo ""
echo "==> [3/4] Smoke test: pico_mod (Mixture of Depths)"
python -m scripts.base_train \
    $COMMON \
    --mod-routing \
    --model-tag="test_mod"

echo ""
echo "==> [4/4] Smoke test: pico_mod_diff_attn (MoD + Differential Attention)"
python -m scripts.base_train \
    $COMMON \
    --mod-routing \
    --differential-attn \
    --model-tag="test_mod_diff_attn"

echo ""
echo "==> All 4 smoke tests passed. Ready for cloud training."
echo "    Next: modal run runs/pico_ablation_modal.py"
