# How to Run the Picochat Ablation on Modal

## Prerequisites

### 1. Install Modal globally (NOT inside the nanochat venv)

```bash
# Make sure you are NOT in the nanochat venv
deactivate   # run this if your terminal shows (.venv)

# Install modal globally
pip install modal
# or with uv:
uv tool install modal
```

Verify it is installed globally:
```bash
which modal   # should NOT show .venv/bin/modal
modal --version
```

If modal is accidentally installed inside the nanochat venv, remove it:
```bash
source .venv/bin/activate
uv pip uninstall modal
deactivate
```

### 2. Log in to Modal

```bash
modal setup   # opens browser to authenticate
```

### 3. Create the Modal secret (one time only)

This injects your API keys into all Modal containers.

Get your keys:
- **W&B key**: https://wandb.ai/authorize
- **HuggingFace token**: https://huggingface.co/settings/tokens (needs read access)

```bash
modal secret create nanochat-secrets \
    WANDB_API_KEY=your_wandb_key_here \
    HF_TOKEN=hf_your_token_here
```

To update the secret later:
```bash
modal secret create nanochat-secrets \
    WANDB_API_KEY=new_key \
    HF_TOKEN=new_token \
    --force
```

---

## Before Running on Modal: Local Smoke Test

Always run the local smoke test first to catch code errors before spending credits.
This runs all 4 configs at tiny scale (~3-5 min, free):

```bash
# From the nanochat repo root, with the nanochat venv active
source .venv/bin/activate
bash runs/runpico_test.sh
deactivate
```

You should see all 4 configs complete without errors:
```
==> [1/4] Smoke test: pico_baseline (ReLU², no CLA)    ✓
==> [2/4] Smoke test: pico_swiglu (SwiGLU, no CLA)     ✓
==> [3/4] Smoke test: pico_cla (ReLU², CLA-2)          ✓
==> [4/4] Smoke test: pico_swiglu_cla (SwiGLU + CLA-2) ✓
```

---

## Running on Modal

**Always run modal from the nanochat repo root, with the venv deactivated.**

```bash
cd /path/to/nanochat
deactivate   # make sure nanochat venv is not active
```

### Simple test first (~15 min, ~$3)

Run just the data and tokenizer stages to verify Modal is working end-to-end before committing to the full run:

```bash
modal run runs/pico_ablation_modal.py::stage_data
modal run runs/pico_ablation_modal.py::stage_tokenizer
```

If both pass without errors, the container image, volume, and secrets are all working correctly.

### Full ablation (~2.5 hours, ~$34)

```bash
modal run runs/pico_ablation_modal.py
```

This runs all stages in order:
```
[0/5] Download 40 FineWeb-EDU shards     CPU       ~5 min
[1/5] Train BPE tokenizer               1×H100    ~2 min
[2a/5] Pretrain pico_baseline           8×H100    ~10 min   ~$5
[2b/5] Pretrain pico_swiglu             8×H100    ~10 min   ~$5
[2c/5] Pretrain pico_cla                8×H100    ~10 min   ~$5
[2d/5] Pretrain pico_swiglu_cla         8×H100    ~10 min   ~$5
[3/5] Eval all 4 models (bpb+CORE)     4×H100    ~120 min  ~$9
```

### Run individual stages

If a stage fails or you want to re-run one step:

```bash
modal run runs/pico_ablation_modal.py::stage_data
modal run runs/pico_ablation_modal.py::stage_tokenizer
modal run runs/pico_ablation_modal.py::stage_pretrain_baseline
modal run runs/pico_ablation_modal.py::stage_pretrain_swiglu
modal run runs/pico_ablation_modal.py::stage_pretrain_cla
modal run runs/pico_ablation_modal.py::stage_pretrain_swiglu_cla
modal run runs/pico_ablation_modal.py::stage_eval
```

Stages are idempotent — data and tokenizer stages skip work already done.
Pretrain stages save checkpoints every 1000 steps and resume automatically if interrupted.

---

## Monitoring

### Live logs
Modal streams logs to your terminal while the job runs. You can also view them at:
https://modal.com/apps → select `nanochat-picochat-ablation`

### Costs
Monitor spend at: https://modal.com/settings/billing

Approximate costs per stage at H100 on-demand pricing (~$3.50/GPU/hr):
| Stage | GPUs | Time | Cost |
|---|---|---|---|
| stage_data | CPU | ~5 min | ~$0.10 |
| stage_tokenizer | 1×H100 | ~2 min | ~$0.12 |
| stage_pretrain_* (each) | 8×H100 | ~10 min | ~$4.70 |
| stage_eval | 4×H100 | ~120 min | ~$9.30 |
| **Total** | | **~2.5 hrs** | **~$34** |

### W&B training curves
If you set a real `WANDB_API_KEY`, training metrics are logged to:
https://wandb.ai → project `picochat-ablation`

All 4 runs appear in the same project for easy comparison.

### Eval results
At the end of `stage_eval`, a summary table is printed to the terminal:
```
============================================================
  PICOCHAT ABLATION RESULTS
============================================================
Model                val_bpb       CORE
------------------------------------------------------------
pico_baseline        X.XXX         X.XXX
pico_swiglu          X.XXX         X.XXX
pico_cla             X.XXX         X.XXX
pico_swiglu_cla      X.XXX         X.XXX
============================================================
GPT-2 CORE threshold: 0.256525
```

Full eval logs are saved to the Modal Volume at:
`/vol/nanochat_cache/<model>_eval.txt`

---

## Troubleshooting

### `AttributeError: 'H2Connection' object has no attribute '_frame_dispatch_table'`
Modal is installed inside the nanochat venv, causing a dependency conflict.
```bash
source .venv/bin/activate
uv pip uninstall modal
deactivate
pip install modal   # install globally
```

### `Secret 'nanochat-secrets' not found`
Run the secret creation command from the Prerequisites section.

### Stage fails mid-run
Re-run just the failed stage — all prior work is saved to the Modal Volume:
```bash
modal run runs/pico_ablation_modal.py::stage_pretrain_swiglu
```

### Container image rebuild taking too long
The first run builds the Docker image (~5-10 min). Subsequent runs reuse the cached image unless you change `gpt.py`, `base_train.py`, or other source files.

### OOM (out of memory)
Reduce `DEVICE_BATCH_SIZE` in `runs/pico_ablation_modal.py`:
```python
DEVICE_BATCH_SIZE = 8   # reduce from 16 if OOM
```
