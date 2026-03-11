# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nanochat** is a minimal, full-stack experimental harness for training and deploying LLMs. It covers the entire pipeline: tokenization → pretraining → evaluation → fine-tuning → chat inference/UI. The focus is simplicity, hackability, and compute efficiency.

## Commands

**Package manager**: `uv` (Python 3.10+)

```bash
# Setup
uv venv
uv sync --extra gpu    # GPU (CUDA 12.8)
uv sync --extra cpu    # CPU only

# Tokenizer
python -m scripts.tok_train

# Pretraining (multi-GPU)
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train

# Evaluation
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# Supervised fine-tuning
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft

# Chat interfaces
python -m scripts.chat_web   # FastAPI web server
python -m scripts.chat_cli   # Terminal chat

# Tests
python -m pytest tests/
python -m pytest tests/ -m "not slow"   # Skip slow tests
python -m pytest tests/test_engine.py -v  # Single test file

# Quick experiment (~5 min on 8×H100, good for iteration)
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 --run=d12 --model-tag=d12 \
    --core-metric-every=999999 --sample-every=-1 --save-every=-1
```

See `runs/speedrun.sh` for the full end-to-end pipeline (~3 hours on 8×H100). All scripts also work on a single GPU (automatically uses gradient accumulation), just omit `torchrun`.

## Architecture

### Core Pipeline

```
scripts/tok_train.py  →  scripts/base_train.py  →  scripts/chat_sft.py  →  scripts/chat_web.py
(tokenizer)              (pretraining)               (fine-tuning)            (inference)
```

### Key Modules

**`nanochat/gpt.py`** — Transformer model. Modern design: rotary embeddings (no positional embeddings), Group-Query Attention, QK normalization, ReLU² MLP activation, untied embedding/lm_head weights, optional sliding window attention (`--window-pattern`).

**`nanochat/flash_attention.py`** — Unified attention interface. Automatically uses Flash Attention 3 on Hopper (sm90) GPUs, falls back to PyTorch SDPA everywhere else (Ampere, Blackwell, MPS, CPU). Drop-in replacement for FA3's API.

**`nanochat/engine.py`** — Inference engine with KV cache, temperature/top-k/top-p sampling, and tool use (calculator, Python execution via `nanochat/execution.py`). Works purely on token IDs (no tokenization knowledge).

**`nanochat/dataloader.py`** — Distributed data loader using BOS-aligned bestfit packing (100% token utilization, no padding waste).

**`nanochat/optim.py`** — AdamW + Muon optimizer combo, both single-GPU (`MuonAdamW`) and distributed (`DistMuonAdamW`) variants.

**`nanochat/core_eval.py`** — CORE metric: 22-benchmark evaluation suite (ARC, MMLU, GSM8K, HumanEval, etc.) based on the DCLM eval harness.

**`nanochat/dataset.py`** — Download and read pretraining data shards. Data cached to `~/.cache/nanochat` (override with `NANOCHAT_BASE_DIR` env var).

**`nanochat/fp8.py`** — FP8 quantization support for low-precision training (H100+).

**`nanochat/checkpoint_manager.py`** — Save/load checkpoints with distributed training support.

### Complexity Dial

The single `--depth` parameter (number of transformer layers) controls all other hyperparameters: width (`depth * aspect_ratio`), number of heads, learning rate schedule, and total training duration. This keeps models compute-optimal at any scale without manual tuning. Any code changes must work across all depth settings. Key scales: d12 = GPT-1 size (~5 min quick experiments), d24-d26 = GPT-2 capability range.

### Tasks & Benchmarks

`tasks/` contains individual evaluation task implementations (ARC, GSM8K, MMLU, HumanEval, SpellingBee, Smoltalk). These are used by `core_eval.py` and `scripts/base_eval.py`.

### Tasks Interface

`tasks/common.py` defines the `Task` base class with `eval_type` (`'generative'` | `'categorical'`), `get_example(index)`, and `evaluate(problem, completion)`. `TaskMixture` shuffles multiple tasks for SFT training; `TaskSequence` chains them for curriculum training.

### Experiment Tracking

Training uses wandb for metrics (`--run=<name>`, `--run=dummy` disables). Key metrics to monitor: `val_bpb` (bits per byte), `core_metric` (DCLM CORE score), `train/mfu` (Model FLOPS utilization), `train/tok_per_sec`. `nanochat/report.py` generates training reports. `dev/LOG.md` is a running experiment log.

### Environment

- Intermediate artifacts (data, checkpoints, tokenizer) cached at `~/.cache/nanochat` by default. Override with `NANOCHAT_BASE_DIR`.
- Always set `OMP_NUM_THREADS=1` when using `torchrun`.
- Distributed helpers: `print0()` prints only on rank 0; `get_dist_info()` returns `(is_ddp, rank, local_rank, world_size)`; `compute_init()`/`compute_cleanup()` handle DDP setup/teardown.

### Contributing

AI policy: When submitting a PR, declare any parts with substantial LLM contribution that you haven't fully written or understood.

## Assignment A4 Context

**Course**: CSC490. **Goal**: Improve GSM8K results through SFT data augmentation and RL.

### A4 Structure (4 parts)

1. **Part 1 — GRPO & RL Review (10 marks)**: Compare nanochat's RL implementation to standard GRPO (Shao et al., 2024).
2. **Part 2 — SFT & Midtraining (20 marks)**: Run original SFT config as baseline, then add math-focused datasets to improve GSM8K. Compare all runs.
3. **Part 3 — Replicating RL (30 marks)**: Replicate Karpathy's GSM8K RL run. Compare reward/eval curves. Cluster correct/incorrect problems.
4. **Part 4 — Complex Rewards (40 marks)**: Design 2+ additional reward systems for GSM8K. Run separately and combined. Compare mistake patterns. Summarize in table.

### Pretrained Model

- **Checkpoint**: `nanochat_d26_diff_attn` on Modal volume `nanochat-vol`
- **Architecture**: d=26, ~830M params, differential attention (Ye et al., ICLR 2025)
- **Pretrain metrics**: val_bpb=0.7576, CORE=0.2405
- **Key constraint**: Differential attention raises `NotImplementedError` with KV cache → must use cache-free generation for generative eval

### Part 2 — SFT Experiments

Five SFT runs on `nanochat_d26_diff_attn`, each saving to `chatsft_checkpoints/<sft_tag>`:

| Run | `--sft-tag` | Flags | Total Rows |
|-----|-------------|-------|------------|
| 1 | `sft_baseline` | (none) | ~1.17M |
| 2 | `sft_metamathqa` | `--metamathqa` | ~1.57M |
| 3 | `sft_orcamath` | `--orcamath` | ~1.37M |
| 4 | `sft_dartmath` | `--dartmath` | ~1.75M |
| 5 | `sft_combo` | (top 2 from runs 2–4) | TBD |

**Added math datasets**:
- `tasks/metamathqa.py` — MetaMathQA 395K (Yu et al., arXiv:2309.12284)
- `tasks/orcamath.py` — Orca-Math 200K (Mitra et al., arXiv:2402.14830)
- `tasks/dartmath.py` — DART-Math-Hard 585K (Tong et al., arXiv:2407.13690)

**Code changes for Part 2**:
- `scripts/chat_sft.py` — added `--metamathqa`, `--orcamath`, `--dartmath`, `--sft-tag` flags
- `scripts/chat_eval.py` — added `generate_batch_no_cache()` for diff_attn models
- `nanochat/checkpoint_manager.py` — strips unknown config keys from old checkpoints (e.g. `swiglu`)
- `runs/part2_sft_modal.py` — Modal script for all 5 SFT runs + eval

### Part 2 — Execution Flow

```
Step 1: Run 4 SFT jobs (baseline + 3 math datasets)          ← DONE (detached on Modal)
Step 2: Run eval on all 4 checkpoints                         ← TODO after step 1 finishes
Step 3: Review GSM8K scores, pick top 2 math datasets         ← TODO (manual)
Step 4: Edit stage_sft_combo() flags in part2_sft_modal.py    ← TODO (currently hardcoded to --metamathqa --dartmath)
Step 5: Run combo SFT (baseline + top 2)                      ← TODO
Step 6: Run eval on combo checkpoint                          ← TODO
```

**Current status** (as of 2026-03-11):
- Step 1 launched as 4 detached Modal apps (all writing to separate dirs on `nanochat-vol`):
  - `stage_sft_baseline` → `chatsft_checkpoints/sft_baseline`
  - `stage_sft_metamathqa` → `chatsft_checkpoints/sft_metamathqa`
  - `stage_sft_orcamath` → `chatsft_checkpoints/sft_orcamath`
  - `stage_sft_dartmath` → `chatsft_checkpoints/sft_dartmath`
- Check status: `PYTHONUTF8=1 modal app list`
- All smoke tests passed locally (8/8)

**Modal commands**:
```bash
# Step 1 — already launched (detached, survives terminal close):
PYTHONUTF8=1 modal run --detach runs/part2_sft_modal.py::stage_sft_baseline
PYTHONUTF8=1 modal run --detach runs/part2_sft_modal.py::stage_sft_metamathqa
PYTHONUTF8=1 modal run --detach runs/part2_sft_modal.py::stage_sft_orcamath
PYTHONUTF8=1 modal run --detach runs/part2_sft_modal.py::stage_sft_dartmath

# Step 2 — run after all 4 SFT jobs finish:
PYTHONUTF8=1 modal run --detach runs/part2_sft_modal.py::stage_eval

# Steps 4–5 — after reviewing eval results, edit combo flags in stage_sft_combo() then:
PYTHONUTF8=1 modal run --detach runs/part2_sft_modal.py::stage_sft_combo

# Step 6 — eval the combo checkpoint:
PYTHONUTF8=1 modal run --detach runs/part2_sft_modal.py::stage_eval
```

### Windows Development Notes

- Set `PYTHONUTF8=1` before all `modal` and `python` commands (Windows charmap encoding issues)
- `torch.compile` fails on Windows CPU (no `cl` compiler) — use `TORCH_COMPILE_DISABLE=1` for local testing
- Need ≥2 data shards for local pretrain (train split = `parquet_files[:-1]`, so 1 shard = empty train set)
