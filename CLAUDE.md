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
| 5 | `sft_combo` | `--metamathqa --dartmath` | ~1.75M |

**Added math datasets**:
- `tasks/metamathqa.py` — MetaMathQA 395K (Yu et al., arXiv:2309.12284)
- `tasks/orcamath.py` — Orca-Math 200K (Mitra et al., arXiv:2402.14830)
- `tasks/dartmath.py` — DART-Math-Hard 585K (Tong et al., arXiv:2407.13690)

**Eval results (runs 1–4, n=50 per task, cache-free generation)**:

| Tag | ARC-E | ARC-C | MMLU | GSM8K | HumanEval | SpellingBee |
|-----|-------|-------|------|-------|-----------|-------------|
| sft_baseline | 56% | 42% | 34% | 4% | 2% | 100% |
| sft_metamathqa | 62% | 40% | 34% | **18%** | 4% | 100% |
| sft_orcamath | 60% | 38% | 40% | 0% | 2% | 100% |
| sft_dartmath | 64% | 28% | **44%** | 0% | 4% | 100% |
| **sft_combo** | 52% | **46%** | 38% | **20%** | **6%** | 100% |

**Key finding**: Combo (MetaMathQA + DART-Math) achieved best GSM8K (20%), ARC-C (46%), HumanEval (6%). Minor ARC-E regression.

**Code changes for Part 2**:
- `scripts/chat_sft.py` — added `--metamathqa`, `--orcamath`, `--dartmath`, `--sft-tag` flags
- `scripts/chat_eval.py` — added `generate_batch_no_cache()` for diff_attn models
- `nanochat/checkpoint_manager.py` — strips unknown config keys from old checkpoints (e.g. `swiglu`)
- `runs/part2_sft_modal.py` — Modal script for all 5 SFT runs + eval (stage_eval accepts `--tags` to filter checkpoints)

### Part 2 — Status: COMPLETE

All 6 steps finished. See `dev/a4_part2_writeup.md` for full analysis and discussion.

### Part 2 — Key Files

| File | Role |
|------|------|
| `tasks/metamathqa.py` | MetaMathQA 395K dataset loader (query→user, response→assistant) |
| `tasks/orcamath.py` | Orca-Math 200K dataset loader (question→user, answer→assistant) |
| `tasks/dartmath.py` | DART-Math-Hard 585K dataset loader (query→user, response→assistant) |
| `scripts/chat_sft.py` | SFT training script — added `--metamathqa`, `--orcamath`, `--dartmath`, `--sft-tag` flags |
| `scripts/chat_eval.py` | Eval script — added `generate_batch_no_cache()` for diff_attn models |
| `nanochat/checkpoint_manager.py` | Strips unknown config keys from old checkpoints (e.g. `swiglu`) |
| `runs/part2_sft_modal.py` | Modal script for all 5 SFT runs + eval (`stage_eval` accepts `--tags` to filter) |
| `dev/a4_part2_writeup.md` | Full writeup: dataset justifications, results, analysis |
| `dev/math_datasets_research.md` | Background research on 10 candidate math datasets |

### Part 2 — Modal Artifacts

All checkpoints live on Modal volume `nanochat-vol` under `chatsft_checkpoints/`:
- `sft_baseline`, `sft_metamathqa`, `sft_orcamath`, `sft_dartmath`, `sft_combo`

W&B project: `nanochat-sft` (user: `iamthebest`)

### Handoff for Parts 3 & 4

**What Parts 3/4 need from Part 2:**
- The RL script (`scripts/chat_rl.py`) loads an SFT checkpoint via `load_model("sft", ...)`. Use `--model-tag=<sft_tag>` to select which SFT checkpoint to start RL from (e.g. `--model-tag=sft_combo` for the best Part 2 model).
- SFT checkpoints are stored on Modal volume `nanochat-vol` at `chatsft_checkpoints/<sft_tag>/`.
- The recommended starting point for RL is `sft_combo` (best GSM8K: 20%).

**Existing RL code:**
- `scripts/chat_rl.py` — GRPO-style RL on GSM8K. Simplified REINFORCE: no KL penalty, no PPO clipping, token-level DAPO normalization, advantage = `(r - mu)` without sigma.
- `tasks/gsm8k.py` — contains `GSM8K.reward()` which returns 1 (correct) or 0 (incorrect) by comparing extracted numerical answers. This is the baseline reward for Part 4.
- `runs/part3_rl_modal.py` — Modal script for Part 3 RL run. Stages: `stage_rl_baseline`, `stage_collect_completions`, `stage_eval`, `stage_read_completions`.

**Differential attention KV cache — FIXED:**
- `nanochat/gpt.py` now supports KV cache for diff_attn. Cache heads are partitioned: first `n_kv_head` slots = k1/v1, last `n_kv_head` slots = k2/v2. No changes needed to `KVCache` or `Engine`.
- `generate_batch_no_cache()` fallback is no longer needed for RL. `Engine.generate_batch()` works directly.
- Speedup: ~10-20x vs cache-free generation (O(T) vs O(T²)).

**Part 3 run status:**
- Run 1 (killed at step ~68, budget): reward 0.22→0.59, Pass@1=11.5% at step 0. W&B: `nanochat-rl/nanochat-part3_rl_baseline`.
- Mid-training evals disabled (`--eval-every=9999`) to save cost. Run `stage_eval` separately after training.
- Budget constraint: ~$0.28/step on 8×H100. Full 467-step run costs ~$130. Kill at step ~250 to stay under $80.

**Known issues to watch for:**
- `swiglu` key in old checkpoints — already fixed in `checkpoint_manager.py`
- Windows: always prefix commands with `PYTHONUTF8=1`; use `TORCH_COMPILE_DISABLE=1` for local CPU testing
- `torch.compile` fails on Windows (no `cl` compiler)
