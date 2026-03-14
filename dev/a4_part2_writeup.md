# A4 Part 2: SFT & Midtraining (20 marks)

## Goal

Improve GSM8K performance through SFT data augmentation on our pretrained nanochat d=26 model (with differential attention from A3 Part 4). We run the original SFT configuration as a baseline, then augment the training mixture with math-focused datasets and compare results.

## Pretrained Model

- **Checkpoint**: `nanochat_d26_diff_attn` (from A3 Part 4)
- **Architecture**: d=26, ~830M params, differential attention, RoPE, GQA, QK-norm, ReLU²
- **Pretrain metrics**: val_bpb=0.7576, CORE=0.2405

## Baseline SFT Mixture (Original `chat_sft.py`)

| Dataset | Size | Purpose |
|---------|------|---------|
| SmolTalk | 460K | General conversations |
| Identity conversations | 1K × 2 epochs | Personality |
| MMLU auxiliary train | 100K × 3 epochs | Multiple choice reasoning |
| GSM8K train | 8K × 4 epochs | Math + tool use |
| SimpleSpelling | 200K | Spelling |
| SpellingBee | 80K | Letter counting |
| **Total** | **~1.17M rows** | |

## Additional Math Datasets

We select three math SFT datasets to test as augmentations, chosen based on our research in `dev/math_datasets_research.md`. Selection criteria: (1) directly targets GSM8K-style problems, (2) plain-text CoT format compatible with nanochat, (3) manageable size (200K–600K) to avoid catastrophic forgetting at 830M params.

### MetaMathQA (Yu et al., arXiv:2309.12284)

- **HuggingFace**: `meta-math/MetaMathQA`
- **Size**: 395K rows
- **License**: MIT
- **Description**: Augmented math Q&A bootstrapped from GSM8K and MATH train sets via GPT-3.5. Uses question rephrasing, answer augmentation, self-verification, and backward reasoning (FOBAR).
- **Justification**: Directly derived from GSM8K questions with diverse rephrasings — the most targeted augmentation for our GSM8K goal. MetaMath-7B achieved 66.5% GSM8K (vs 14.6% base LLaMA-2-7B).

### Orca-Math (Mitra et al., arXiv:2402.14830)

- **HuggingFace**: `microsoft/orca-math-word-problems-200k`
- **Size**: 200K rows
- **License**: MIT
- **Description**: Synthetic grade-school math word problems created via multi-agent GPT-4 Turbo pipeline (agent generates, teacher verifies/refines).
- **Justification**: Specifically designed for small language models on grade-school math. A 7B model trained *only* on this 200K dataset surpassed LLaMA-2-70B and ChatGPT-3.5 on GSM8K. Smallest dataset in our set — tests whether quality > quantity at 830M params.

### DART-Math-Hard (Tong et al., arXiv:2407.13690, NeurIPS 2024)

- **HuggingFace**: `hkust-nlp/dart-math-hard`
- **Size**: 585K rows
- **License**: MIT
- **Description**: Created via Difficulty-Aware Rejection Tuning (DARS), which allocates more generation attempts to harder problems from GSM8K and MATH train sets.
- **Justification**: Oversamples problems that models struggle with — addresses the failure mode where easy problems dominate training. Largest dataset in our set — tests whether hard-problem oversampling helps at 830M scale.

## Experimental Runs

| Run | Tag | Mixture | Total Rows (approx) |
|-----|-----|---------|---------------------|
| 1 | `sft_baseline` | Original SFT | ~1.17M |
| 2 | `sft_metamathqa` | Original + MetaMathQA | ~1.57M |
| 3 | `sft_orcamath` | Original + Orca-Math | ~1.37M |
| 4 | `sft_dartmath` | Original + DART-Math-Hard | ~1.75M |
| 5 | `sft_combo` | Original + MetaMathQA + DART-Math-Hard | ~1.75M |

All runs use:
- Same pretrained checkpoint: `nanochat_d26_diff_attn`
- Same training config: `--device-batch-size=8` (d=26 OOM at 16 during SFT)
- Same optimizer settings inherited from pretrain checkpoint
- W&B project: `nanochat-part2`

Run 5 combines MetaMathQA (the only dataset that improved GSM8K) with DART-Math-Hard (strongest non-GSM8K gains: best ARC-Easy and MMLU).

## Implementation Changes

### New Task Classes

Three new files in `tasks/`, following the `SmolTalk` pattern:
- `tasks/metamathqa.py` — `MetaMathQA(split="train")`, fields: `query` → user, `response` → assistant
- `tasks/orcamath.py` — `OrcaMath(split="train")`, fields: `question` → user, `answer` → assistant
- `tasks/dartmath.py` — `DARTMath(split="train")`, fields: `query` → user, `response` → assistant

### Modified `scripts/chat_sft.py`

Added CLI flags:
- `--metamathqa` — include MetaMathQA in training mixture
- `--orcamath` — include Orca-Math in training mixture
- `--dartmath` — include DART-Math-Hard in training mixture
- `--sft-tag` — override output checkpoint directory name

Datasets are conditionally appended to `train_tasks` before `TaskMixture` construction.

### Cache-Free Generative Eval (`scripts/chat_eval.py`)

Our differential attention model raises `NotImplementedError` when KV cache is used, blocking generative evaluation (GSM8K, HumanEval, SpellingBee). We added:

- `generate_batch_no_cache()` — full-sequence forward pass per token (no KV cache), with tool use support (calculator via python_start/python_end detection). Same interface as `Engine.generate_batch()`.
- `run_generative_eval()` auto-detects `model.config.differential_attn` and routes to cache-free generation when needed.

This is ~O(T²) slower than KV-cached generation but produces identical results. We use `--max-problems=50` for generative tasks to keep eval tractable.

### Modal Script (`runs/part2_sft_modal.py`)

Follows `nanochat_modal.py` patterns. Stages:
- `stage_sft_baseline` through `stage_sft_dartmath` — individual SFT runs
- `stage_sft_combo` — run after analyzing results
- `stage_eval` — evaluates all available SFT checkpoints on ChatCORE

## Results

*To be filled after runs complete.*

### ChatCORE Evaluation

| Run | Tag | ARC-Easy | ARC-Challenge | MMLU | GSM8K | HumanEval | SpellingBee |
|-----|-----|----------|---------------|------|-------|-----------|-------------|
| 1 | sft_baseline | 56% | 42% | 34% | 4% | 2% | 100% |
| 2 | sft_metamathqa | 62% | 40% | 34% | **18%** | 4% | 100% |
| 3 | sft_orcamath | 60% | 38% | 40% | 0% | 2% | 100% |
| 4 | sft_dartmath | 64% | 28% | **44%** | 0% | 4% | 100% |
| 5 | sft_combo | 52% | **46%** | 38% | **20%** | **6%** | 100% |

All evaluations use `--max-problems=50` with cache-free generation (differential attention).

### Training Metrics

| Run | Tag | val_bpb | Training Time | Total Rows |
|-----|-----|---------|---------------|------------|
| 1 | sft_baseline | | | ~1.17M |
| 2 | sft_metamathqa | | | ~1.57M |
| 3 | sft_orcamath | | | ~1.37M |
| 4 | sft_dartmath | | | ~1.75M |
| 5 | sft_combo | | | ~1.75M |

## Analysis (Runs 1–4)

### 1. Does any single math dataset meaningfully improve GSM8K over baseline?

**Yes, but only MetaMathQA.** It improved GSM8K from 4% to 18% (+14pp). Orca-Math and DART-Math-Hard both scored 0%, *worse* than baseline. This suggests that MetaMathQA's direct bootstrapping from GSM8K train questions (via rephrasing, FOBAR, etc.) transfers far more effectively than general math problem sets at 830M scale.

### 2. Does adding math data hurt other tasks (catastrophic forgetting)?

**Minimal forgetting observed.** All math-augmented runs maintained or improved ARC-Easy (56% → 60–64%) and MMLU (34% → 34–44%). ARC-Challenge dropped for DART-Math (42% → 28%), suggesting some trade-off with the largest augmentation. SpellingBee remained at 100% across all runs. HumanEval stayed near floor (2–4%) regardless.

### 3. Does quality (Orca-Math, 200K) or quantity (DART-Math, 585K) matter more?

**Neither helped GSM8K at 830M params.** Orca-Math (quality-focused, 200K) and DART-Math (hard-problem oversampling, 585K) both scored 0% on GSM8K. The key differentiator appears to be *domain alignment*: MetaMathQA is bootstrapped directly from GSM8K questions, while the others generate novel problems. At small model scale, direct augmentation of the target distribution matters more than general math coverage.

### 4. Non-GSM8K benefits

DART-Math showed the strongest gains on MMLU (+10pp) and ARC-Easy (+8pp), suggesting its hard-problem diversity helps general reasoning even if not GSM8K specifically. This motivated its selection for the combo run alongside MetaMathQA.

### Combo Run Rationale

We combine MetaMathQA (the only GSM8K improver) with DART-Math-Hard (strongest general reasoning gains). The hypothesis: MetaMathQA provides GSM8K-targeted signal while DART-Math provides broad math reasoning. The risk: the larger combined mixture (~1.75M rows) may dilute MetaMathQA's GSM8K benefit.

## Analysis (Run 5 — Combo)

### 5. Does combining datasets preserve MetaMathQA's GSM8K gains?

**Yes — and slightly improves them.** The combo achieved 20% GSM8K, edging out MetaMathQA alone (18%). DART-Math's additional math reasoning data did not dilute the GSM8K signal from MetaMathQA.

### 6. Does the combo achieve "best of both worlds" or average out?

**Mostly best of both worlds, with one trade-off.** The combo achieved the best scores across GSM8K (20%), ARC-Challenge (46%), and HumanEval (6%). MMLU landed between MetaMathQA (34%) and DART-Math (44%) at 38%, which is a mild averaging effect. However, ARC-Easy unexpectedly dropped to 52% (below even baseline's 56%), suggesting some interference in easy factual recall.

### Overall Summary

The combo run (`sft_combo = baseline + MetaMathQA + DART-Math-Hard`) is our best overall model:
- **GSM8K +16pp** over baseline (4% → 20%) — the primary goal
- **ARC-Challenge +4pp** over baseline (42% → 46%)
- **HumanEval +4pp** over baseline (2% → 6%)
- **MMLU +4pp** over baseline (34% → 38%)
- **ARC-Easy -4pp** vs baseline (56% → 52%) — minor regression
- **SpellingBee** unchanged at 100%

At 830M params with only 1 epoch of SFT, a +16pp GSM8K improvement from data augmentation alone (no RL) is a meaningful result. The finding that only MetaMathQA (directly bootstrapped from GSM8K) improved GSM8K, while general math datasets did not, highlights the importance of domain-specific augmentation at small model scale.

## Cost Estimate

| Item | Time (8×H100) | Approx Cost |
|------|---------------|-------------|
| SFT Run 1 (baseline, ~1.17M rows) | ~30 min | ~\$14 |
| SFT Run 2 (+MetaMath, ~1.57M rows) | ~40 min | ~\$19 |
| SFT Run 3 (+Orca, ~1.37M rows) | ~35 min | ~\$16 |
| SFT Run 4 (+DART, ~1.75M rows) | ~45 min | ~\$21 |
| SFT Run 5 (combo, TBD) | ~50 min | ~\$23 |
| Eval (5 models, cache-free gen, 4×H100) | ~150 min | ~\$35 |
| **Total** | **~6 hrs** | **~\$128** |

## Running the Pipeline

```bash
# Full pipeline (runs 1–4 + eval):
modal run runs/part2_sft_modal.py

# Individual stages:
modal run runs/part2_sft_modal.py::stage_sft_baseline
modal run runs/part2_sft_modal.py::stage_sft_metamathqa
modal run runs/part2_sft_modal.py::stage_sft_orcamath
modal run runs/part2_sft_modal.py::stage_sft_dartmath

# After reviewing runs 2–4, edit combo flags in part2_sft_modal.py then:
modal run runs/part2_sft_modal.py::stage_sft_combo

# Evaluate all checkpoints:
modal run runs/part2_sft_modal.py::stage_eval
```

Code changes: `tasks/metamathqa.py`, `tasks/orcamath.py`, `tasks/dartmath.py`, `scripts/chat_sft.py`, `scripts/chat_eval.py`, `runs/part2_sft_modal.py`
