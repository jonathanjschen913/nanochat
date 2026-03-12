# A4 Part 3: Replicating the RL Run with Additional Analysis (30 marks)

## Goal

Replicate Karpathy's GSM8K RL run using our best Part 2 SFT checkpoint (`sft_combo`),
compare reward/eval curves to his published results, and conduct an error analysis on
what the model gets right and wrong.

---

## Setup

### Starting Checkpoint

We use `sft_combo` — the best performing SFT model from Part 2 (baseline + MetaMathQA
+ DART-Math-Hard). This is a stronger starting point than Karpathy's vanilla SFT:

| Checkpoint | GSM8K (before RL) |
|---|---|
| Karpathy's vanilla SFT | 4.55% |
| Our `sft_combo` | **20%** |

The difference is expected — our SFT included 395K MetaMathQA examples directly
bootstrapped from GSM8K training questions.

### RL Configuration

All hyperparameters match Karpathy's defaults in `scripts/chat_rl.py`, with mid-training
eval disabled (`--eval-every=9999`) to reduce cost — `stage_eval` runs a full evaluation
separately after training completes:

| Parameter | Value |
|---|---|
| `--num-epochs` | 1 |
| `--examples-per-step` | 16 |
| `--num-samples` | 16 |
| `--max-new-tokens` | 256 |
| `--temperature` | 1.0 |
| `--top-k` | 50 |
| `--device-batch-size` | 8 |
| `--eval-every` | 60 steps |
| `--eval-examples` | 400 |
| GPU | 8×H100 |

### Differential Attention KV Cache Fix

Our model uses differential attention (Ye et al., ICLR 2025). The original code raised
`NotImplementedError` when KV cache inference was attempted with diff_attn, requiring
an O(T²) cache-free fallback that made each RL step ~10-20x slower than Karpathy's run.

We implemented KV cache support for differential attention in `nanochat/gpt.py`. The key
insight is that `self.n_kv_head` is halved in diff_attn (e.g. 13 instead of 26), so
`2 × self.n_kv_head == config.n_kv_head` — the existing `KVCache` allocation is already
the right size. We partition the cache heads:

- First `n_kv_head` slots → k1 / v1
- Last  `n_kv_head` slots → k2 / v2

During each decode step, new k1/k2/v1/v2 tokens are written into their respective cache
slots, then attention is computed over the full cached sequence using `flash_attn_func`.
No changes were needed to `KVCache` or `Engine`. This restores O(T) per-token generation,
bringing RL step time in line with the standard attention baseline.

---

## RL Algorithm: What Nanochat Does vs Standard GRPO

*(See also Part 1 for the full comparison.)*

Standard GRPO (Shao et al., 2024) maintains a reference model and penalises KL divergence
from it, uses PPO-style importance sampling with ratio+clip, and normalises advantages with
z-score `(r - μ) / σ` at sequence level.

Nanochat's RL simplifies this to near-REINFORCE:

1. **No KL regularization** — reference model removed entirely. On-policy training makes
   the ratio always 1, so the clip is also unnecessary.
2. **No PPO clip** — we are always on-policy; importance weights are trivially 1.
3. **Token-level DAPO normalization** — advantage applied per-token rather than
   per-sequence, following DAPO (Yu et al., 2024).
4. **Advantage = `r - μ`** — subtract group mean reward but do not divide by σ.
   This avoids instability when all samples in a group get the same reward (σ → 0).

The reward itself is binary: 1 if the model's `#### <answer>` matches the ground truth,
0 otherwise (`tasks/gsm8k.py`, `reward()`).

---

## Results

### Reward and Pass@k Curves

*[Plots exported from W&B — insert here after run completes]*

| Metric | Karpathy (vanilla SFT → RL) | Our Run (`sft_combo` → RL) |
|---|---|---|
| GSM8K before RL | 4.55% | 20% |
| GSM8K after RL | **7.58%** | **[fill after run]** |
| RL training time | ~1.5 hrs | ~2-3 hrs (diff_attn overhead) |
| Pass@1 at step 0 | 4.55% (after SFT) | **11.5%** (stronger SFT start) |
| Pass@8 at step 0 | N/A | **33.25%** |
| Pass@8 >> Pass@1? | Yes | Yes (33.25% vs 11.5% at step 0) |
| GSM8K after RL | 7.58% | [fill after run] |

**Expected differences and reasons:**

1. **Higher starting and ending pass@1**: Our `sft_combo` already encodes GSM8K-specific
   reasoning patterns from MetaMathQA. RL starts from a higher floor and should converge
   to a higher ceiling.
2. **Reward curve shape**: Karpathy observed a clear upward trend in mean reward over
   training. We expect a similar shape but potentially faster initial rise given the
   stronger SFT starting point.
3. **Pass@8 gap**: Karpathy noted pass@8 >> pass@1, indicating the model often knows
   the right approach but fails to execute it consistently. We will check whether our
   model shows the same pattern or a reduced gap due to stronger SFT.

---

## Error Analysis

### Methodology

After RL training, we run `stage_collect_completions` to generate the model's response
to every problem in the GSM8K test set (1,319 problems) at temperature=0, saving each
completion with its correctness label to `part3_completions.jsonl`.

We then categorize incorrect responses into error types.

### Error Categories

Based on manual inspection and pattern matching on the completions, we group errors into:

| Category | Description | Detection |
|---|---|---|
| **Arithmetic error** | Correct setup, wrong calculation | Has `####`, answer off by small amount |
| **Wrong setup** | Incorrect equation or misread problem | Has `####`, large numerical gap from gold |
| **Missing answer** | No `#### <number>` in output | `extract_answer()` returns `None` |
| **Truncated** | Response cut off before `####` | Completion fills full 256 tokens |
| **Unit/scale error** | Correct logic, wrong unit (e.g. hours vs minutes) | Gold is a round multiple of answer |
| **Correct** | Answer matches gold | `is_correct = True` |

*[Fill in actual counts and percentages after run completes]*

### Exploratory Data Analysis

#### Problem Length vs Accuracy

*[Plot: histogram of question word count split by correct/incorrect]*

Hypothesis: longer problems with more steps are harder. We expect accuracy to decrease
as problem length increases, since our 256-token generation budget limits multi-step
reasoning chains.

#### Answer Magnitude Distribution

*[Plot: distribution of gold answer values for correct vs incorrect problems]*

Large answers (e.g. in the thousands) require more arithmetic steps. We expect the
model to be more accurate on problems with small final answers (< 100) vs large ones.

#### Error Category Breakdown

*[Pie/bar chart of error categories]*

The most common error type informs Part 4's reward design:
- If **missing answer** dominates → add a format reward for producing `####`
- If **arithmetic errors** dominate → add partial credit for correct intermediate steps
- If **truncated** dominates → increase `--max-new-tokens` or add length reward

#### Problem Clusters

Using keyword clustering on the problem text, we group problems into domains:
- **Rate/time problems** (e.g. speed, hourly wages)
- **Multi-step counting** (e.g. people, objects across several conditions)
- **Percentage/fraction** problems
- **Geometry/area** problems

*[Plot: accuracy by problem cluster]*

This identifies which problem types the model handles well vs poorly, directly motivating
the reward shaping in Part 4.

---

## Commentary on Differences from Karpathy's Run

**Architecture**: Our model uses differential attention (Ye et al., ICLR 2025) vs
Karpathy's standard multi-head attention. Differential attention is designed to suppress
noise in attention maps and improve focus on relevant tokens — this may help on multi-step
arithmetic where attending to the right intermediate value matters.

**SFT data**: Our combo SFT included MetaMathQA (395K examples bootstrapped from GSM8K)
and DART-Math-Hard (585K hard problems with CoT). This gives the RL phase a much stronger
prior over GSM8K-style reasoning.

**Cache-free generation**: Our rollout generation is O(T²) per step vs O(T) with KV cache.
This makes our RL runs slower per step but produces identical gradient signal. We mitigate
this by using 8×H100 GPUs.

---

## Cost

| Stage | GPU | Time | Cost |
|---|---|---|---|
| RL training (467 steps, evals disabled, 8×H100) | 8×H100 | ~2-3 hrs | ~\$56-84 |
| Completion collection (1319 examples) | 4×H100 | ~20 min | ~\$5 |
| Final eval (pass@1, pass@8) | 4×H100 | ~20 min | ~\$5 |
| **Total** | | **~130 min** | **~\$52** |

---

## Running the Pipeline

```bash
# Full pipeline
modal run runs/part3_rl_modal.py

# Individual stages
modal run runs/part3_rl_modal.py::stage_rl_baseline
modal run runs/part3_rl_modal.py::stage_collect_completions
modal run runs/part3_rl_modal.py::stage_eval
```

Completions saved to: `$NANOCHAT_BASE_DIR/part3_completions.jsonl`
W&B project: `nanochat-part3`

---

## Key Files

| File | Role |
|---|---|
| `nanochat/gpt.py` | KV cache support for diff_attn — partitions cache heads into k1/v1 and k2/v2 slots |
| `scripts/chat_rl.py` | RL training script — uses Engine directly (KV cache now works for diff_attn) |
| `tasks/gsm8k.py` | GSM8K task + binary reward (`reward()` returns 0 or 1) |
| `runs/part3_rl_modal.py` | Modal script: RL train + collect completions + eval |
| `dev/a4_part3_writeup.md` | This writeup |
