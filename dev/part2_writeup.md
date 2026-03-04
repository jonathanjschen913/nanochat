# Part 2: Ablations on a Nano Nanochat

## Overview

We select two architecture changes absent from the Feb 2026 nanochat codebase — **Cross-Layer Attention (CLA)** (Brandon et al., NeurIPS 2024) and **Shared FFN** (MobiLlama, Thawakar et al., 2024) — and measure their impact on a small nanochat configuration we call **picochat**. Both require real implementation in `gpt.py`. We train four models: a baseline, each change in isolation, and both combined, comparing val_bpb and CORE score against the same d=12 baseline.

---

## Picochat Configuration and Justification

We define picochat as a depth=12 nanochat model with the following hyperparameters:

| Hyperparameter | Value | Rationale |
|---|---|---|
| `--depth` | 12 | Matches reference `quick_test` depth (~85M non-embedding params); gives 6 CLA-2 sharing pairs. Preliminary d=8 runs showed CLA hurt quality, motivating the switch to d=12 |
| `--aspect-ratio` | 64 | nanochat default; model_dim = 12 × 64 = 768 |
| `--head-dim` | 64 | Speedrun default is 128, but at model_dim=768 that gives only 6 heads (768/128=6). head-dim=64 gives 12 heads, providing richer head diversity for CLA-2 sharing |
| `--max-seq-len` | 512 | Sufficient for document-level context at this scale |
| `--window-pattern` | L | Full attention; appropriate for short sequences |
| `--device-batch-size` | 16 | Matches reference speedrun `DEVICE_BATCH_SIZE=16` |
| Training horizon | Chinchilla (10.5×) | Automatically computed from parameter count; mirrors reference speedrun approach |
| Data | 80 FineWeb-EDU shards | Chinchilla-optimal token budget for ~85M params ≈ 892M tokens ≈ 20 shards needed. 80 shards provides comfortable headroom |
| GPU | 8×H100 | Matches reference speedrun `GPU_PRETRAIN`; eval runs on 4×H100 to halve eval cost |

**Justification for d=12:** We initially ran preliminary ablations at d=8 to validate the pipeline cheaply before committing to full runs. The d=8 results showed CLA hurting quality — val_bpb increased by 0.023 and CORE dropped from 0.066 to 0.057. d=12 is the reference `quick_test` depth and provides 6 CLA sharing pairs (vs 4 at d=8), giving CLA a fairer test.

**Why not larger?** d=16+ would give stronger statistical signal but each run costs significantly more on 8×H100. d=12 is the sweet spot between cost and meaningful depth.

### Preliminary results at d=8 (pilot runs)

These runs validated the pipeline and motivated the switch to d=12:

| Model | val_bpb | CORE | Interpretation |
|---|---|---|---|
| pico_baseline (d=8) | 1.027 | 0.0660 | Control |
| pico_cla (d=8) | 1.050 | 0.0574 | CLA hurts at d=8 |

---

## The Four Models

### Model 1: pico_baseline
**Config:** d=12, model_dim=768, 12 heads, ReLU² activation, independent KV per layer, per-layer MLPs.

The control model. Matches the picochat configuration above with no modifications — identical to the Feb 2026 nanochat architecture at this scale. All other models are compared against this checkpoint.

### Model 2: pico_shared_ffn
**Config:** d=12, model_dim=768, 12 heads, ReLU² activation, **one MLP shared across all 12 layers**.

Implements MobiLlama's (Thawakar et al., 2024) FFN weight sharing. Instead of per-layer MLPs, a single MLP is instantiated at the `GPT` level and reused by every block:

```
baseline d=12:      12 × (2.36M attn + 4.72M MLP) = 84.9M params
shared_ffn d=12:    12 × 2.36M attn  + 1 × 4.72M MLP = ~33M params
```

```python
# In GPT.__init__:
self.shared_mlp = MLP(config)  # one MLP for all layers

# In GPT.forward:
for block in self.transformer.h:
    x = block(x, ..., mlp=self.shared_mlp)  # same weights every layer
```

This is an **iso-depth** (not iso-parameter) comparison: same number of layers as baseline, but with 52M fewer parameters due to MLP sharing. The reduced parameter count means any quality difference conflates weight sharing with raw model capacity. Training horizon set by Chinchilla ratio based on the actual ~33M param count.

### Model 3: pico_cla
**Config:** d=12, model_dim=768, 12 heads, ReLU² activation, **CLA-2 KV sharing**.

Implements Cross-Layer Attention with a sharing factor of 2 (Brandon et al., NeurIPS 2024). Even-numbered layers reuse the K and V tensors computed by the preceding odd-numbered layer, computing only Q themselves. This halves the number of independent KV projections in the model:

```python
# In GPT.forward():
shared_kv = None
for i, block in enumerate(self.transformer.h):
    if cla_sharing > 1 and i % cla_sharing != 0:
        # CLA follower: reuse K and V from previous leader layer
        x = block(x, ve, cos_sin, window_size, kv_cache, shared_kv=shared_kv)
    elif cla_sharing > 1:
        # CLA leader: compute fresh K/V and cache them
        x, k, v = block(x, ve, cos_sin, window_size, kv_cache, return_kv=True)
        shared_kv = (k, v)
    else:
        x = block(x, ve, cos_sin, window_size, kv_cache)
```

All other hyperparameters are identical to the baseline. The change is isolated to `gpt.py`. CLA follower layers do not create `c_k`/`c_v` weight matrices, reducing the parameter count by ~7M.

### Model 4: pico_cla_shared_ffn
**Config:** d=12, model_dim=768, 12 heads, **CLA-2 KV sharing + shared FFN**.

Both changes applied simultaneously at d=12. Determines whether the changes are independent, synergistic, or interfering. Since CLA operates on attention and shared FFN operates on the MLP, there is no direct architectural interaction — independence is the prior expectation. Parameter count is the smallest of all four models: 12 × 2.36M attn (with CLA halving KV projections) + 1 × 4.72M MLP ≈ ~26M params.

---

## Training Setup

All models were trained on Modal using 8×H100 GPUs on 80 FineWeb-EDU shards with a shared BPE tokenizer trained on 2B characters (matching the reference speedrun.sh tokenizer exactly). Training was tracked using Weights & Biases under the project `picochat-ablation`. All four models use d=12 with Chinchilla-auto training horizon (`--target-param-data-ratio=10.5`); training steps differ between models since Chinchilla scales with parameter count.

**Run commands:**
```bash
# Local smoke test (verify code runs, ~3-5 min)
bash runs/runpico_test.sh

# Cloud training (full ablation)
modal run runs/pico_ablation_modal.py

# Individual stages
modal run runs/pico_ablation_modal.py::stage_pretrain_baseline
modal run runs/pico_ablation_modal.py::stage_pretrain_cla
modal run runs/pico_ablation_modal.py::stage_pretrain_shared_ffn
modal run runs/pico_ablation_modal.py::stage_pretrain_cla_shared_ffn
modal run runs/pico_ablation_modal.py::stage_eval
```

---

## Results

### Main results

| Model | depth | FFN | KV sharing | Params (non-emb) | val_bpb ↓ | CORE ↑ | vs baseline |
|---|---|---|---|---|---|---|---|
| pico_baseline | 12 | per-layer | None (MHA) | ~85M | **0.9250** | **0.1273** | — |
| pico_cla | 12 | per-layer | CLA-2 | ~78M | **0.9447** | **0.0655** | +0.020 bpb, -49% CORE |
| pico_shared_ffn | 12 | shared (1×) | None (MHA) | ~33M | **1.0058** | **0.0810** | +0.081 bpb, -36% CORE |
| pico_cla_shared_ffn | 12 | shared (1×) | CLA-2 | ~26M | **1.0410** | **0.0575** | +0.116 bpb, -55% CORE |
| GPT-2 target | — | — | — | ~1.5B | ~0.748 | 0.2565 | — |

**Note on CORE scores:** At picochat scale, CORE scores are well below the GPT-2 threshold of 0.256525. Relative differences between models are what matter.

**Note on parameter counts:** pico_shared_ffn (~33M) and pico_cla_shared_ffn (~26M) have significantly fewer parameters than the baseline (~85M) because MLP weights are shared across all layers rather than per-layer. This is an iso-depth comparison, not iso-parameter. Any quality difference partially reflects fewer parameters, not only the effect of weight sharing.

### Preliminary results at d=8 (pilot runs)

| Model | val_bpb | CORE |
|---|---|---|
| pico_baseline (d=8) | 1.027 | 0.066 |
| pico_cla (d=8) | 1.050 | 0.057 |

CLA degraded quality at d=8, motivating the switch to d=12 for the main ablation.

---

## Commentary on Results

### CLA (pico_cla vs pico_baseline)

CLA-2 achieved val_bpb of **0.9447** vs the baseline's **0.9250** — a small degradation of **+0.020 bpb (+2.1%)**. CORE dropped from **0.1273 to 0.0655 (-49%)**. This is a much milder result than the d=8 pilot:

| Depth | Baseline val_bpb | CLA val_bpb | Δbpb | Baseline CORE | CLA CORE | ΔCORE |
|---|---|---|---|---|---|---|
| d=8 | 1.027 | 1.050 | +0.023 | 0.066 | 0.057 | -14% |
| d=12 | 0.925 | 0.945 | +0.020 | 0.127 | 0.066 | -49% |

Brandon et al. (2024) reported roughly equal perplexity to a full-KV baseline at 1B scale with CLA-2. Our val_bpb degradation at d=12 is small (+2.1%), consistent with near quality-neutrality. However, the CORE drop (-49%) is large relative to the bpb change, suggesting the model's benchmark reasoning ability is more sensitive to CLA than raw language modelling loss.

**Nanochat-specific consideration:** nanochat's **value embeddings** add a token-ID-indexed embedding directly to the V tensor before attention. When CLA follower layers reuse the leader's V, they inherit representations shaped by the leader's shallower residual stream. This Q/KV depth mismatch may explain the CORE degradation. Brandon et al.'s experiments used vanilla transformers without value embeddings, so their quality-neutrality finding may not transfer directly.

The parameter reduction from CLA is real (~7M fewer params from removed c_k/c_v on follower layers) with only a small bpb cost, suggesting CLA could be worth investigating at larger scale with value-embedding-aware pairing.

### Shared FFN (pico_shared_ffn vs pico_baseline)

The shared FFN variant achieved val_bpb of **1.0058** vs the baseline's **0.9250** (+8.7%), and CORE of **0.0810** vs **0.1273** (-36%).

This is an **iso-depth** comparison: both models use d=12 and the same Chinchilla training budget, but pico_shared_ffn has only ~33M parameters vs the baseline's ~85M. The quality degradation therefore reflects two confounded factors: (1) the effect of MLP weight sharing, and (2) fewer total parameters. It is not possible to isolate weight sharing as the sole cause.

MobiLlama reported 0.5B models with shared FFN achieving a 2.4% average gain over comparable SLMs, but their comparison was iso-parameter (freed MLP params reinvested in depth). A fair iso-parameter test would require d=34 shared_ffn (~85M params), which is significantly more expensive to train to convergence.

**Why shared FFN degrades at iso-depth:** When the MLP is shared, each layer uses identical feed-forward weights regardless of position in the network. The early layers and late layers of a transformer process fundamentally different representations — early layers handle local syntactic patterns while later layers handle semantic abstractions. A single shared MLP cannot specialise for both. nanochat's residual scalars (`resid_λ`, `x0_λ`) provide per-layer output scaling that partially mitigates this, but cannot replace the expressivity of per-layer MLP weights.

### Combined (pico_cla_shared_ffn vs pico_baseline)

The combined model performs worse than either change in isolation (val_bpb 1.0410, +12.5%). Since both CLA and shared FFN individually degrade quality at this scale, their combination degrades further. There is no synergy — the changes appear to be independently harmful, compounding additively. This is consistent with their operating on separate architectural components (attention vs MLP).

---

## W&B Visualisation

All four training runs are tracked in W&B under the project `picochat-ablation`. The primary plots to examine:

1. **val_bpb vs step** — main quality comparison; all 4 runs on the same axis
2. **val_bpb vs total_training_flops** — iso-FLOP comparison (more honest than iso-step since CLA and deep models have different FLOPs per step)
3. **CORE score** — evaluated once after training for each model; plotted as a bar chart
4. **tok/sec** — throughput; CLA should be faster due to fewer KV projections
5. **train/mfu** — model FLOP utilisation

*[Insert W&B screenshot here.]*

---

## Cost of Training

| Item | Time | GPUs | Cost |
|---|---|---|---|
| d=8 pilot runs (baseline + cla) | ~20 min | 8×H100 | ~$10.40 |
| Data + tokenizer | ~10 min | CPU / 1×H100 | ~$5.30 |
| pico_baseline pretrain (d=12, Chinchilla) | ~40 min | 8×H100 | ~$18.70 |
| pico_cla pretrain (d=12, Chinchilla) | ~40 min | 8×H100 | ~$18.70 |
| pico_shared_ffn pretrain (d=12, Chinchilla) | ~15 min | 8×H100 | ~$7.00 |
| pico_cla_shared_ffn pretrain (d=12, Chinchilla) | ~15 min | 8×H100 | ~$7.00 |
| Eval: bpb + CORE (×4 models) | ~120 min | 4×H100 | ~$28.00 |
| **Total** | **~250 min** | | **~$90** |

*Pricing based on Modal H100 on-demand rate (~$3.50/GPU/hr × 8 = $28/hr node, $14/hr for 4×H100).*

**Credit efficiency decisions:**
- `--core-metric-every=-1` skips CORE during training — CORE is run once per model after training in `stage_eval`.
- Eval uses 4×H100 instead of 8×H100 — not compute-bound; 4 GPUs gives the same result for half the cost.
- 80 shards instead of 240: Chinchilla-optimal for ~85M params needs ~20 shards; 80 provides headroom without redundant data.
- Local smoke tests (d=4, 50 steps, MPS) validated all code paths before cloud runs.
- shared_ffn runs use d=12 (iso-depth) rather than d=34 (iso-parameter) to avoid ~$190/run Chinchilla cost at d=34; the parameter count difference is acknowledged as a limitation.

---

## Translation to Larger Runs

### Shared FFN at full scale

MobiLlama demonstrated shared FFN benefits at 0.5B–1B scale in an iso-parameter setup (freed MLP params reinvested in depth). Our picochat result used iso-depth (d=12) rather than iso-parameter, so the quality degradation partially reflects fewer parameters. A fair test at speedrun scale would use d=34 shared_ffn (~85M params, iso-parameter with d=12 baseline) trained to Chinchilla-optimal — this is the experiment that would directly test MobiLlama's hypothesis in the nanochat architecture.

### CLA at full scale

Our results show CLA causes a small val_bpb degradation (+2.1%) at d=12 picochat scale, with a larger CORE penalty (-49%). The val_bpb result is closer to Brandon et al.'s quality-neutrality finding than our initial d=8 pilot suggested.

Two factors may improve the picture at full speedrun scale (d=24–26):

1. **Value embeddings use alternating layers** (`has_ve` function): at d=24, only half the layers have value embeddings. A CLA implementation that restricts sharing to pairs where neither layer has a value embedding would avoid the Q/KV depth mismatch.

2. **The SSSL window pattern creates natural sharing boundaries**: at d=24 with SSSL, restricting CLA to pairs of same-window-type layers (SS pairs share, LL pairs share, no cross-boundary sharing) would test CLA in a more controlled setting.

These refinements were not included in our picochat implementation. A follow-up at d=24 with value-embedding-aware CLA pairing is the most promising path to recovering Brandon et al.'s quality-neutrality result in nanochat.
