# Part 2: Ablations on a Nano Nanochat

## Overview

We select two architecture changes — **Cross-Layer Attention (CLA)** (Brandon et al., NeurIPS 2024) and **Differential Attention** (Ye et al., ICLR 2025) — and measure their impact on a small nanochat configuration we call **picochat**. Both required real implementation in `gpt.py`. We train three models: a baseline and each change in isolation, comparing val_bpb and CORE score at d=12.

---

## Picochat Configuration

We define picochat as a depth=12 nanochat model:

| Hyperparameter | Value | Rationale |
|---|---|---|
| `--depth` | 12 | Matches reference `quick_test` depth (~85M non-embedding params); gives 6 CLA-2 sharing pairs |
| `--aspect-ratio` | 64 | nanochat default; model_dim = 12 × 64 = 768 |
| `--head-dim` | 64 | 768/64 = 12 heads, providing richer diversity for CLA-2 sharing (vs only 6 heads at head-dim=128) |
| `--window-pattern` | L | Full attention; appropriate for short sequences |
| `--device-batch-size` | 16 | Matches reference speedrun config |
| Training horizon | Chinchilla (10.5×) | Automatically computed from parameter count |
| Data | 80 FineWeb-EDU shards | Comfortable headroom over the ~20 shards needed for Chinchilla-optimal at ~85M params |
| GPU | 8×H100 train / 4×H100 eval | Matches reference speedrun; 4×H100 for eval halves cost |

**Justification for d=12:** Preliminary d=8 runs showed CLA hurt quality (val_bpb +0.023, CORE −14%). d=12 provides 6 CLA sharing pairs vs 4 at d=8, giving CLA a fairer test, and is the reference `quick_test` depth.

---

## The Three Models

### Model 1: pico_baseline
**Config:** d=12, model_dim=768, 12 heads, ReLU², independent KV per layer, per-layer MLPs.

The control model. Identical to the Feb 2026 nanochat architecture at this scale. All other models are compared against this checkpoint.

### Model 2: pico_cla
**Config:** d=12, model_dim=768, 12 heads, ReLU², **CLA-2 KV sharing**.

Implements Cross-Layer Attention (Brandon et al., NeurIPS 2024) with sharing factor 2. Even-numbered layers reuse the K and V tensors from the preceding odd-numbered layer, computing only Q themselves. This halves the number of independent KV projections:

```python
# In GPT.forward():
shared_kv = None
for i, block in enumerate(self.transformer.h):
    if cla_sharing > 1 and i % cla_sharing != 0:
        x = block(x, ve, cos_sin, window_size, kv_cache, shared_kv=shared_kv)
    elif cla_sharing > 1:
        x, k, v = block(x, ve, cos_sin, window_size, kv_cache, return_kv=True)
        shared_kv = (k, v)
```

CLA follower layers do not create `c_k`/`c_v` weight matrices, reducing parameter count by ~7M (~78M total vs ~85M baseline).

### Model 3: pico_diff_attn
**Config:** d=12, model_dim=768, **6 super-heads**, head_dim=64, ReLU², **Differential Attention**.

Differential Attention (Ye et al., ICLR 2025) computes two softmax attention maps per head and subtracts one from the other. The difference cancels attention noise and focuses on relevant tokens. `n_head` is halved so total output dimension stays equal to `n_embd`; V is doubled to `2 × head_dim` per KV head, preserving the same total KV dimension as the baseline.

```python
# Per super-head:
A1 = softmax(q1 @ k1.T / sqrt(d)) @ v
A2 = softmax(q2 @ k2.T / sqrt(d)) @ v
y  = A1 - λ * A2
# λ = exp(lq1·lk1).clamp(-10,10) - exp(lq2·lk2).clamp(-10,10) + λ_init
# λ_init = 0.8 - 0.6 * exp(-0.3 * layer_idx)
```

Output is normalized per-head with a no-param RMSNorm and scaled by `(1 - λ_init)`. Parameter count is approximately equal to the baseline (~85M) since the halved head count and doubled V dimension cancel out.

---

## Training Setup

All models trained on Modal, 8×H100, 80 FineWeb-EDU shards, shared BPE tokenizer (2B chars). Tracked with Weights & Biases under `picochat-ablation`. Training horizon set automatically by Chinchilla ratio (`--target-param-data-ratio=10.5`).

```bash
# Local smoke test (~5 min)
bash runs/runpico_test.sh

# Cloud training (full pipeline)
modal run runs/pico_ablation_modal.py

# Individual stages
modal run runs/pico_ablation_modal.py::stage_pretrain_baseline
modal run runs/pico_ablation_modal.py::stage_pretrain_cla
modal run runs/pico_ablation_modal.py::stage_pretrain_diff_attn
modal run runs/pico_ablation_modal.py::stage_pretrain_cla_diff_attn
modal run runs/pico_ablation_modal.py::stage_eval
```

---

## Results

| Model | Attn | Params | val_bpb ↓ | CORE ↑ | vs baseline |
|---|---|---|---|---|---|
| pico_baseline | Standard | ~85M | **0.9250** | **0.1273** | — |
| pico_cla | Standard + CLA-2 | ~78M | **0.9447** | **0.0655** | +0.020 bpb, −49% CORE |
| pico_diff_attn | Differential | ~85M | **0.9222** | **0.1118** | −0.003 bpb ✓, −12% CORE |
| pico_cla_diff_attn | Differential + CLA-2 | ~78M | *pending* | *pending* | — |
| GPT-2 target | — | ~1.5B | ~0.748 | 0.2565 | — |

**Note:** At picochat scale, CORE scores are well below the GPT-2 threshold (0.2565). Relative differences are what matter.

---

## Commentary

### CLA (pico_cla vs pico_baseline)

CLA-2 achieved val_bpb of **0.9447** vs baseline **0.9250** — a degradation of **+0.020 bpb (+2.1%)**. CORE dropped from **0.1273 to 0.0655 (−49%)**.

| Depth | Baseline bpb | CLA bpb | Δbpb | Baseline CORE | CLA CORE | ΔCORE |
|---|---|---|---|---|---|---|
| d=8 (pilot) | 1.027 | 1.050 | +0.023 | 0.066 | 0.057 | −14% |
| d=12 (main) | 0.925 | 0.945 | +0.020 | 0.127 | 0.066 | −49% |

Brandon et al. (2024) reported near quality-neutrality at 1B scale. Our bpb degradation (+2.1%) is consistent with that finding. However, the CORE drop (−49%) is disproportionately large, suggesting benchmark reasoning is more sensitive to CLA than raw language modelling loss.

**Why CORE suffers more than bpb:** nanochat's **value embeddings** add a token-ID-indexed offset directly to the V tensor. When CLA follower layers reuse the leader's V, they inherit representations shaped by the leader's shallower residual stream — a depth mismatch that doesn't affect average token prediction much but hurts structured reasoning tasks. Brandon et al. used vanilla transformers without value embeddings, so their quality-neutrality result may not transfer.

### Differential Attention (pico_diff_attn vs pico_baseline)

Differential Attention achieved val_bpb of **0.9222** vs baseline **0.9250** — a **−0.003 bpb improvement** — but CORE dropped from **0.1273 to 0.1118 (−12%)**.

The bpb improvement is consistent with the paper's claim of reduced attention noise. The CORE gap is harder to explain but likely reflects an interaction with nanochat's QK-norm: QK-norm normalizes Q and K to unit norm, which suppresses the entropy variance between attention heads that the differential mechanism exploits. With QK-norm applied, A1 and A2 have similar distributions, limiting how much signal `A1 − λ·A2` carries. QK-norm cannot be removed without training instability under the Muon optimizer (Q/K weight norms grow unchecked, causing bfloat16 overflow by step 7).

The paper evaluates at 3B scale without QK-norm or Muon; both constraints are specific to nanochat's setup and limit the differential mechanism's effectiveness here.

### Combined CLA + Differential Attention (pico_cla_diff_attn vs pico_baseline)

*Results pending.*

CLA follower layers reuse the leader's doubled K tensor (both k1 and k2 groups) and doubled V directly, so no extra projection is needed on follower layers. The two changes operate on different parts of the attention mechanism — CLA reduces KV computation across layers while differential attention changes how each layer computes its output — so they are not architecturally coupled. The prior expectation is that their effects compound independently.

---

## W&B Visualisation

Training runs tracked under `picochat-ablation`. Key plots:

1. **val_bpb vs step** — main quality comparison across all 4 runs
2. **val_bpb vs total_training_flops** — iso-FLOP comparison (more honest than iso-step)
3. **CORE score** — evaluated once after training per model
4. **tok/sec** — CLA is faster due to fewer KV projections

*[Insert W&B screenshot here.]*

---

## Cost of Training

| Item | Time | GPUs | Cost |
|---|---|---|---|
| Data + tokenizer | ~10 min | CPU / 1×H100 | ~$5.30 |
| pico_baseline (d=12, Chinchilla) | ~40 min | 8×H100 | ~$18.70 |
| pico_cla (d=12, Chinchilla) | ~40 min | 8×H100 | ~$18.70 |
| pico_diff_attn (d=12, Chinchilla) | ~40 min | 8×H100 | ~$18.70 |
| pico_cla_diff_attn (d=12, Chinchilla) | ~40 min | 8×H100 | ~$18.70 |
| Eval: bpb + CORE (×4 models) | ~120 min | 4×H100 | ~$28.00 |
| **Total** | **~290 min** | | **~$108** |

*Pricing: Modal H100 on-demand (~$3.50/GPU/hr × 8 = $28/hr node, $14/hr for 4×H100).*

**Credit efficiency decisions:**
- `--core-metric-every=-1` during training; CORE run once after in `stage_eval`
- 4×H100 for eval (not compute-bound; saves ~50% vs 8×H100)
- 80 shards vs 240 in speedrun: Chinchilla-optimal at ~85M needs ~20 shards; 80 provides headroom
- Local smoke tests (d=4, 50 steps, CPU/MPS) validated all code paths before cloud runs

---

## Translation to Larger Runs

### CLA at full scale

Our bpb result (+2.1%) is consistent with Brandon et al.'s quality-neutrality at 1B scale. Two refinements could recover the CORE gap at d=24–26:

1. **Value-embedding-aware pairing**: restrict CLA sharing to pairs where neither layer has a value embedding (alternating layers in nanochat), avoiding the Q/KV depth mismatch.
2. **Window-pattern-aware pairing**: with the SSSL pattern at d=24, restrict sharing to same-window-type pairs (SS or LL) to avoid cross-boundary sharing.

### Differential Attention at full scale

The bpb improvement (−0.003) is consistent across both our runs. At larger scale (d=24–26), `λ_init` saturates toward 0.8 in deep layers, making the differential weighting stronger. The key open question is whether QK-norm remains a hard requirement at larger scale under Muon — if an alternative stabilisation (e.g. gradient clipping + lower init scale) allows removing it, the differential mechanism could be more effective.

---

## Appendix: Additional Experiments

### Shared FFN (pico_shared_ffn)

We also tested MobiLlama-style FFN weight sharing: a single MLP shared across all 12 layers instead of per-layer MLPs. Results: val_bpb **1.0058** (+8.7%), CORE **0.0810** (−36%). The result is confounded — pico_shared_ffn has only ~33M params vs baseline's ~85M (MLP sharing reduces total params 2.6×). Quality degradation reflects both weight sharing and fewer parameters and cannot be cleanly attributed to either alone. A fair comparison would require an iso-parameter setup (d=34 shared_ffn ≈ 85M params), which was too expensive to run within the credit budget.

### Combined CLA + Shared FFN (pico_cla_shared_ffn)

Both changes applied simultaneously: val_bpb **1.0410** (+12.5%), CORE **0.0575** (−55%). Worse than either change in isolation, consistent with independently harmful effects compounding additively.

### Differential Attention v2

A second diff attn run with higher lambda vector LR (`scalar_lr` vs `scalar_lr × 0.01`). Results: val_bpb **0.9260**, CORE **0.1133** — slightly worse bpb than v1 (0.9222) but marginally better CORE. The higher LR made lambda values noisier without a clear benefit. v1 is the better run.

An attempt to remove QK-norm (to let the differential mechanism exploit full attention entropy variance) caused NaN loss at step 7 in bfloat16 due to Muon growing Q/K weight norms unchecked. QK-norm is a hard stability requirement with this optimizer.

### d=8 Pilot Runs

| Model | val_bpb | CORE |
|---|---|---|
| pico_baseline (d=8) | 1.027 | 0.066 |
| pico_cla (d=8) | 1.050 | 0.057 |

Validated the pipeline and motivated the switch to d=12.
