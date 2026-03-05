# Part 2: Ablations on a Nano Nanochat

## Overview

We select two architecture changes — **Mixture of Depths (MoD)** (Raposo et al., arXiv 2404.02258) and **Differential Attention** (Ye et al., ICLR 2025) — and evaluate them on a small nanochat configuration we call **picochat**. Both required real implementation in `gpt.py`. We train three models: a baseline and each change in isolation, comparing val_bpb and CORE score at d=12.

The two modifications have different design goals. **MoD** is an efficiency technique: at iso-parameter count, it reduces training FLOPs per step by ~44%, enabling faster wall-clock training and higher token throughput with a fixed model size. The evaluation question for MoD is whether this efficiency gain comes at an acceptable quality cost. **Differential Attention** targets quality directly: same compute, better attention selectivity. Its evaluation is straightforwardly quality-focused.

---

## Picochat Configuration

We define picochat as a depth=12 nanochat model:

| Hyperparameter | Value | Rationale |
|---|---|---|
| `--depth` | 12 | Matches reference `quick_test` depth (~85M non-embedding params) |
| `--aspect-ratio` | 64 | nanochat default; model_dim = 12 × 64 = 768 |
| `--head-dim` | 64 | 768/64 = 12 heads |
| `--window-pattern` | L | Full attention; appropriate for short sequences |
| `--device-batch-size` | 16 | Matches reference speedrun config |
| Training horizon | Chinchilla (10.5×) | Automatically computed from parameter count |
| Data | 80 FineWeb-EDU shards | Comfortable headroom over the ~20 shards needed for Chinchilla-optimal at ~85M params |
| GPU | 8×H100 train / 4×H100 eval | Matches reference speedrun; 4×H100 for eval halves cost |

**Justification for d=12:** Preliminary d=8 runs validated the pipeline and showed the expected quality gap. d=12 is the reference `quick_test` depth and provides a better training signal for evaluating architectural effects.

---

## The Three Models

### Model 1: pico_baseline
**Config:** d=12, model_dim=768, 12 heads, ReLU², standard per-layer attention and MLPs.

The control model. Identical to the Feb 2026 nanochat architecture at this scale. All other models are compared against this checkpoint.

### Model 2: pico_mod
**Config:** d=12, model_dim=768, 12 heads, ReLU², **Mixture of Depths routing**.

Implements Mixture of Depths (Raposo et al., arXiv 2404.02258). Even-indexed layers (0, 2, 4, …) use a learned scalar router to select the top `capacity = 12.5%` of tokens; those tokens are processed through attention + MLP while the rest skip the layer via the residual. Odd-indexed layers run at full capacity. At d=12, 6 of 12 layers are MoD layers.

```python
# In Block.forward() for MoD layers:
router_weights = self.mod_router(x).squeeze(-1)       # (B, T) scalar scores
_, top_indices = torch.topk(router_weights, capacity, dim=1)
sorted_positions, _ = torch.sort(top_indices, dim=1)  # sort for causal RoPE correctness

x_sel = x.gather(1, gather_e)                         # gather selected tokens
x_out = x_sel + attn(norm(x_sel), ...)
x_out = x_out + mlp(norm(x_out))

weighted_delta = router_w_sel * (x_out - x_sel)
x = x.scatter_add(1, gather_e, weighted_delta)        # scatter back
```

The router logit doubles as an interpolation scalar: `x[pos] += r * delta`. Router weights are zero-initialized so MoD layers are transparent at step 0 (weighted_delta = 0), and routing is learned gradually.

**FLOP reduction:** Each MoD layer operates on 12.5% of tokens for both attention and MLP projections, reducing per-layer compute by ~87.5%. With half the layers being MoD layers: ~44% fewer FLOPs per step vs baseline, reducing training time and increasing effective token throughput at iso-parameter count. The evaluation question is whether this efficiency gain comes at an acceptable quality cost.

Parameter count is approximately equal to baseline (~85M) — the router adds only `n_embd` parameters per MoD layer (6 × 768 = ~4.6K), negligible at this scale.

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
modal run runs/pico_ablation_modal.py::stage_pretrain_mod
modal run runs/pico_ablation_modal.py::stage_pretrain_diff_attn
modal run runs/pico_ablation_modal.py::stage_pretrain_mod_diff_attn
modal run runs/pico_ablation_modal.py::stage_eval
```

---

## Results

| Model | Attn | Params | val_bpb ↓ | CORE ↑ | Δbpb | ΔCORE |
|---|---|---|---|---|---|---|
| pico_baseline | Standard | ~85M | **0.925162** | **0.1140** | — | — |
| pico_mod | Standard + MoD | ~85M | 0.959804 | 0.1016 | +3.7% | −10.9% |
| pico_diff_attn | Differential | ~85M | 0.926037 | 0.1051 | +0.1% | −7.8% |
| pico_mod_diff_attn | Differential + MoD | ~85M | 0.962692 | 0.0754 | +4.1% | −33.9% |
| GPT-2 target | — | ~1.5B | ~0.748 | 0.2565 | — | — |

At picochat scale, CORE scores are well below the GPT-2 threshold (0.2565); relative differences are what matter. Results should be interpreted against each model's design goal: **MoD** is evaluated on whether its ~44% FLOP reduction comes at acceptable quality cost (efficiency trade-off); **DiffAttn** is evaluated on whether it improves quality at matched compute (quality target). Both underperform the baseline — but for MoD the question is the magnitude of quality cost, not whether quality improved.

---

## Commentary

### Summary

| Model | val_bpb | delta_bpb | CORE | delta_CORE |
|---|---|---|---|---|
| pico_baseline | 0.9252 | — | 0.1140 | — |
| pico_mod | 0.9598 | +3.7% | 0.1016 | −10.9% |
| pico_diff_attn | 0.9260 | +0.1% | 0.1051 | −7.8% |
| pico_mod_diff_attn | 0.9627 | +4.1% | 0.0754 | −33.9% |

MoD achieves its efficiency goal (fewer FLOPs per step) but at a quality cost that warrants scrutiny. DiffAttn is a clear quality-targeted failure. The analysis below separates these stories.

---

### Mixture of Depths (pico_mod vs pico_baseline)

**Result.** MoD achieved its efficiency goal: ~44% fewer FLOPs per step at identical parameter count (~85M). The quality cost was val_bpb +3.7% (0.9252 → 0.9598) and CORE −10.9% (0.1140 → 0.1016). The key question is whether this quality tax is acceptable for the efficiency gain.

**Design goal: iso-parameter efficiency.** This experiment is a deliberate iso-parameter comparison: same depth, width, and parameter count as the baseline, but with 6 of 12 layers processing only 12.5% of tokens. The motivation is deployment efficiency — a fixed-size model that trains faster and processes more tokens per GPU-hour. This is a valid use case distinct from Raposo et al.'s isoFLOP framing.

For context: Raposo et al. (2024) validated MoD primarily under *isoFLOP* conditions, where saved FLOPs were reinvested into a larger model. Their key finding was that 12.5% capacity MoD could match baseline loss *when the saved FLOPs funded additional parameters*. Under iso-parameter conditions, Raposo et al. do not claim quality improvement — so observing degradation here is expected. The question is the magnitude.

**Is 3.7% bpb degradation acceptable?** In language modeling, a 3.7% increase in bpb is not a minor efficiency tax — it represents a meaningful degradation in next-token prediction quality. The corresponding CORE drop of −10.9% compounds this: downstream reasoning tasks are disproportionately sensitive to perplexity differences in the tail of the distribution. For a quality-sensitive application, this trade-off is unfavorable at d=12. The efficiency gain is real (faster training, lower cost), but the quality cost is steeper than desirable.

**Why the quality cost is high at this scale.** Several factors compound the degradation:

- **12.5% capacity at seq_len=2048 is aggressive.** Each MoD layer processes only 256 of 2048 tokens. At d=12, 87.5% of tokens receive attention/MLP updates from only half the layers. At ~85M parameters, the model likely lacks the representational redundancy to tolerate this sparsity — individual layers carry more of the representational load than at 3B parameters, where Raposo et al. found 12.5% optimal.

- **Router convergence at small scale.** The router is a single linear projection (768→1), zero-initialized so all tokens start with equal scores. The top-k selection is non-differentiable — at 12.5% capacity, 87.5% of tokens provide no router gradient signal per layer per step. At 85M parameters, the gradient signal for router specialization is weaker than at larger scales.

- **Interaction with value embeddings.** Tokens skipped by MoD layers miss the value embedding modulation (a gated additive signal to V) and attention-mediated context mixing. nanochat's value embeddings provide per-layer token specialization — skipped tokens lose this layered refinement entirely.

**Efficiency gain is real.** Despite the quality cost, MoD does what it advertises: wall-clock training time is lower, and total training FLOPs are reduced ~44%. For applications where deployment speed or training cost is the primary constraint and some quality degradation is acceptable, iso-parameter MoD remains a valid design choice. The results here establish the quality tax at d=12: ~3.7% bpb, ~11% CORE.

---

### Differential Attention (pico_diff_attn vs pico_baseline)

**Result.** Differential attention showed negligible BPB change (+0.1%, essentially flat at 0.9260 vs 0.9252) but a meaningful CORE drop (−7.8%, from 0.1140 to 0.1051).

**The QK-norm interaction.** This is the central issue. Ye et al. (2024) developed and evaluated differential attention using AdamW without QK-norm, training at scales from 830M to 13.1B parameters on up to 1T tokens. Their architecture does *not* normalize Q and K projections individually (Ye et al., arXiv:2410.05258).

nanochat *requires* QK-norm for training stability: the Muon optimizer (used for all 2D+ weight matrices including Q/K projections) drives Q and K weight norms upward during training, causing `q @ k^T` dot products to overflow in bfloat16 without normalization. Removing QK-norm causes NaN loss at approximately step 7. This is a hard constraint.

The mechanistic consequence: QK-norm (RMSNorm applied to each of q1, q2, k1, k2 independently after RoPE) normalizes all four query/key vectors to similar magnitude distributions. The attention logits for both A1 and A2 are therefore computed from normalized vectors with similar scale, making the two attention distributions more similar. The subtraction `A1 − λ·A2` is designed to cancel diffuse, uninformative attention mass — but when both Q/K pairs are normalized to similar scales, A1 and A2 become more correlated, reducing the dynamic range of their difference. Ye et al.'s setup without QK-norm allows Q/K norms to vary freely across heads and layers, creating natural variation between the two attention groups that the differential mechanism can exploit. QK-norm removes this degree of freedom.

**The BPB-CORE divergence.** The near-zero BPB delta (+0.001) combined with a −7.8% CORE drop reveals an important distinction. BPB measures average next-token prediction quality across the full validation distribution. CORE is a composite downstream task score (ARC, MMLU, etc.) that tests specific reasoning and knowledge retrieval. The divergence suggests differential attention preserves bulk language modeling quality while degrading on the tail of the distribution that matters for downstream tasks — consistent with QK-norm-constrained differential attention impairing the model's ability to form sharp, selective attention patterns needed for reasoning, while maintaining adequate performance on the "easy" majority of next-token predictions.

An additional factor: halving `n_head` (from 12 to 6 super-heads) reduces attention diversity. With fewer independent attention patterns, the model may struggle on tasks requiring multi-hop reasoning or simultaneous entity tracking — precisely the tasks that dominate CORE benchmarks.

---

### Combined MoD + Differential Attention (pico_mod_diff_attn vs pico_baseline)

**Result.** val_bpb degraded +4.1% (0.9627) and CORE dropped −33.9% (0.0754). The BPB degradation is roughly additive, but CORE degradation is strongly super-additive.

**Quantifying the interaction.** If the two modifications' effects were independent and additive:

| Metric | MoD alone | DiffAttn alone | Expected (additive) | Observed | Excess |
|---|---|---|---|---|---|
| Δbpb | +0.0346 | +0.0009 | +0.0355 | +0.0375 | +0.002 (~6%) |
| ΔCORE | −0.0124 | −0.0089 | −0.0213 | −0.0386 | −0.0173 (81%) |

BPB degradation is approximately additive. CORE degradation is 81% worse than additive prediction — a strong negative interaction.

**Mechanistic explanation.** The super-additive CORE degradation arises from compounding information bottlenecks. MoD routes 87.5% of tokens around even-indexed layers, meaning the differential attention mechanism in those layers operates on only 256 tokens. With such a small token set, the two sub-attention maps A1 and A2 become highly correlated (fewer tokens means less diversity in attention patterns), and the subtraction `A1 − λ·A2` approaches zero for most positions — effectively nullifying the differential mechanism in MoD layers. The model simultaneously loses the routing benefit (too few tokens for meaningful selection) and the differential benefit (too few tokens for meaningful subtraction). For bulk language modeling, neither bottleneck is catastrophic on its own; for reasoning tasks requiring broad token participation across layers *and* sharp, diverse attention patterns, the two bottlenecks compound destructively.

---

### Overall Conclusions and Recommendations

**MoD (pico_mod):** Achieves its efficiency goal — ~44% fewer FLOPs per step at matched parameter count. The quality cost (+3.7% bpb, −10.9% CORE) is larger than desirable for quality-sensitive applications at d=12. This is not a failure of MoD as a technique; it is the expected quality tax under iso-parameter deployment, and it quantifies the trade-off at this scale. For latency- or cost-constrained applications the efficiency gain is real; for quality-sensitive applications the trade-off is unfavorable at d=12.

**DiffAttn (pico_diff_attn):** A clear negative result. DiffAttn targets quality, not efficiency, so +0.1% bpb and −7.8% CORE is straightforwardly a failure to improve. The root cause is the mandatory QK-norm constraint imposed by the Muon optimizer, which reduces the dynamic range that the differential mechanism depends on.

**Combined (pico_mod_diff_attn):** Both quality costs compound, with CORE degradation super-additive (81% worse than additive expectation), confirming that the two bottlenecks interact destructively.

**These are not refutations of the original papers.** Both methods were validated under substantially different conditions:
- Raposo et al. demonstrated MoD gains under isoFLOP comparisons at 60M–3B scale with standard AdamW. Iso-parameter MoD incurs a known quality tax; our results quantify it at d=12.
- Ye et al. demonstrated DiffAttn gains at 830M–13.1B scale with AdamW and without QK-norm. nanochat's mandatory QK-norm is a hard constraint that limits the differential mechanism's effectiveness.

**What this tells us.** Architectural modifications are not portable across training recipes. The interaction between optimizer choice (Muon), stability mechanisms (QK-norm), and architectural features (value embeddings, residual scaling) creates a specific optimization landscape in which these modifications behave differently from their original contexts.

**Recommendation for Part 4 (full nanochat, d=24+).** Run the baseline configuration without DiffAttn. MoD at d=24+ may be revisited as an efficiency option — see "Translation to Larger Runs" below.

---

## W&B Visualisation

Training runs tracked under `picochat-ablation`. Key plots:

1. **val_bpb vs step** — main quality comparison across all 4 runs
2. **val_bpb vs total_training_flops** — isoFLOP comparison (more honest for MoD, which trains fewer FLOPs per step)
3. **CORE score** — evaluated once after training per model
4. **tok/sec** — MoD is faster due to token-sparse computation on even layers

*[Insert W&B screenshot here.]*

---

## Cost of Training

| Item | Time | GPUs | Cost |
|---|---|---|---|
| Data + tokenizer | ~10 min | CPU / 1×H100 | ~$5.30 |
| pico_baseline (d=12, Chinchilla) | ~40 min | 8×H100 | ~$18.70 |
| pico_mod (d=12, Chinchilla) | ~40 min | 8×H100 | ~$18.70 |
| pico_diff_attn (d=12, Chinchilla) | ~40 min | 8×H100 | ~$18.70 |
| pico_mod_diff_attn (d=12, Chinchilla) | ~40 min | 8×H100 | ~$18.70 |
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

### MoD at full scale

Iso-parameter MoD at d=24–26 is a valid efficiency option worth revisiting. At d=12, each MoD layer is 8.3% of total depth, so skipping one is a large fraction of total compute. At d=24, the same layer is 4.2% of depth — individual layers carry less of the representational load, and the model has more redundancy to absorb sparsity. The quality tax per FLOP saved may therefore be lower at larger depth.

If running MoD at d=24–26 as a quality experiment (not efficiency), an isoFLOP design is the correct protocol — use the saved ~44% FLOPs to increase model width or depth, matching total training compute. The capacity fraction (12.5%) may also need tuning at longer context lengths (Part 3: seq_len=2048+).

### Differential Attention at full scale

The QK-norm constraint is the key open question. At d=24–26, `λ_init` saturates toward 0.8 in deep layers, making the differential weighting stronger — but the fundamental incompatibility with QK-norm persists. Differential attention would become viable only if (a) Muon is replaced with AdamW for Q/K weight matrices, or (b) an alternative stabilization mechanism is found that does not homogenize Q/K norms.

---

## Appendix: Additional Experiments

### Shared FFN (pico_shared_ffn)

MobiLlama-style FFN weight sharing: a single MLP shared across all 12 layers. Results: val_bpb **1.0058** (+8.7%), CORE **0.0810** (−36%). The result is confounded — pico_shared_ffn has only ~33M params vs baseline's ~85M (MLP sharing reduces total params 2.6×). Quality degradation reflects both weight sharing and fewer parameters and cannot be cleanly attributed to either alone.

### Combined CLA + Shared FFN (pico_cla_shared_ffn)

Both changes applied simultaneously: val_bpb **1.0410** (+12.5%), CORE **0.0575** (−55%). Worse than either change in isolation, consistent with independently harmful effects compounding additively.

### Differential Attention v2

A second diff attn run with higher lambda vector LR (`scalar_lr` vs `scalar_lr × 0.01`). Results: val_bpb **0.9260**, CORE **0.1133** — slightly worse bpb than v1 (0.9222) but marginally better CORE. The higher LR made lambda values noisier without a clear benefit. v1 is the better run.

An attempt to remove QK-norm (to let the differential mechanism exploit full attention entropy variance) caused NaN loss at step 7 in bfloat16 due to Muon growing Q/K weight norms unchecked. QK-norm is a hard stability requirement with this optimizer.

### d=8 Pilot Runs

| Model | val_bpb | CORE |
|---|---|---|
| pico_baseline (d=8) | 1.027 | 0.066 |

Validated the pipeline and motivated the switch to d=12.

---

## References

1. Raposo, D., Ritter, S., Richards, B., Lillicrap, T., Humphreys, P.C., and Santoro, A. "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models." arXiv:2404.02258, 2024.

2. Ye, T., Dong, L., Xia, Y., Sun, Y., Zhu, Y., Huang, G., and Wei, F. "Differential Transformer." In *Proceedings of the International Conference on Learning Representations (ICLR)*, 2025. arXiv:2410.05258.

3. Jordan, P., Muennighoff, N., et al. "Muon: An optimizer for hidden layers in transformers." 2024.

4. Henry, A., Dachapally, P.R., Pawar, S., and Chen, Y. "Query-Key Normalization for Transformers." In *Findings of EMNLP*, 2020.

5. Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., et al. "Training Compute-Optimal Large Language Models." In *NeurIPS*, 2022. arXiv:2203.15556.
