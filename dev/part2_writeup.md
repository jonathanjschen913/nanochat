# Part 2: Ablations on a Nano Nanochat

## Overview

We select two of the three architecture changes identified in Part 1 — **SwiGLU activation** and **Cross-Layer Attention (CLA)** — and measure their impact on a small nanochat configuration we call **picochat**. Both changes are absent from the Feb 2026 nanochat codebase and require real implementation in `gpt.py`. We train four models: a baseline, each change in isolation, and both changes combined. The combined run tests whether the two changes are independent, synergistic, or interfering — a key question for recommending them together in the full speedrun.

---

## Picochat Configuration and Justification

We define picochat as a depth=12 nanochat model with the following hyperparameters:

| Hyperparameter | Value | Rationale |
|---|---|---|
| `--depth` | 12 | Matches reference `quick_test` depth (~125M params); gives 6 CLA-2 sharing pairs. Preliminary d=8 runs showed CLA hurt quality (see below), motivating the switch to d=12 |
| `--aspect-ratio` | 64 | nanochat default; model_dim = 12 × 64 = 768 |
| `--head-dim` | 64 | Speedrun default is 128, but at model_dim=768 that gives only 6 heads (768/128=6). head-dim=64 gives 12 heads, providing richer head diversity for CLA-2 sharing |
| `--max-seq-len` | 512 | Sufficient for document-level context at this scale |
| `--window-pattern` | L | Full attention; appropriate for short sequences |
| `--device-batch-size` | 16 | Matches reference speedrun `DEVICE_BATCH_SIZE=16` |
| Training horizon | Chinchilla (10.5×) | Automatically computed from parameter count; mirrors reference speedrun approach |
| Data | 80 FineWeb-EDU shards | d=12 has ~125M params; Chinchilla-optimal at 10.5× ≈ 1.3B tokens ≈ 26 shards needed. 80 shards provides comfortable headroom (~1/3 of the 240-shard speedrun) |
| GPU | 8×H100 | Matches reference speedrun `GPU_PRETRAIN`; eval runs on 4×H100 to halve eval cost |

**Justification for d=12:** We initially ran preliminary ablations at d=8 to validate the pipeline cheaply before committing to full runs. The d=8 results (Table 1 below) showed CLA *hurting* quality — val_bpb increased by 0.022 and CORE dropped from 0.066 to 0.057. The likely cause is that at d=8, adjacent layers have insufficient KV representation correlation for safe sharing — the redundancy that CLA exploits is a property of deeper models. d=12 is the reference `quick_test` depth, well-studied in the nanochat LOG.md, and provides 6 CLA sharing pairs (vs 4 at d=8), giving CLA a better chance to show its quality-neutral behaviour as reported by Brandon et al. at 1B scale.

**Why not larger?** d=16+ would give stronger statistical signal but each run costs ~$10+ on 8×H100, making the full ablation suite prohibitively expensive. d=12 is the sweet spot between cost and meaningful depth.

### Preliminary results at d=8 (pilot runs)

These runs validated the pipeline and motivated the switch to d=12:

| Model | val_bpb | CORE | Interpretation |
|---|---|---|---|
| pico_baseline (d=8) | 1.027 | 0.0660 | Control |
| pico_cla (d=8) | 1.050 | 0.0574 | CLA hurts at d=8 |

The quality degradation at d=8 is consistent with CLA's theoretical motivation: KV sharing only works when adjacent layers compute redundant representations, which requires sufficient model depth.

---

## The Four Models

### Model 1: pico_baseline
**Config:** d=12, model_dim=768, 12 heads, ReLU² activation, independent KV per layer.

The control model. Matches the picochat configuration above with no modifications — identical to the Feb 2026 nanochat architecture at this scale. All other models are compared against this checkpoint.

### Model 2: pico_swiglu
**Config:** d=12, model_dim=768, 12 heads, **SwiGLU activation**, independent KV per layer.

Replaces the MLP's ReLU² activation with SwiGLU (Shazeer, 2020). The change is isolated to the `MLP` class in `gpt.py`:

```python
# Before (ReLU²)
x = self.c_fc(x)        # up-project to 4 * n_embd
x = F.relu(x).square()
x = self.c_proj(x)      # down-project

# After (SwiGLU)
x, gate = self.c_gate(x).chunk(2, dim=-1)  # up-project to 8/3 * n_embd, split
x = x * F.silu(gate)                        # gated activation
x = self.c_proj(x)                          # down-project
```

The up-projection width changes from `4 × n_embd` to `8/3 × n_embd` (rounded to nearest multiple of 64) so that parameter count remains approximately equal to the baseline. This is the standard parameter-matching convention used in LLaMA and PaLM.

### Model 3: pico_cla
**Config:** d=12, model_dim=768, 12 heads, ReLU² activation, **CLA-2 KV sharing**.

Implements Cross-Layer Attention with a sharing factor of 2 (Brandon et al., NeurIPS 2024). Even-numbered layers reuse the K and V tensors computed by the preceding odd-numbered layer, computing only Q themselves. This halves the number of independent KV projections in the model:

```python
# In GPT.forward():
for i, block in enumerate(self.transformer.h):
    if cla_sharing > 1 and i % cla_sharing != 0:
        # CLA: reuse K and V cached by the previous group-leader layer
        prev_block = self.transformer.h[i - 1]
        shared_kv = (prev_block.attn._cla_k, prev_block.attn._cla_v)
        x = block(x, ve, cos_sin, window_size, kv_cache, shared_kv=shared_kv)
    else:
        # Normal layer: computes fresh K/V, stores in block.attn._cla_k/v
        x = block(x, ve, cos_sin, window_size, kv_cache)
```

All other hyperparameters are identical to the baseline. The change is isolated to `gpt.py`.

### Model 4: pico_swiglu_cla
**Config:** d=12, model_dim=768, 12 heads, **SwiGLU activation + CLA-2 KV sharing**.

Both changes applied simultaneously. The expected val_bpb allows us to determine whether the changes are:
- **Independent**: combined Δbpb ≈ swiglu Δbpb + cla Δbpb
- **Synergistic**: combined Δbpb > sum of individual deltas
- **Interfering**: combined Δbpb < sum of individual deltas

CLA reuses V projections across layers; SwiGLU changes how the MLP transforms features before they become the next layer's input. There is no obvious direct interaction between them, so independence is the prior expectation — but empirical confirmation is valuable.

---

## Training Setup

All models were trained on Modal using 8×H100 GPUs (matching the reference speedrun configuration) on 40 FineWeb-EDU shards with a shared BPE tokenizer trained on 2B characters (matching the reference speedrun.sh tokenizer exactly). Training was tracked using Weights & Biases under the project `picochat-ablation`. The training horizon was set automatically by the Chinchilla ratio (`--target-param-data-ratio=10.5`), the same mechanism used in the reference speedrun — no manual `--num-iterations` was specified.

**Run commands:**
```bash
# Local smoke test (verify code runs, ~3-5 min)
bash runs/runpico_test.sh

# Cloud training (full ablation, ~$34)
modal run runs/pico_ablation_modal.py

# Individual stages
modal run runs/pico_ablation_modal.py::stage_pretrain_baseline
modal run runs/pico_ablation_modal.py::stage_pretrain_swiglu
modal run runs/pico_ablation_modal.py::stage_pretrain_cla
modal run runs/pico_ablation_modal.py::stage_pretrain_swiglu_cla
modal run runs/pico_ablation_modal.py::stage_eval
```

---

## Results

### Training metrics

| Model | Activation | KV sharing | Params (non-emb) | val_bpb ↓ | CORE ↑ | tok/sec ↑ |
|---|---|---|---|---|---|---|
| pico_baseline | ReLU² | None (MHA) | ~125M | **[TBD]** | **[TBD]** | [TBD] |
| pico_cla | ReLU² | CLA-2 | ~118M | **[TBD]** | **[TBD]** | [TBD] |

*[Fill in from Modal stage_eval output and W&B dashboard after cloud run completes.]*

**Note on CORE scores:** At picochat scale (d=8, ~40M params), CORE scores will be well below the GPT-2 threshold of 0.256525 — expect values in the 0.10–0.15 range. The absolute score is not the goal; what matters is the **relative difference** between models. A meaningful CORE improvement alongside val_bpb improvement would strongly validate the architectural change.

### Preliminary results at d=8 (see Picochat Configuration section above)

| Model | val_bpb | CORE |
|---|---|---|
| pico_baseline (d=8) | 1.027 | 0.066 |
| pico_cla (d=8) | 1.050 | 0.057 |

CLA degraded quality at d=8, motivating the switch to d=12 for the main ablation.

---

## Commentary on Results

### SwiGLU (pico_swiglu vs pico_baseline)

*[To be completed after cloud run. Template:]*

The SwiGLU variant achieved a val_bpb of [X.XXX] vs the baseline's [X.XXX], a [improvement/regression] of [delta] bpb. This [supports/contradicts] Shazeer (2020)'s finding that GEGLU/SwiGLU consistently outperform ReLU by ~0.04 log-perplexity on C4.

**Nanochat-specific observations:**
- nanochat's current ReLU² is an unusual choice — squaring the ReLU output creates a smoother activation with stronger sparsity than plain ReLU. SwiGLU's multiplicative gating provides a fundamentally different inductive bias: rather than hard-zeroing negative activations, it softly gates the entire feature map. Whether this is better at picochat scale is an open question.
- The parameter count is matched by adjusting the up-projection from `4 × n_embd` to `8/3 × n_embd`. At very small model dims (512), rounding to the nearest multiple of 64 introduces a small parameter discrepancy (~2%) which is noted but not corrected, consistent with standard practice.
- SwiGLU's improvement in larger models is attributed partly to better gradient flow through the gating mechanism. nanochat's residual scalars (`resid_λ`, `x0_λ`) already address gradient flow at the block level — SwiGLU addresses it within the MLP. The two mechanisms are complementary.

### CLA (pico_cla vs pico_baseline)

*[To be completed after cloud run. Template:]*

The CLA-2 variant achieved a val_bpb of [X.XXX] vs the baseline's [X.XXX]. Brandon et al. (2024) reported roughly equal perplexity to a full-KV baseline at 1B scale with CLA-2, so [near-neutral/degraded] quality at picochat scale would be [consistent with/worse than] expectations.

The parameter reduction from CLA is meaningful: halving the KV projections removes ~[X]M parameters, which could be reinvested in depth or width in a future experiment. At picochat scale the KV cache memory savings are modest, but the architectural pattern is most valuable at larger scales and longer sequences (see Translation to Larger Runs).

**Nanochat-specific observations:**
- nanochat's value embeddings interact with CLA in an interesting way: even layers borrow the V projection from the preceding odd layer, but the value embedding (token-ID indexed) is still added independently at each layer. This means CLA layers are not fully identical in their value computation — they share the projection but retain per-layer value specialization through the embedding. This is a nanochat-specific advantage over vanilla transformer CLA implementations.
- The SSSL sliding window pattern means that in 3 of every 4 layers, attention only covers half the context. CLA sharing is applied uniformly across all layer types in this experiment; a future refinement would restrict sharing to pairs of same-window-type layers.

---

## W&B Visualisation

All four training runs are tracked in W&B under the project `picochat-ablation`. The primary plots to examine:

1. **val_bpb vs step** — main quality comparison; all 3 runs on the same axis
2. **val_bpb vs total_training_flops** — iso-FLOP comparison (more honest than iso-step since CLA has fewer FLOPs per step)
3. **CORE score** — evaluated once after training for each model; plotted as a bar chart
4. **tok/sec** — throughput; CLA should be faster due to fewer KV projections
5. **train/mfu** — model FLOP utilisation

*[Insert W&B screenshot here after cloud run.]*

---

## Cost of Training

| Item | Time | GPUs | Cost |
|---|---|---|---|
| d=8 pilot runs (baseline + cla) | ~20 min | 8×H100 | ~$10.40 |
| Data + tokenizer (d=12) | ~10 min | CPU / 1×H100 | ~$5.30 |
| pico_baseline pretrain (d=12) | ~5 min | 8×H100 | ~$2.60 |
| pico_cla pretrain (d=12) | ~5 min | 8×H100 | ~$2.60 |
| Eval: bpb + CORE + sample (×2) | ~60 min | 4×H100 | ~$4.70 |
| **Total** | **~100 min** | | **~$25.60** |

*Pricing based on Modal H100 on-demand rate (~$3.50/GPU/hr × 8 = $28/hr node, $14/hr for 4×H100).*

**Credit efficiency decisions:**
- `--core-metric-every=-1` skips CORE during training — CORE takes ~30 min per evaluation and is not useful mid-training at picochat scale. Instead, CORE is run once per model after training completes in `stage_eval`, giving a final comparable score for each model.
- Eval uses 4×H100 instead of 8×H100 — `base_eval` is not as compute-bound as training; 4 GPUs gives the same result for half the cost.
- The combined run (`pico_swiglu_cla`) adds only ~$5 to the total budget while providing the interaction analysis needed to recommend the changes together in the full speedrun.
- 40 shards instead of 240: the Chinchilla-optimal token budget for ~40M parameters is ~420M tokens; 40 shards (~2.5B characters) exceeds this comfortably without wasting credits on redundant data.
- Sequential pretrain stages share a single Modal Volume, avoiding race conditions and redundant tokenizer/data downloads.
- 4×H100 instead of 8×H100 halves the GPU-hour cost with minimal wall-clock impact at d=8 scale — gradient accumulation compensates automatically.
- Local smoke tests (d=4, 50 steps, MPS) validated all code paths before committing to cloud runs, avoiding wasted credits on broken configs.

---

## Translation to Larger Runs

### SwiGLU at full scale

SwiGLU's advantages are well-documented at large scale — it is the default MLP activation in LLaMA (7B–70B), PaLM (540B), and Mistral (7B). If the picochat result shows improvement, it is likely to be conservative: the gating mechanism's benefit compounds with model depth and width as the learned gates become more expressive. A follow-up experiment replacing ReLU² with SwiGLU in the d=24 speedrun would directly test whether the val_bpb improvement translates — this would be a strong candidate for a leaderboard submission given the zero additional compute cost.

**Risk at larger scale:** nanochat's ReLU² was likely chosen deliberately — its sparsity and smoothness properties may interact well with Muon's Newton-Schulz orthogonalization in ways that SwiGLU does not. Any speedrun submission would need to verify that Muon's convergence properties are preserved.

### CLA at full scale

CLA's primary motivation is memory efficiency, which scales with model size and sequence length. At d=24 with `seq_len=2048`, the KV cache per token is `2 × n_layer × n_kv_head × head_dim` bytes — roughly 48KB per token in float16. CLA-2 halves this, enabling larger effective batch sizes and longer contexts at the same memory budget. Combined with GQA (4:1 ratio), CLA-2 could yield a 4× total KV cache reduction relative to baseline MHA.

The quality-neutrality finding from Brandon et al. at 1B scale is likely to hold or improve at d=24 because deeper models have more redundancy between adjacent layers' KV representations — the empirical correlation that motivates CLA grows with depth. The picochat result at d=8 is the most conservative test of this hypothesis.

**Practical path to the speedrun:** implement CLA-2 in the d=24 speedrun with the shared KV restricted to pairs of same-window-type layers (LL pairs share, SS pairs share, but LS boundaries do not share). This avoids the cross-window-type interaction noted in the implementation above and should be quality-neutral while delivering meaningful memory savings.
