"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # SwiGLU: replace ReLU² MLP activation with SwiGLU (Shazeer 2020)
    swiglu: bool = False
    # CLA: cross-layer attention KV sharing factor (Brandon et al. 2024)
    # 1 = disabled (every layer computes its own KV)
    # 2 = CLA-2 (even layers reuse KV from the preceding odd layer)
    cla_sharing: int = 1
    # FFN sharing: share one MLP across all transformer layers (MobiLlama, 2024)
    # When True, all blocks use the same MLP weights instead of per-layer MLPs
    shared_ffn: bool = False
    # Differential Attention (Microsoft Research, ICLR 2025, arXiv 2410.05258)
    # Two softmax maps whose difference cancels attention noise, focusing on relevant tokens
    differential_attn: bool = False
    # Mixture of Depths (Raposo et al., arXiv 2404.02258)
    # Even-indexed layers route top mod_capacity fraction of tokens; odd layers are full-capacity.
    # Incompatible with cla_sharing > 1. KV cache inference not supported.
    mod_routing: bool = False
    mod_capacity: float = 0.125  # top 12.5% of tokens pass through MoD layers (paper recommendation)


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.differential_attn = config.differential_attn
        self.n_embd = config.n_embd
        # For differential attention, each "super-head" uses two Q/K groups.
        # We halve n_head/n_kv_head so total output dim stays n_embd (n_head * 2 * head_dim).
        if config.differential_attn:
            assert config.n_head % 2 == 0 and config.n_kv_head % 2 == 0, \
                "n_head and n_kv_head must be even for differential attention"
            self.n_head = config.n_head // 2
            self.n_kv_head = config.n_kv_head // 2
        else:
            self.n_head = config.n_head
            self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head  # always use original n_head for head_dim
        assert self.n_embd % config.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        # CLA follower layers reuse K and V from the preceding leader layer.
        # Not creating c_k/c_v avoids None gradients in the Muon optimizer.
        self.is_cla_follower = config.cla_sharing > 1 and layer_idx % config.cla_sharing != 0

        if config.differential_attn:
            # Two Q groups per super-head (always needed)
            self.c_q = nn.Linear(self.n_embd, 2 * self.n_head * self.head_dim, bias=False)
            # CLA followers reuse k/v from leader — don't create c_k, c_v
            if not self.is_cla_follower:
                self.c_k = nn.Linear(self.n_embd, 2 * self.n_kv_head * self.head_dim, bias=False)
                self.c_v = nn.Linear(self.n_embd, 2 * self.n_kv_head * self.head_dim, bias=False)
            # Lambda scalars for differential weighting (learned, one per head_dim)
            self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim))
            self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim))
            self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim))
            self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim))
            # Fixed init scalar (not learned): lambda_init = 0.8 - 0.6 * exp(-0.3 * layer_idx)
            self.lambda_init = 0.8 - 0.6 * exp(-0.3 * layer_idx)
            # Per-head RMSNorm over 2*head_dim, no learnable params
            self.subln = nn.RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)
        else:
            self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
            if not self.is_cla_follower:
                self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
                self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache, shared_kv=None, return_kv=False):
        B, T, C = x.size()
        cos, sin = cos_sin

        if self.differential_attn:
            if kv_cache is not None:
                raise NotImplementedError("KV cache inference not supported for differential attention")

            # Split Q into q1, q2
            q_full = self.c_q(x).view(B, T, self.n_head, 2, self.head_dim)
            q1, q2 = q_full[..., 0, :], q_full[..., 1, :]  # (B, T, n_head, head_dim) each

            if shared_kv is not None:
                # CLA follower: leader passed k as (B,T,n_kv_head,2*head_dim), v as same
                k_flat, v = shared_kv
                k1 = k_flat[..., :self.head_dim]   # (B, T, n_kv_head, head_dim)
                k2 = k_flat[..., self.head_dim:]
            else:
                k_full = self.c_k(x).view(B, T, self.n_kv_head, 2, self.head_dim)
                k1, k2 = k_full[..., 0, :], k_full[..., 1, :]
                v = self.c_v(x).view(B, T, self.n_kv_head, 2 * self.head_dim)

            # Value embedding (ve shape: (B, T, n_kv_head * 2 * head_dim))
            if ve is not None:
                ve = ve.view(B, T, self.n_kv_head, 2 * self.head_dim)
                gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head)
                v = v + gate.unsqueeze(-1) * ve

            # RoPE + QK-norm for all four groups.
            # QK-norm is required for training stability with Muon: without it,
            # Muon grows Q/K weight norms until q@k^T overflows in bfloat16.
            q1, q2 = apply_rotary_emb(q1, cos, sin), apply_rotary_emb(q2, cos, sin)
            k1, k2 = apply_rotary_emb(k1, cos, sin), apply_rotary_emb(k2, cos, sin)
            q1, q2, k1, k2 = norm(q1), norm(q2), norm(k1), norm(k2)

            # Differential lambda scalar: exp(lq1·lk1) - exp(lq2·lk2) + lambda_init
            # Clamp dot products before exp to prevent bfloat16 overflow (exp(>10) = inf)
            lam = (torch.dot(self.lambda_q1, self.lambda_k1).clamp(-10, 10).exp()
                   - torch.dot(self.lambda_q2, self.lambda_k2).clamp(-10, 10).exp()
                   + self.lambda_init)

            # Two attention maps over the same V (shape: (B, T, n_head, 2*head_dim))
            A1 = flash_attn.flash_attn_func(q1, k1, v, causal=True, window_size=window_size)
            A2 = flash_attn.flash_attn_func(q2, k2, v, causal=True, window_size=window_size)
            y = A1 - lam * A2  # (B, T, n_head, 2*head_dim)

            # Per-head RMSNorm scaled by (1 - lambda_init)
            y = self.subln(y) * (1 - self.lambda_init)

            y = y.contiguous().view(B, T, -1)  # (B, T, n_embd)
            y = self.c_proj(y)
            if return_kv:
                # Return k as flat (B,T,n_kv_head,2*head_dim) for CLA follower layers
                k_flat = torch.cat([k1, k2], dim=-1)
                return y, k_flat, v
            return y

        # --- Standard (non-differential) path ---
        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)

        if shared_kv is not None:
            # CLA: reuse K and V from the previous layer instead of computing new ones
            k, v = shared_kv
        else:
            k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        # Note: applied after CLA reuse so even CLA layers get per-layer value specialization
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 2)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        if return_kv:
            return y, k, v
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.swiglu = config.swiglu
        if config.swiglu:
            # SwiGLU: two projections of size 8/3 * n_embd, rounded to nearest multiple of 64
            # This keeps parameter count approximately equal to the ReLU² MLP (4 * n_embd)
            # because 2 * (8/3) ≈ 4 * (2/2) in terms of total projection parameters
            hidden = round(config.n_embd * 8 / 3 / 64) * 64
            self.c_gate = nn.Linear(config.n_embd, hidden * 2, bias=False)  # projects to [gate | value]
            self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        else:
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        if self.swiglu:
            x, gate = self.c_gate(x).chunk(2, dim=-1)
            x = x * F.silu(gate)
        else:
            x = self.c_fc(x)
            x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class MoDRouter(nn.Module):
    """Scalar router for Mixture of Depths. Projects n_embd -> 1 (no bias)."""
    def __init__(self, n_embd: int):
        super().__init__()
        self.proj = nn.Linear(n_embd, 1, bias=False)

    def forward(self, x):
        return self.proj(x)  # (B, T, 1)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        # If shared_ffn is enabled, this block's mlp is unused — the shared MLP
        # is owned by GPT and passed in at forward time to avoid duplicate params.
        if not config.shared_ffn:
            self.mlp = MLP(config)
        # MoD router on even-indexed layers only
        self.mod_router = (
            MoDRouter(config.n_embd)
            if config.mod_routing and layer_idx % 2 == 0
            else None
        )
        self._mod_capacity_frac = config.mod_capacity

    def forward(self, x, ve, cos_sin, window_size, kv_cache, shared_kv=None, return_kv=False, mlp=None):
        # mlp argument allows GPT to pass in the shared MLP when shared_ffn=True
        active_mlp = mlp if mlp is not None else self.mlp

        if self.mod_router is not None:
            if kv_cache is not None:
                raise NotImplementedError("KV cache inference not supported for MoD layers.")
            B, T, D_embd = x.shape
            capacity = max(1, int(self._mod_capacity_frac * T))

            # 1. Router scores + top-k selection
            router_weights = self.mod_router(x).squeeze(-1)          # (B, T)
            _, top_indices = torch.topk(router_weights, capacity, dim=1)

            # 2. Sort by original position for causal RoPE correctness
            sorted_positions, _ = torch.sort(top_indices, dim=1)     # (B, capacity)

            # 3. Gather selected tokens
            gather_e = sorted_positions[:, :, None].expand(-1, -1, D_embd)
            x_sel = x.gather(1, gather_e)                            # (B, capacity, D_embd)

            # 4. Gather router weights at sorted positions
            router_w_sel = router_weights.gather(1, sorted_positions).unsqueeze(-1)  # (B, capacity, 1)

            # 5. Gather cos/sin at original positions
            cos, sin = cos_sin
            D2 = cos.size(-1)
            gather_r = sorted_positions[:, :, None, None].expand(-1, -1, 1, D2)
            cos_sel = cos.expand(B, -1, -1, -1).gather(1, gather_r)  # (B, capacity, 1, D2)
            sin_sel = sin.expand(B, -1, -1, -1).gather(1, gather_r)

            # 6. Gather value embeddings at selected positions (if present)
            ve_sel = None
            if ve is not None:
                gather_v = sorted_positions[:, :, None].expand(-1, -1, ve.size(-1))
                ve_sel = ve.gather(1, gather_v)

            # 7. Run attention + MLP on compact tensor
            x_out = x_sel + self.attn(norm(x_sel), ve_sel, (cos_sel, sin_sel),
                                      window_size, kv_cache=None, shared_kv=None)
            x_out = x_out + active_mlp(norm(x_out))

            # 8. Weighted delta + scatter back
            delta = x_out - x_sel
            weighted_delta = router_w_sel * delta                    # (B, capacity, D_embd)
            x = x.scatter_add(1, gather_e, weighted_delta)
            return x

        # --- Standard path (odd layers, or mod_routing=False) ---
        if return_kv:
            attn_out, k, v = self.attn(norm(x), ve, cos_sin, window_size, kv_cache, shared_kv=shared_kv, return_kv=True)
            x = x + attn_out
            x = x + active_mlp(norm(x))
            return x, k, v
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache, shared_kv=shared_kv)
        x = x + active_mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        if config.mod_routing and config.cla_sharing > 1:
            raise ValueError("mod_routing=True is incompatible with cla_sharing > 1.")
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        # Shared FFN (MobiLlama): one MLP owned here, passed to every block at forward time.
        # Block.mlp is not created when shared_ffn=True, so no duplicate parameters.
        self.shared_mlp = MLP(config) if config.shared_ffn else None
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            if not block.attn.is_cla_follower:  # followers have no c_k, c_v (standard or diff_attn)
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            if not self.config.shared_ffn:
                if self.config.swiglu:
                    torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)
                else:
                    torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Shared FFN init (done once, not per-block)
        if self.config.shared_ffn:
            if self.config.swiglu:
                torch.nn.init.uniform_(self.shared_mlp.c_gate.weight, -s, s)
            else:
                torch.nn.init.uniform_(self.shared_mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(self.shared_mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # MoD router: zero init so weighted_delta=0 at step 0 (MoD layers start as pure residuals)
        for block in self.transformer.h:
            if block.mod_router is not None:
                torch.nn.init.zeros_(block.mod_router.proj.weight)

        # Differential attention lambda vectors: small normal init (per paper)
        if self.config.differential_attn:
            for block in self.transformer.h:
                torch.nn.init.normal_(block.attn.lambda_q1, mean=0, std=0.1)
                torch.nn.init.normal_(block.attn.lambda_k1, mean=0, std=0.1)
                torch.nn.init.normal_(block.attn.lambda_q2, mean=0, std=0.1)
                torch.nn.init.normal_(block.attn.lambda_k2, mean=0, std=0.1)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for i, window_size in enumerate(self.window_sizes):
            window = window_size[0]  # (left, right) tuple, we use left
            if self.config.mod_routing and i % 2 == 0:
                effective_seq = max(1, int(self.config.mod_capacity * t))
            else:
                effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        if self.shared_mlp is not None:
            transformer_matrices += sum(p.numel() for p in self.shared_mlp.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        # Lambda vectors (differential attn) are 1D — Muon requires 2D+, so split them out
        all_h_params = list(self.transformer.h.parameters())
        matrix_params = [p for p in all_h_params if p.ndim >= 2]
        lambda_params = [p for p in all_h_params if p.ndim < 2]
        # shared_mlp lives outside transformer.h — include its params in matrix group
        if self.shared_mlp is not None:
            matrix_params += list(self.shared_mlp.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert (len(list(self.parameters())) ==
                len(matrix_params) + len(lambda_params) + len(embedding_params) +
                len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
        ]
        # Differential attention lambda vectors are 1D — add to AdamW (Muon requires 2D+).
        # Use scalar_lr (not *0.01) since lambdas are the core learned mechanism of diff attn.
        if lambda_params:
            param_groups.append(dict(kind='adamw', params=lambda_params, lr=scalar_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx) # embed current token
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        cla_sharing = self.config.cla_sharing
        mlp = self.shared_mlp  # None if shared_ffn=False, shared MLP otherwise
        shared_kv = None  # holds (k, v) from leader layer for CLA follower layers
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            if cla_sharing > 1 and i % cla_sharing != 0:
                # CLA follower: reuse K and V from the leader layer via local variable
                x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, shared_kv=shared_kv, mlp=mlp)
            elif cla_sharing > 1:
                # CLA leader: compute fresh K and V, return them explicitly for the next layer
                x, k, v = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, return_kv=True, mlp=mlp)
                shared_kv = (k, v)
            else:
                # CLA disabled: standard forward
                x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, mlp=mlp)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
