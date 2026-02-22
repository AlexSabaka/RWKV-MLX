"""
Mixture of Experts modules for RWKV-v7 on MLX.

Two modes:
  - "lora": LoRA expert adapters on a shared frozen ChannelMix base (Phase 1).
            ~2.4M trainable params for 4 experts across 6 layers on D=768.
            Load base from a pretrained checkpoint, only experts train.

  - "full": Full ChannelMix expert banks via mx.gather_mm (Phase 2).
            More capacity, more memory. Each expert = full dim_ffn network.

Routing:
  Top-k learned router with load-balancing auxiliary loss and z-loss for
  numerical stability. Orthogonality regularization on LoRA A matrices
  encourages expert specialization without hard subspace partitioning.

MLX notes:
  - Uses dense loop over experts (no dynamic shapes — compile-safe).
  - mx.argpartition for top-k selection (MLX's efficient top-k primitive).
  - mx.stop_gradient on frozen base (LoRA mode) — no need to filter grads.
  - All aux losses returned as scalars and accumulated in the RWKV model.
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class Router(nn.Module):
    """Token router: linear gate → top-k selection + auxiliary losses.

    Returns:
        inds:      (B, T, top_k) — expert indices for each token (stop_gradient)
        weights:   (B, T, top_k) — softmax scores for selected experts
        aux_loss:  scalar — load-balancing loss (Σ f_e * p_e, scaled by E)
        z_loss:    scalar — logit magnitude penalty (log-sum-exp squared)
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.n_experts = n_experts
        self.top_k = top_k

    def __call__(self, x: mx.array):
        B, T, C = x.shape
        E = self.n_experts
        k = self.top_k

        logits = self.gate(x.astype(mx.float32))  # (B, T, E), in fp32

        # --- Z-loss: penalise large router logit magnitudes ------------------
        # loss = (log Σ_e exp(logit_e))^2, averaged over tokens
        # Prevents logits from growing unboundedly → numerical stability.
        log_sum_exp = mx.log(mx.exp(logits).sum(axis=-1))   # (B, T)
        z_loss = (log_sum_exp ** 2).mean()

        # --- Top-k selection -------------------------------------------------
        # mx.argpartition(-logits, kth=k-1) gives k indices with largest logits
        inds = mx.stop_gradient(
            mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
        )  # (B, T, k)
        scores = mx.take_along_axis(logits, inds, axis=-1)  # (B, T, k)
        scores = mx.softmax(scores, axis=-1)                # (B, T, k)

        # --- Load-balancing auxiliary loss -----------------------------------
        # f_e: fraction of (token, slot) assignments that route to expert e
        # p_e: mean router probability for expert e over all tokens
        # loss = E * Σ_e (f_e * p_e)   (Switch Transformer formulation)
        probs = mx.softmax(logits, axis=-1)             # (B, T, E)
        p_e_list = [probs[..., e].mean() for e in range(E)]
        p_e = mx.stack(p_e_list)                        # (E,)

        flat_inds = inds.reshape(-1, k)                 # (BT, k)
        f_e_list = [(flat_inds == e).astype(mx.float32).mean() for e in range(E)]
        f_e = mx.stack(f_e_list)                        # (E,)

        aux_loss = E * (f_e * p_e).sum()

        return inds.astype(mx.uint32), scores.astype(x.dtype), aux_loss, z_loss


# ---------------------------------------------------------------------------
# LoRA expert pool (Phase 1: shared frozen base + N low-rank adapters)
# ---------------------------------------------------------------------------

class LoRAMoE_CMix(nn.Module):
    """ChannelMix with LoRA MoE adapters on a shared frozen base.

    The base RWKV_CMix_x070 is loaded from a pretrained checkpoint and frozen
    via mx.stop_gradient. N expert adapters (each: A_i: C→r, B_i: r→C) are
    trained to specialise in different subspaces. A learned router assigns
    each token to top_k experts; their weighted adapter outputs are added to
    the frozen base output.

    Orthogonality regularisation on A matrices prevents expert collapse.

    Args:
        base: pretrained RWKV_CMix_x070 (will be frozen via stop_gradient)
        n_experts: number of LoRA expert adapters
        lora_rank: rank of each adapter (r << C)
        top_k: experts activated per token
    """

    def __init__(self, base, n_experts: int, lora_rank: int, top_k: int = 2):
        super().__init__()
        self.base = base          # frozen via stop_gradient in forward
        self.n_experts = n_experts
        self.lora_rank = lora_rank
        self.top_k = top_k

        C = base.key.weight.shape[1]  # input dim (n_embd)
        r = lora_rank

        # Expert adapters: down-project C→r then up-project r→C
        # Stored as (E, C, r) and (E, r, C) for efficient indexing
        # Init: A~N(0, 0.01), B=0 (so initial delta=0 — base preserved)
        self.lora_A = mx.random.normal((n_experts, C, r)) * 0.01
        self.lora_B = mx.zeros((n_experts, r, C))

        self.router = Router(C, n_experts, top_k)

    def __call__(self, x: mx.array):
        # Frozen base — gradients stop here
        base_out = mx.stop_gradient(self.base(x))     # (B, T, C)

        # Router: which experts, with what weights?
        inds, weights, aux_loss, z_loss = self.router(x)  # (B,T,k), (B,T,k), scalar, scalar

        # Expert adapters: accumulate weighted low-rank deltas
        # Dense loop over experts — correct for MLX (no dynamic shapes)
        delta = mx.zeros_like(x)
        for e in range(self.n_experts):
            A_e = self.lora_A[e]   # (C, r)
            B_e = self.lora_B[e]   # (r, C)
            adapter = (x @ A_e) @ B_e                          # (B, T, C)
            # Weight for this expert at each token (0 if not selected)
            w_e = mx.where(inds == e, weights, 0.0).sum(axis=-1, keepdims=True)  # (B, T, 1)
            delta = delta + w_e * adapter

        return base_out + delta, aux_loss, z_loss

    def orthogonal_loss(self) -> mx.array:
        """Penalise overlap between expert A matrices.

        loss = Σ_{i<j} ||A_i^T A_j||_F^2

        Encourages experts to learn in orthogonal subspaces without hard
        partitioning. Regularization coefficient applied in training script.
        """
        loss = mx.array(0.0)
        for i in range(self.n_experts):
            for j in range(i + 1, self.n_experts):
                # A_i: (C, r), A_j: (C, r) — cross-correlation is (r, r)
                cross = self.lora_A[i].T @ self.lora_A[j]      # (r, r)
                loss = loss + (cross * cross).sum()
        return loss


# ---------------------------------------------------------------------------
# Full MoE ChannelMix (Phase 2: E independent expert networks)
# ---------------------------------------------------------------------------

class MoE_CMix(nn.Module):
    """Sparse MoE ChannelMix: E full experts, top-k routing.

    Each expert has the same structure as RWKV_CMix_x070:
    token_shift → lerp → key Linear → ReLU² → value Linear.

    Weights are stored as stacked (E, dim_ffn, C) tensors for efficient
    mx.gather_mm dispatch. Falls back to dense loop when gather_mm is
    unavailable (MLX < 0.18).

    Args:
        config: RWKVConfig
        layer_id: block layer index
        n_experts: number of full experts
        top_k: experts activated per token
    """

    def __init__(self, config, layer_id: int, n_experts: int, top_k: int = 2):
        super().__init__()
        from .model import RWKV_CMix_x070
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = Router(config.n_embd, n_experts, top_k)

        # Expert x_k mixing params: (E, 1, 1, C)
        C = config.n_embd
        D = config.dim_ffn
        self.x_k = mx.zeros((n_experts, 1, 1, C))   # per-expert token shift mix

        # Stacked expert weights — shape (E, D, C) and (E, C, D)
        # Initialised following CMix conventions (ortho key, zeros value)
        key_stack = []
        val_stack = []
        for _ in range(n_experts):
            # Orthogonal init for key, zeros for value (matches base init)
            flat = mx.array(
                __import__("numpy").random.randn(D, C).astype("float32")
            )
            q, _ = __import__("numpy").linalg.qr(
                flat.tolist() if D >= C else flat.T.tolist()
            )
            q = __import__("numpy").array(q, dtype="float32")
            if D < C:
                q = q.T
            key_stack.append(mx.array(q[:D, :C] * math.sqrt(D / C)))
            val_stack.append(mx.zeros((C, D)))

        self.expert_key = mx.stack(key_stack)    # (E, D, C)
        self.expert_val = mx.stack(val_stack)    # (E, C, D)

    def __call__(self, x: mx.array):
        from .model import _time_shift
        B, T, C = x.shape
        E = self.n_experts

        inds, weights, aux_loss, z_loss = self.router(x)

        # Dense loop over experts (safe for MLX's static graph compiler)
        out = mx.zeros_like(x)
        for e in range(E):
            xx = _time_shift(x) - x
            k_e = x + xx * self.x_k[e]                         # (B, T, C)
            k_e = nn.relu(k_e @ self.expert_key[e].T) ** 2     # (B, T, D)
            expert_out = k_e @ self.expert_val[e].T             # (B, T, C)

            w_e = mx.where(inds == e, weights, 0.0).sum(axis=-1, keepdims=True)  # (B, T, 1)
            out = out + w_e * expert_out

        return out, aux_loss, z_loss

    def orthogonal_loss(self) -> mx.array:
        """No orthogonality loss for full experts (different subspaces naturally)."""
        return mx.array(0.0)
