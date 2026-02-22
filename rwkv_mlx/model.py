"""
RWKV-v7 "Goose" (x070) model implementation in MLX for Apple Silicon.

This is a clean reimplementation of the RWKV-7 architecture using MLX,
designed to fully utilize Apple's Metal GPU via unified memory.

Architecture reference: RWKV-v7/rwkv_v7_demo.py and RWKV-v7/train_temp/src/model.py
"""

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class RWKVConfig:
    """Configuration for RWKV-v7 model."""
    n_layer: int = 12
    n_embd: int = 768
    vocab_size: int = 65568  # 65536 base + 10 special + padding for alignment
    head_size: int = 64
    ctx_len: int = 4096
    dim_att: int = 0      # defaults to n_embd
    dim_ffn: int = 0      # defaults to int((n_embd * 3.5) // 32 * 32)
    # LoRA dimension overrides (0 = use formula defaults)
    d_decay_lora: int = 0
    d_aaa_lora: int = 0
    d_mv_lora: int = 0
    d_gate_lora: int = 0
    # MoE configuration (0 = disabled, fully backward-compatible)
    moe_num_experts: int = 0       # number of experts (0 = plain ChannelMix)
    moe_top_k: int = 2             # experts activated per token
    moe_mode: str = "lora"         # "lora" (adapter experts) | "full" (independent FFNs)
    moe_lora_rank: int = 64        # rank r for LoRA experts (mode="lora" only)
    moe_layers: tuple = ()         # which layer indices get MoE (empty = all layers)
    moe_aux_loss_coeff: float = 0.01    # load-balancing loss weight
    moe_z_loss_coeff: float = 0.001     # router z-loss weight
    moe_ortho_coeff: float = 0.001      # orthogonality loss weight (lora mode only)

    def __post_init__(self):
        if self.dim_att <= 0:
            self.dim_att = self.n_embd
        if self.dim_ffn <= 0:
            self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32)

    @property
    def n_head(self) -> int:
        return self.dim_att // self.head_size

    def lora_dim(self, kind: str) -> int:
        """Compute LoRA dimensions. Uses explicit overrides if set, else formulas."""
        C = self.n_embd
        if kind == "decay":
            return self.d_decay_lora or max(32, int(round((2.5 * (C ** 0.5)) / 32) * 32))
        elif kind == "aaa":
            return self.d_aaa_lora or max(32, int(round((2.5 * (C ** 0.5)) / 32) * 32))
        elif kind == "mv":
            return self.d_mv_lora or max(32, int(round((1.7 * (C ** 0.5)) / 32) * 32))
        elif kind == "gate":
            return self.d_gate_lora or max(32, int(round((5.0 * (C ** 0.5)) / 32) * 32))
        raise ValueError(f"Unknown LoRA kind: {kind}")


def _time_shift(x: mx.array) -> mx.array:
    """Shift tokens by 1 position along the time dimension.

    Equivalent to nn.ZeroPad2d((0, 0, 1, -1)) in PyTorch.
    Prepends a zero row and drops the last row.
    """
    # x shape: (B, T, C)
    zero = mx.zeros_like(x[:, :1, :])
    return mx.concatenate([zero, x[:, :-1, :]], axis=1)


from .kernels.wkv7_metal import wkv7_metal_forward, wkv7_metal_backward


@mx.custom_function
def _wkv7_custom(r, w, k, v, a, b):
    """WKV-7 with custom Metal forward+backward (head_size=64)."""
    y, _, _ = wkv7_metal_forward(r, w, k, v, a, b, head_size=64)
    return y


@_wkv7_custom.vjp
def _wkv7_custom_vjp(primals, cotangent, output):
    """Custom VJP using Metal backward kernel."""
    r, w, k, v, a, b = primals
    dy = cotangent
    # Re-run forward to get saved tensors (s, sa)
    _, s, sa = wkv7_metal_forward(r, w, k, v, a, b, head_size=64)
    dr, dw, dk, dv, da, db = wkv7_metal_backward(
        r, w, k, v, a, b, dy, s, sa, head_size=64
    )
    return dr, dw, dk, dv, da, db


def wkv7_recurrent(r, w, k, v, a, b, head_size: int = 64):
    """WKV-7 computation using custom Metal kernels for forward+backward.

    Uses fused Metal kernels (ported from CUDA) that process the entire
    sequence in a single GPU dispatch, avoiding the massive overhead of
    autograd through 512+ sequential Python operations.

    Args:
        r, w, k, v, a, b: (B, T, C) tensors
        head_size: size of each head (N=64)

    Returns:
        output: (B, T, C) tensor
    """
    assert head_size == 64, f"Metal WKV kernel only supports head_size=64, got {head_size}"
    return _wkv7_custom(r, w, k, v, a, b)


class RWKV_Tmix_x070(nn.Module):
    """RWKV-7 Time Mixing (attention) module.

    This is the core innovation of RWKV — a linear attention mechanism
    with a recurrent state that can process sequences in parallel during
    training and recurrently during inference.
    """

    def __init__(self, config: RWKVConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.head_size = config.head_size
        self.n_head = config.n_head
        C = config.n_embd
        H = self.n_head
        N = self.head_size

        # Token shift mixing parameters
        self.x_r = mx.zeros((1, 1, C))
        self.x_w = mx.zeros((1, 1, C))
        self.x_k = mx.zeros((1, 1, C))
        self.x_v = mx.zeros((1, 1, C))
        self.x_a = mx.zeros((1, 1, C))
        self.x_g = mx.zeros((1, 1, C))

        # Decay (w) LoRA
        D_DECAY = config.lora_dim("decay")
        self.w0 = mx.zeros((1, 1, C))
        self.w1 = mx.zeros((C, D_DECAY))
        self.w2 = mx.zeros((D_DECAY, C))

        # In-context learning rate (a) LoRA
        D_AAA = config.lora_dim("aaa")
        self.a0 = mx.zeros((1, 1, C))
        self.a1 = mx.zeros((C, D_AAA))
        self.a2 = mx.zeros((D_AAA, C))

        # Value residual (v) LoRA
        D_MV = config.lora_dim("mv")
        self.v0 = mx.zeros((1, 1, C))
        self.v1 = mx.zeros((C, D_MV))
        self.v2 = mx.zeros((D_MV, C))

        # Gate LoRA
        D_GATE = config.lora_dim("gate")
        self.g1 = mx.zeros((C, D_GATE))
        self.g2 = mx.zeros((D_GATE, C))

        # Key normalization & bonus parameters
        self.k_k = mx.zeros((1, 1, C))
        self.k_a = mx.zeros((1, 1, C))
        self.r_k = mx.zeros((H, N))

        # Linear projections
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)

        # Group normalization with RWKV-specific eps
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

    def __call__(self, x: mx.array, v_first: mx.array) -> tuple[mx.array, mx.array]:
        B, T, C = x.shape
        H = self.n_head
        N = self.head_size

        # Token shift
        xx = _time_shift(x) - x

        # Lerp mixing
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        # Compute r, w, k, v, a, g
        r = self.receptance(xr)
        w = -nn.softplus(-(self.w0 + mx.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)

        # Value residual from first layer
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * mx.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        # In-context learning rate
        a = mx.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)

        # Gate
        g = mx.sigmoid(xg @ self.g1) @ self.g2

        # Key normalization
        kk = k * self.k_k
        kk_reshaped = kk.reshape(B, T, H, -1)
        kk_norm = kk_reshaped / (mx.sqrt((kk_reshaped * kk_reshaped).sum(axis=-1, keepdims=True)) + 1e-12)
        kk = kk_norm.reshape(B, T, C)

        # Scale k by in-context learning rate
        k = k * (1 + (a - 1) * self.k_a)

        # WKV7 core computation
        x = wkv7_recurrent(r, w, k, v, -kk, kk * a, N)

        # Group normalization
        x = self.ln_x(x.reshape(B * T, C)).reshape(B, T, C)

        # Bonus term: per-head dot product attention shortcut
        r_h = r.reshape(B, T, H, -1)
        k_h = k.reshape(B, T, H, -1)
        v_h = v.reshape(B, T, H, -1)
        bonus = ((r_h * k_h * self.r_k).sum(axis=-1, keepdims=True) * v_h).reshape(B, T, C)
        x = x + bonus

        # Output projection with gate
        x = self.output(x * g)
        return x, v_first


class RWKV_CMix_x070(nn.Module):
    """RWKV-7 Channel Mixing (FFN) module.

    Simple feed-forward network with token shift mixing and squared ReLU activation.
    """

    def __init__(self, config: RWKVConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        C = config.n_embd

        self.x_k = mx.zeros((1, 1, C))
        self.key = nn.Linear(C, config.dim_ffn, bias=False)
        self.value = nn.Linear(config.dim_ffn, C, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        xx = _time_shift(x) - x
        k = x + xx * self.x_k
        k = nn.relu(self.key(k)) ** 2
        return self.value(k)


class Block(nn.Module):
    """RWKV-7 Transformer block: LayerNorm → TimeMix → LayerNorm → ChannelMix.

    When MoE is configured for this layer, ChannelMix is replaced by a
    LoRAMoE_CMix or MoE_CMix module. Aux losses (load-balance, z-loss,
    ortho) are returned alongside the hidden state for accumulation.
    """

    def __init__(self, config: RWKVConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.is_moe = False

        if layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.att = RWKV_Tmix_x070(config, layer_id)

        # MoE ChannelMix if configured for this layer
        use_moe = (
            config.moe_num_experts > 0
            and (not config.moe_layers or layer_id in config.moe_layers)
        )
        if use_moe:
            self.is_moe = True
            # Note: LoRA mode requires base weights to be injected after init
            # via Block.init_moe_from_base(). For fresh init, base is a plain
            # RWKV_CMix_x070 that will be overwritten during weight loading.
            base_ffn = RWKV_CMix_x070(config, layer_id)
            if config.moe_mode == "lora":
                from .moe import LoRAMoE_CMix
                self.ffn = LoRAMoE_CMix(
                    base=base_ffn,
                    n_experts=config.moe_num_experts,
                    lora_rank=config.moe_lora_rank,
                    top_k=config.moe_top_k,
                )
            else:
                from .moe import MoE_CMix
                self.ffn = MoE_CMix(config, layer_id, config.moe_num_experts, config.moe_top_k)
        else:
            self.ffn = RWKV_CMix_x070(config, layer_id)

    def __call__(self, x: mx.array, v_first: mx.array) -> tuple:
        """Returns (x, v_first) for plain blocks, (x, v_first, aux, z, ortho) for MoE blocks."""
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn

        if self.is_moe:
            ffn_out, aux_loss, z_loss = self.ffn(self.ln2(x))
            x = x + ffn_out
            ortho_loss = self.ffn.orthogonal_loss()
            return x, v_first, aux_loss, z_loss, ortho_loss
        else:
            x = x + self.ffn(self.ln2(x))
            return x, v_first


class RWKV(nn.Module):
    """RWKV-v7 "Goose" Language Model.

    A complete implementation of the RWKV-7 architecture in MLX,
    supporting both training (parallel mode) and inference.
    """

    def __init__(self, config: RWKVConfig):
        super().__init__()
        self.config = config

        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config, i) for i in range(config.n_layer)]
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def __call__(self, idx: mx.array) -> tuple[mx.array, mx.array]:
        """Forward pass in GPT (parallel) mode.

        Args:
            idx: (B, T) integer token indices

        Returns:
            logits:   (B, T, vocab_size) unnormalized log-probabilities
            aux_loss: scalar — sum of MoE load-balance, z-loss, and
                      orthogonality losses across all MoE layers.
                      Is mx.array(0.0) when no MoE layers are present.
        """
        x = self.emb(idx)

        v_first = mx.zeros_like(x)
        aux_loss = mx.array(0.0)
        for block in self.blocks:
            result = block(x, v_first)
            if len(result) == 5:
                # MoE block: (x, v_first, aux, z, ortho)
                x, v_first, aux, z, ortho = result
                cfg = self.config
                aux_loss = aux_loss + cfg.moe_aux_loss_coeff * aux \
                                    + cfg.moe_z_loss_coeff * z \
                                    + cfg.moe_ortho_coeff * ortho
            else:
                x, v_first = result

        x = self.ln_out(x)
        return self.head(x), aux_loss

    def forward_rnn(self, token: int, state: list) -> tuple[mx.array, list]:
        """Forward pass in RNN (sequential) mode for inference.

        Processes a single token with maintained state for efficient
        autoregressive generation.

        Args:
            token: single token index
            state: list of [att_x_prev, att_state, ffn_x_prev] per layer

        Returns:
            logits: (vocab_size,) unnormalized log-probabilities
            state: updated state list
        """
        config = self.config
        C = config.n_embd
        H = config.n_head
        N = config.head_size

        x = self.emb.weight[token]

        v_first = mx.zeros((C,))
        for i, block in enumerate(self.blocks):
            # Layer norm
            if i == 0:
                x = _layer_norm_1d(x, block.ln0.weight, block.ln0.bias)
            xx = _layer_norm_1d(x, block.ln1.weight, block.ln1.bias)

            # Time mixing (RNN mode)
            xx, state[i * 3], state[i * 3 + 1], v_first = _time_mixing_rnn(
                i, H, N, xx, state[i * 3], v_first, state[i * 3 + 1],
                block.att
            )
            x = x + xx

            # Channel mixing (RNN mode)
            xx = _layer_norm_1d(x, block.ln2.weight, block.ln2.bias)
            xx, state[i * 3 + 2] = _channel_mixing_rnn(
                xx, state[i * 3 + 2], block.ffn
            )
            x = x + xx

        x = _layer_norm_1d(x, self.ln_out.weight, self.ln_out.bias)
        logits = self.head.weight @ x
        return logits, state

    def init_state(self) -> list:
        """Create initial RNN state (all zeros)."""
        config = self.config
        state = []
        for i in range(config.n_layer):
            state.append(mx.zeros((config.n_embd,)))  # att_x_prev
            state.append(mx.zeros((config.n_head, config.head_size, config.head_size)))  # att_state
            state.append(mx.zeros((config.n_embd,)))  # ffn_x_prev
        return state


def _layer_norm_1d(x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    """Layer normalization for a 1D vector."""
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    return (x - mean) / mx.sqrt(var + 1e-5) * weight + bias


def _time_mixing_rnn(
    layer_id: int, H: int, N: int,
    x: mx.array, x_prev: mx.array,
    v_first: mx.array, state: mx.array,
    att: RWKV_Tmix_x070
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """RNN-mode time mixing for single token inference."""
    xx = x_prev - x
    xr = x + xx * att.x_r.squeeze()
    xw = x + xx * att.x_w.squeeze()
    xk = x + xx * att.x_k.squeeze()
    xv = x + xx * att.x_v.squeeze()
    xa = x + xx * att.x_a.squeeze()
    xg = x + xx * att.x_g.squeeze()

    r = att.receptance.weight @ xr
    w = mx.tanh(xw @ att.w1) @ att.w2
    k = att.key.weight @ xk
    v = att.value.weight @ xv
    a = mx.sigmoid(att.a0.squeeze() + (xa @ att.a1) @ att.a2)
    g = mx.sigmoid(xg @ att.g1) @ att.g2

    kk = k * att.k_k.squeeze()
    kk_reshaped = kk.reshape(H, N)
    kk_norm = kk_reshaped / (mx.sqrt((kk_reshaped * kk_reshaped).sum(axis=-1, keepdims=True)) + 1e-12)
    kk = kk_norm.reshape(-1)

    k = k * (1 + (a - 1) * att.k_a.squeeze())

    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * mx.sigmoid(att.v0.squeeze() + (xv @ att.v1) @ att.v2)

    # Fused w clamping: exp(-exp(-softplus(-w0 - w) - 0.5)) = exp(-0.606531 * sigmoid(w0 + w))
    w_raw = att.w0.squeeze() + w.astype(mx.float32)
    w_decay = mx.exp(-0.606531 * mx.sigmoid(w_raw))

    # State update
    vk = v.reshape(H, N, 1) @ k.reshape(H, 1, N)
    ab = (-kk).reshape(H, N, 1) @ (kk * a).reshape(H, 1, N)
    state = state * w_decay.reshape(H, 1, N) + state @ ab.astype(mx.float32) + vk.astype(mx.float32)
    out = (state.astype(x.dtype) @ r.reshape(H, N, 1)).squeeze(-1)

    # Group norm (use the pretrained ln_x weights, not a fresh GroupNorm)
    out_flat = out.reshape(1, H * N)
    out_flat = att.ln_x(out_flat)
    out = out_flat.reshape(H * N)

    # Bonus
    r_k = att.r_k.reshape(-1)
    bonus = ((r * k * r_k).reshape(H, N).sum(axis=-1, keepdims=True) * v.reshape(H, N)).reshape(H * N)
    out = out + bonus

    return att.output.weight @ (out * g), x, state, v_first


def _channel_mixing_rnn(
    x: mx.array, x_prev: mx.array,
    ffn: RWKV_CMix_x070
) -> tuple[mx.array, mx.array]:
    """RNN-mode channel mixing for single token inference."""
    xx = x_prev - x
    k = x + xx * ffn.x_k.squeeze()
    k = mx.maximum(ffn.key.weight @ k, 0) ** 2
    return ffn.value.weight @ k, x
