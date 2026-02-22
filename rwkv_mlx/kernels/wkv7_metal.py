"""
Custom Metal kernels for WKV-7 forward and backward on Apple Silicon.

Ported from the CUDA implementation in RWKV-v7/train_temp/cuda/wkv7_cuda.cu.
The backward kernel enables efficient training by avoiding autograd through
512+ sequential state updates (which causes massive GPU dispatch overhead).

Both kernels process the sequence timestep-by-timestep, parallelizing across
heads and state dimensions using Metal threadgroups.
"""

import mlx.core as mx

# Chunk length for state checkpointing (must divide T evenly).
# Smaller = more accurate gradients (less reverse-step error) but more memory.
# 16 gives <0.1% gradient error while using ~12MB for typical configs.
_CHUNK_LEN = 16


def _make_forward_source(N: int) -> str:
    """Generate WKV7 forward Metal kernel that also saves states for backward.

    The forward kernel saves:
    - sa: per-timestep dot(state_row, a) values needed by backward
    - s: state snapshots at chunk boundaries for backward recomputation
    """
    CL = _CHUNK_LEN
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint gid = threadgroup_position_in_grid.x;

    const uint _N_ = {N};
    const uint _CL_ = {CL};

    uint H = wkv_H[0];
    uint T_val = wkv_T[0];
    uint batch_idx = gid / H;
    uint head_idx = gid % H;

    threadgroup float shared_r[{N}];
    threadgroup float shared_k[{N}];
    threadgroup float shared_w[{N}];
    threadgroup float shared_a[{N}];
    threadgroup float shared_b[{N}];

    float state_local[{N}];
    for (uint j = 0; j < _N_; j++) {{
        state_local[j] = 0.0f;
    }}

    uint n_chunks = T_val / _CL_;

    for (uint t = 0; t < T_val; t++) {{
        uint offset = batch_idx * T_val * H * _N_ + t * H * _N_ + head_idx * _N_ + tid;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared_r[tid] = float(r[offset]);
        shared_w[tid] = metal::exp(-metal::exp(float(w[offset])));
        shared_k[tid] = float(k[offset]);
        shared_a[tid] = float(a[offset]);
        shared_b[tid] = float(b[offset]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // sa = dot(state_row[tid], a)
        float sa_val = 0.0f;
        for (uint j = 0; j < _N_; j++) {{
            sa_val += shared_a[j] * state_local[j];
        }}
        // Save sa for backward
        sa_out[offset] = sa_val;

        float vv = float(v[offset]);
        float y_val = 0.0f;
        for (uint j = 0; j < _N_; j++) {{
            state_local[j] = state_local[j] * shared_w[j] + shared_k[j] * vv + sa_val * shared_b[j];
            y_val += state_local[j] * shared_r[j];
        }}

        y[offset] = T(y_val);

        // Save state at chunk boundaries for backward
        if ((t + 1) % _CL_ == 0) {{
            uint chunk_idx = t / _CL_;
            // s layout: (B, H, n_chunks, N, N) stored as flat
            // Index: ((batch_idx * H + head_idx) * n_chunks + chunk_idx) * N * N + tid * N + j
            uint base = ((batch_idx * H + head_idx) * n_chunks + chunk_idx) * _N_ * _N_ + tid * _N_;
            for (uint j = 0; j < _N_; j++) {{
                s_out[base + j] = state_local[j];
            }}
        }}
    }}
"""


def _make_backward_source(N: int) -> str:
    """Generate WKV7 backward Metal kernel.

    Processes timesteps in reverse order, using saved states and sa values.
    Computes gradients for all 6 inputs (r, w, k, v, a, b).
    """
    CL = _CHUNK_LEN
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint gid = threadgroup_position_in_grid.x;

    const uint _N_ = {N};
    const uint _CL_ = {CL};

    uint H = wkv_H[0];
    uint T_val = wkv_T[0];
    uint batch_idx = gid / H;
    uint head_idx = gid % H;
    uint n_chunks = T_val / _CL_;

    // stateT[j] = state[j, tid] (transposed view for backward)
    float stateT[{N}];
    float dstate[{N}];
    float dstateT[{N}];
    for (uint j = 0; j < _N_; j++) {{
        stateT[j] = 0.0f;
        dstate[j] = 0.0f;
        dstateT[j] = 0.0f;
    }}

    threadgroup float shared_r[{N}];
    threadgroup float shared_k[{N}];
    threadgroup float shared_w[{N}];
    threadgroup float shared_a[{N}];
    threadgroup float shared_b[{N}];
    threadgroup float shared_v[{N}];
    threadgroup float shared_dy[{N}];
    threadgroup float shared_sa[{N}];
    threadgroup float shared_dSb[{N}];

    for (int t = (int)T_val - 1; t >= 0; t--) {{
        uint offset = batch_idx * T_val * H * _N_ + (uint)t * H * _N_ + head_idx * _N_ + tid;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        float ri = float(r[offset]);
        float wi_fac = -metal::exp(float(w[offset]));
        float wi = metal::exp(wi_fac);
        float ki = float(k[offset]);
        float ai = float(a[offset]);
        float bi = float(b[offset]);
        shared_r[tid] = ri;
        shared_w[tid] = wi;
        shared_k[tid] = ki;
        shared_a[tid] = ai;
        shared_b[tid] = bi;
        shared_v[tid] = float(v[offset]);
        shared_dy[tid] = float(dy[offset]);
        shared_sa[tid] = sa[offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load saved state at chunk boundaries
        if (((uint)t + 1) % _CL_ == 0) {{
            uint chunk_idx = (uint)t / _CL_;
            // s layout: stored as state[tid][j], we need stateT[j] = state[j][tid]
            // So we read state[tid][j] from: base = (...) * N*N + tid*N + j
            // But we need state[j][tid], so: base_j = (...) * N*N + j*N + tid
            uint base_batch_head = (batch_idx * H + head_idx) * n_chunks + chunk_idx;
            for (uint j = 0; j < _N_; j++) {{
                stateT[j] = s[base_batch_head * _N_ * _N_ + j * _N_ + tid];
            }}
        }}

        // dr = dot(stateT, dy)
        float dr_val = 0.0f;
        for (uint j = 0; j < _N_; j++) {{
            dr_val += stateT[j] * shared_dy[j];
        }}
        dr[offset] = T(dr_val);

        // Reverse state update: stateT[j] = (stateT[j] - k[tid]*v[j] - b[tid]*sa[j]) / w[tid]
        float iwi = 1.0f / wi;
        float dyi = shared_dy[tid];
        for (uint j = 0; j < _N_; j++) {{
            stateT[j] = (stateT[j] - ki * shared_v[j] - bi * shared_sa[j]) * iwi;
            dstate[j] += dyi * shared_r[j];
            dstateT[j] += ri * shared_dy[j];
        }}

        // Compute gradients
        float dw_val = 0.0f, dk_val = 0.0f, dv_val = 0.0f, db_val = 0.0f, dSb_val = 0.0f;
        for (uint j = 0; j < _N_; j++) {{
            dw_val += dstateT[j] * stateT[j];
            dk_val += dstateT[j] * shared_v[j];
            dv_val += dstate[j] * shared_k[j];
            dSb_val += dstate[j] * shared_b[j];
            db_val += dstateT[j] * shared_sa[j];
        }}
        // dw includes chain rule through exp(-exp(w)): d/dw_raw = wi * wi_fac
        dw_out[offset] = T(dw_val * wi * wi_fac);
        dk_out[offset] = T(dk_val);
        dv_out[offset] = T(dv_val);
        db_out[offset] = T(db_val);

        // da requires shared memory reduction
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared_dSb[tid] = dSb_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float da_val = 0.0f;
        for (uint j = 0; j < _N_; j++) {{
            da_val += stateT[j] * shared_dSb[j];
        }}
        da_out[offset] = T(da_val);

        // Propagate gradients to previous timestep
        for (uint j = 0; j < _N_; j++) {{
            dstate[j] = dstate[j] * shared_w[j] + dSb_val * shared_a[j];
            dstateT[j] = dstateT[j] * wi + ai * shared_dSb[j];
        }}
    }}
"""


# Cache compiled kernels by head_size
_wkv7_fwd_kernels: dict[int, object] = {}
_wkv7_bwd_kernels: dict[int, object] = {}


def _get_wkv7_fwd_kernel(head_size: int):
    """Lazily compile the WKV7 forward Metal kernel."""
    if head_size not in _wkv7_fwd_kernels:
        source = _make_forward_source(head_size)
        _wkv7_fwd_kernels[head_size] = mx.fast.metal_kernel(
            name=f"wkv7_forward_n{head_size}",
            input_names=["r", "w", "k", "v", "a", "b", "wkv_H", "wkv_T"],
            output_names=["y", "s_out", "sa_out"],
            source=source,
        )
    return _wkv7_fwd_kernels[head_size]


def _get_wkv7_bwd_kernel(head_size: int):
    """Lazily compile the WKV7 backward Metal kernel."""
    if head_size not in _wkv7_bwd_kernels:
        source = _make_backward_source(head_size)
        _wkv7_bwd_kernels[head_size] = mx.fast.metal_kernel(
            name=f"wkv7_backward_n{head_size}",
            input_names=["r", "w", "k", "v", "a", "b", "dy", "s", "sa",
                         "wkv_H", "wkv_T"],
            output_names=["dr", "dw_out", "dk_out", "dv_out", "da_out", "db_out"],
            source=source,
        )
    return _wkv7_bwd_kernels[head_size]


def wkv7_metal_forward(r, w, k, v, a, b, head_size=64):
    """WKV-7 forward pass with state saving for backward.

    Returns (y, s, sa) where s and sa are saved for the backward pass.
    """
    B, T, C = r.shape
    H = C // head_size
    N = head_size
    n_chunks = T // _CHUNK_LEN

    r_flat = r.reshape(B, T, H, N)
    w_flat = w.reshape(B, T, H, N)
    k_flat = k.reshape(B, T, H, N)
    v_flat = v.reshape(B, T, H, N)
    a_flat = a.reshape(B, T, H, N)
    b_flat = b.reshape(B, T, H, N)

    H_arr = mx.array([H], dtype=mx.uint32)
    T_arr = mx.array([T], dtype=mx.uint32)

    kernel = _get_wkv7_fwd_kernel(N)
    outputs = kernel(
        inputs=[r_flat, w_flat, k_flat, v_flat, a_flat, b_flat, H_arr, T_arr],
        template=[("T", r.dtype)],
        grid=(B * H * N, 1, 1),
        threadgroup=(N, 1, 1),
        output_shapes=[(B, T, H, N), (B * H * n_chunks * N * N,), (B, T, H, N)],
        output_dtypes=[r.dtype, mx.float32, mx.float32],
    )
    return outputs[0].reshape(B, T, C), outputs[1], outputs[2]


def wkv7_metal_backward(r, w, k, v, a, b, dy, s, sa, head_size=64):
    """WKV-7 backward pass using saved states.

    Args:
        r, w, k, v, a, b: original inputs (B, T, C)
        dy: gradient of output (B, T, C)
        s: saved states from forward
        sa: saved sa values from forward

    Returns:
        dr, dw, dk, dv, da, db: gradients for each input
    """
    B, T, C = r.shape
    H = C // head_size
    N = head_size

    r_flat = r.reshape(B, T, H, N)
    w_flat = w.reshape(B, T, H, N)
    k_flat = k.reshape(B, T, H, N)
    v_flat = v.reshape(B, T, H, N)
    a_flat = a.reshape(B, T, H, N)
    b_flat = b.reshape(B, T, H, N)
    dy_flat = dy.reshape(B, T, H, N)

    H_arr = mx.array([H], dtype=mx.uint32)
    T_arr = mx.array([T], dtype=mx.uint32)

    kernel = _get_wkv7_bwd_kernel(N)
    outputs = kernel(
        inputs=[r_flat, w_flat, k_flat, v_flat, a_flat, b_flat,
                dy_flat, s, sa, H_arr, T_arr],
        template=[("T", r.dtype)],
        grid=(B * H * N, 1, 1),
        threadgroup=(N, 1, 1),
        output_shapes=[(B, T, H, N)] * 6,
        output_dtypes=[r.dtype] * 6,
    )
    return tuple(o.reshape(B, T, C) for o in outputs)


# Legacy forward-only API (for inference)
def wkv7_metal(r, w, k, v, a, b, head_size=64):
    """Compute WKV-7 using Metal kernel (forward only, no grad support)."""
    y, _, _ = wkv7_metal_forward(r, w, k, v, a, b, head_size)
    return y
