"""
Convert PyTorch RWKV-v7 weights to MLX format.

Supports loading .pth files (PyTorch state dicts) and saving as
safetensors for efficient MLX loading.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def convert_pytorch_to_mlx(
    pth_path: str,
    output_dir: str,
    dtype: str = "float16",
) -> dict:
    """Convert a PyTorch RWKV-v7 .pth checkpoint to MLX safetensors.

    Args:
        pth_path: path to the .pth file
        output_dir: directory to save the MLX weights
        dtype: target dtype ("float16" or "float32")

    Returns:
        config dict inferred from the weights
    """
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required for weight conversion. Install with: pip install torch")

    print(f"Loading {pth_path}...")
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)

    # Infer config from weights
    n_embd = state_dict["emb.weight"].shape[1]
    vocab_size = state_dict["emb.weight"].shape[0]
    n_layer = 0
    for key in state_dict:
        if key.startswith("blocks.") and key.endswith(".ln1.weight"):
            layer_id = int(key.split(".")[1])
            n_layer = max(n_layer, layer_id + 1)

    head_size = state_dict["blocks.0.att.r_k"].shape[1] if "blocks.0.att.r_k" in state_dict else 64

    # Detect dim_ffn from FFN key weight shape
    dim_ffn = 0
    if "blocks.0.ffn.key.weight" in state_dict:
        dim_ffn = state_dict["blocks.0.ffn.key.weight"].shape[0]

    # Detect LoRA dimensions from weight shapes
    d_decay_lora = state_dict["blocks.0.att.w1"].shape[1] if "blocks.0.att.w1" in state_dict else 0
    d_aaa_lora = state_dict["blocks.0.att.a1"].shape[1] if "blocks.0.att.a1" in state_dict else 0
    d_mv_lora = state_dict["blocks.1.att.v1"].shape[1] if "blocks.1.att.v1" in state_dict else 0
    d_gate_lora = state_dict["blocks.0.att.g1"].shape[1] if "blocks.0.att.g1" in state_dict else 0

    config = {
        "n_layer": n_layer,
        "n_embd": n_embd,
        "vocab_size": vocab_size,
        "head_size": head_size,
        "dim_ffn": dim_ffn,
        "d_decay_lora": d_decay_lora,
        "d_aaa_lora": d_aaa_lora,
        "d_mv_lora": d_mv_lora,
        "d_gate_lora": d_gate_lora,
    }

    print(f"Detected config: {config}")

    # Convert weights
    target_np_dtype = np.float16 if dtype == "float16" else np.float32
    mlx_weights = {}

    # Mapping from PyTorch names to MLX model parameter paths
    for name, tensor in state_dict.items():
        t = tensor.squeeze().float().numpy()

        # Determine which params need special handling
        if name.endswith("att.w0"):
            # w0 (decay bias) should remain float32 for numerical stability
            mlx_weights[name] = mx.array(t, dtype=mx.float32)
        elif "ln_x.weight" in name or "ln_x.bias" in name:
            # GroupNorm params: keep float32
            mlx_weights[name] = mx.array(t, dtype=mx.float32)
        else:
            mlx_weights[name] = mx.array(t.astype(target_np_dtype))

    # Map PyTorch state dict keys to MLX model structure
    mapped_weights = _map_weight_names(mlx_weights, config)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mx.save_safetensors(str(output_path / "model.safetensors"), mapped_weights)

    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved MLX weights to {output_path}")
    print(f"  - model.safetensors ({len(mapped_weights)} tensors)")
    print(f"  - config.json")

    return config


def _map_weight_names(weights: dict, config: dict) -> dict:
    """Map PyTorch state dict names to MLX model parameter paths.

    Handles:
    - Reshaping 1D params (x_r, x_w, w0, a0, v0, k_k, k_a) to (1,1,C)
    - Creating blocks.0.att.v0/v1/v2 (PyTorch doesn't save these for layer 0)
    """
    # 1D params that need reshaping to (1, 1, C) in MLX
    RESHAPE_1D = {"x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "w0", "a0", "v0", "k_k", "k_a"}

    mapped = {}
    for name, value in weights.items():
        # Reshape 1D mixing/bias params to (1, 1, C)
        param_name = name.split(".")[-1]
        if param_name in RESHAPE_1D and value.ndim == 1:
            value = value.reshape(1, 1, -1)

        mapped[name] = value

    # Layer 0 doesn't have v0/v1/v2 in PyTorch (they're unused, aliased to a0/a1/a2).
    # Create them as zeros so MLX model.load_weights() doesn't complain.
    n_embd = config.get("n_embd", 768)
    # Use detected LoRA dim from config if available, else fall back to formula
    D_MV = config.get("d_mv_lora", 0) or max(32, int(round((1.7 * (n_embd ** 0.5)) / 32) * 32))
    if "blocks.0.att.v0" not in mapped:
        mapped["blocks.0.att.v0"] = mx.zeros((1, 1, n_embd))
        mapped["blocks.0.att.v1"] = mx.zeros((n_embd, D_MV))
        mapped["blocks.0.att.v2"] = mx.zeros((D_MV, n_embd))

    return mapped


def load_mlx_weights(model, weight_dir: str):
    """Load MLX safetensors weights into an RWKV model.

    This handles the name mapping between flat safetensor keys
    and the nested MLX module structure.
    """
    weight_path = Path(weight_dir) / "model.safetensors"
    flat_weights = mx.load(str(weight_path))

    # Build the nested dict structure that MLX expects
    nested = {}
    for key, value in flat_weights.items():
        parts = key.split(".")
        d = nested
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    # Special handling: convert 'blocks' dict keys from strings to indices
    if "blocks" in nested:
        blocks_list = []
        n_blocks = len(nested["blocks"])
        for i in range(n_blocks):
            blocks_list.append(nested["blocks"][str(i)])
        nested["blocks"] = blocks_list

    model.load_weights(list(flat_weights.items()))
    return model


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch RWKV-v7 weights to MLX format")
    parser.add_argument("input", type=str, help="Path to .pth file")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: same dir as input)")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32"],
                        help="Target dtype")

    args = parser.parse_args()

    output_dir = args.output or str(Path(args.input).parent / "mlx_weights")
    convert_pytorch_to_mlx(args.input, output_dir, args.dtype)


if __name__ == "__main__":
    main()
