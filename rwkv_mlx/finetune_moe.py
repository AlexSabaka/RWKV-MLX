"""
MoE Fine-tuning: inject LoRA expert adapters into a frozen RWKV-v7 base.

Workflow:
  1. Load a pretrained base checkpoint (e.g. out-try-4/rwkv-71000.safetensors)
  2. Build a new model with MoE config — base ChannelMix weights are copied
     and frozen via mx.stop_gradient; only expert adapters + routers train.
  3. Train with: CE loss + L2wrap + load-balance aux + z-loss + ortho loss
  4. Save: only the expert/router delta weights (can reload with build_moe_from_base)

Quick start (LoRA experts, layers 6-11, 4 experts, rank 64):

  python -m rwkv_mlx.finetune_moe \\
      --base_model models/out-try-4 \\
      --resume models/out-try-4/rwkv-71000.safetensors \\
      --data data/test_full.bin \\
      --output models/moe-try-1 \\
      --n_experts 4 --lora_rank 64 --top_k 2 \\
      --moe_layers 6,7,8,9,10,11 \\
      --total_steps 10000 --lr_init 1e-4 --lr_final 1e-5

Memory: frozen base ~370MB + 4 experts × 6 layers × ~400KB adapters ≈ +10MB trainable.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np
from tqdm import tqdm

from .model import RWKV, RWKVConfig
from .train import TextDataset, l2_wrap_loss


# ---------------------------------------------------------------------------
# Weight injection: copy base FFN weights into MoE blocks
# ---------------------------------------------------------------------------

def build_moe_from_base(base_model: RWKV, moe_config: RWKVConfig) -> RWKV:
    """Create an MoE model and populate its base FFN weights from a loaded model.

    The base_model must be loaded (e.g. via model.load_weights) BEFORE calling
    this. Its FFN weights are copied into each MoE block's ffn.base submodule.
    All other weights (emb, head, ln_out, att, ln0/1/2) are copied directly.

    Expert adapter matrices (lora_A, lora_B) and router.gate are freshly
    initialised — they start with zero delta (base output is preserved on step 0).

    Args:
        base_model: RWKV model loaded from checkpoint (no MoE config)
        moe_config: RWKVConfig with moe_num_experts > 0

    Returns:
        moe_model: RWKV model with MoE blocks, ready to fine-tune
    """
    # Build the new model structure
    moe_model = RWKV(moe_config)

    # Flatten both models' params for name-based copying
    base_params = dict(tree_flatten(base_model.parameters()))
    moe_params  = dict(tree_flatten(moe_model.parameters()))

    updates = {}
    for moe_name in moe_params:
        # Translate MoE param name → base param name
        # MoE layers: blocks.i.ffn.base.key.weight → blocks.i.ffn.key.weight
        base_name = moe_name.replace(".ffn.base.", ".ffn.")

        if base_name in base_params:
            updates[moe_name] = base_params[base_name]
        # Expert adapters (lora_A, lora_B, router.gate) are NOT in base_params;
        # they keep their fresh initialisation from RWKV(moe_config).__init__.

    # Reconstruct nested dict and apply
    from mlx.utils import tree_unflatten
    moe_model.update(tree_unflatten(list(updates.items())))
    mx.eval(moe_model.parameters())
    return moe_model


def save_expert_weights(model: RWKV, output_path: str):
    """Save only the MoE adapter + router delta weights.

    Saves: lora_A, lora_B, router.gate.weight (the MoE-specific params).
    Excludes: base FFN weights, TimeMix, embeddings, head, layer norms.
    The base model is loaded separately from the original checkpoint.
    This keeps the delta file tiny (~10MB vs 370MB for LoRA experts).
    """
    expert_weights = {}
    for name, param in tree_flatten(model.parameters()):
        # Only keep the MoE adapter params: lora_A, lora_B, router weights
        is_moe_adapter = (
            ".ffn.lora_A" in name
            or ".ffn.lora_B" in name
            or ".ffn.router." in name
            or ".ffn.x_k" in name  # per-expert x_k in MoE_CMix (full mode)
        )
        if is_moe_adapter:
            expert_weights[name] = param

    mx.save_safetensors(output_path, expert_weights)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def finetune_moe(
    base_checkpoint: str,
    moe_config: RWKVConfig,
    data_path: str,
    output_dir: str,
    lr_init: float = 1e-4,
    lr_final: float = 1e-5,
    warmup_steps: int = 50,
    micro_bsz: int = 2,
    total_steps: int = 10000,
    save_every: int = 1000,
    log_every: int = 10,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
):
    os.makedirs(output_dir, exist_ok=True)

    E = moe_config.moe_num_experts
    r = moe_config.moe_lora_rank
    k = moe_config.moe_top_k
    mode = moe_config.moe_mode
    layers = list(moe_config.moe_layers) if moe_config.moe_layers else list(range(moe_config.n_layer))

    print(f"RWKV-v7 MoE Fine-tuning")
    print(f"  Base: L{moe_config.n_layer} D{moe_config.n_embd}")
    print(f"  Mode: {mode}, {E} experts, top-{k}, rank={r if mode=='lora' else 'N/A'}")
    print(f"  MoE layers: {layers}")
    print(f"  Checkpoint: {base_checkpoint}")

    # 1. Load base model (no MoE) to get the pretrained FFN weights
    base_config = RWKVConfig(
        n_layer=moe_config.n_layer,
        n_embd=moe_config.n_embd,
        vocab_size=moe_config.vocab_size,
        head_size=moe_config.head_size,
        ctx_len=moe_config.ctx_len,
        d_decay_lora=moe_config.d_decay_lora,
        d_aaa_lora=moe_config.d_aaa_lora,
        d_mv_lora=moe_config.d_mv_lora,
        d_gate_lora=moe_config.d_gate_lora,
    )
    base_model = RWKV(base_config)
    print(f"  Loading base weights from {base_checkpoint}...")
    base_model.load_weights(base_checkpoint)
    mx.eval(base_model.parameters())

    # 2. Build MoE model, copy base weights into it
    print("  Building MoE model...")
    model = build_moe_from_base(base_model, moe_config)
    del base_model  # free memory

    # Count trainable vs frozen params
    total_params  = sum(p.size for _, p in tree_flatten(model.parameters()))
    frozen_params = sum(
        p.size for name, p in tree_flatten(model.parameters())
        if ".ffn.base." in name
    )
    trainable_params = total_params - frozen_params

    print(f"  Total params:     {total_params:,}")
    print(f"  Frozen (base):    {frozen_params:,}")
    print(f"  Trainable (MoE):  {trainable_params:,}")

    est_train_gb = trainable_params * 4 * 3 / 1e9  # param + m + v in fp32
    est_base_gb  = frozen_params * 2 / 1e9          # frozen in fp16
    print(f"  Est. memory: base={est_base_gb:.2f}GB frozen + {est_train_gb:.2f}GB optimizer")

    # 3. Data
    dataset = TextDataset(data_path, moe_config.ctx_len)

    # 4. Optimizer — only over trainable (non-frozen) parameters
    # We use stop_gradient in the forward pass, so all gradients for
    # ffn.base.* are already zero. We still pass all params to the
    # optimizer but frozen ones receive zero gradient so won't move.
    warmup  = optim.linear_schedule(lr_init * 0.01, lr_init, warmup_steps)
    decay   = optim.cosine_decay(lr_init, total_steps - warmup_steps, lr_final)
    lr_sched = optim.join_schedules([warmup, decay], [warmup_steps])
    optimizer = optim.AdamW(
        learning_rate=lr_sched,
        betas=[0.9, 0.99],
        eps=1e-8,
        weight_decay=0.0,
    )

    # Weight decay mask: only 2D+ non-frozen weights
    wd_mask = tree_map(
        lambda p: mx.ones_like(p) if p.ndim >= 2 else mx.zeros_like(p),
        model.parameters()
    )
    # Zero WD for frozen base FFN weights (they're frozen anyway, but be explicit)
    for name, _ in tree_flatten(model.parameters()):
        if ".ffn.base." in name:
            parts = name.split(".")
            d = wd_mask
            for p in parts[:-1]:
                d = d[int(p)] if p.isdigit() else d[p]
            d[parts[-1]] = mx.zeros(())

    # 5. Loss and gradient
    def loss_fn(model, x, y):
        logits, aux_loss = model(x)
        return l2_wrap_loss(logits, y) + aux_loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # 6. Training loop (no mx.compile — MoE graph has Python-level loops)
    print(f"\nStarting MoE fine-tuning ({total_steps} steps)...\n")
    start_time = time.time()
    running_loss = 0.0
    last_log_time = start_time
    step_tokens = micro_bsz * moe_config.ctx_len

    with tqdm(range(total_steps + 1), total=total_steps, desc="MoE fine-tune", unit="step") as pbar:
        for step in pbar:
            x, y = dataset.get_batch(micro_bsz)
            loss, grads = loss_and_grad_fn(model, x, y)
            mx.eval(loss, grads)

            # Gradient clipping
            grad_norm_sq = sum((g * g).sum() for _, g in tree_flatten(grads))
            grad_norm = mx.sqrt(grad_norm_sq)
            clip = mx.minimum(mx.array(1.0), mx.array(grad_clip) / (grad_norm + 1e-6))
            grads = tree_map(lambda g: g * clip, grads)

            optimizer.update(model, grads)

            # Manual weight decay on 2D+ trainable params
            cur_lr = lr_sched(optimizer.state["step"])
            new_params = tree_map(
                lambda p, m: p - p * m * (weight_decay * cur_lr),
                model.parameters(), wd_mask
            )
            model.update(new_params)
            mx.eval(model.state, optimizer.state, grad_norm)

            running_loss += loss.item()

            if step % log_every == 0 and step > 0:
                avg_loss = running_loss / log_every
                now = time.time()
                kt_per_sec = log_every * step_tokens / (now - last_log_time) / 1000
                ppl = math.exp(min(avg_loss, 20))
                cur_lr_val = cur_lr.item() if hasattr(cur_lr, "item") else float(cur_lr)
                pbar.set_postfix({
                    "loss": f"{avg_loss:.3f}",
                    "ppl":  f"{ppl:.2f}",
                    "lr":   f"{cur_lr_val:.2e}",
                    "gn":   f"{grad_norm.item():.2f}",
                    "Kt/s": f"{kt_per_sec:.1f}",
                })
                running_loss = 0.0
                last_log_time = now

            if step > 0 and (step % save_every == 0 or step == total_steps):
                # Save full model (base + experts) for easy loading
                ckpt_path = os.path.join(output_dir, f"moe-{step}.safetensors")
                flat_params = dict(tree_flatten(model.parameters()))
                mx.save_safetensors(ckpt_path, flat_params)

                # Also save lightweight delta-only file
                delta_path = os.path.join(output_dir, f"moe-delta-{step}.safetensors")
                save_expert_weights(model, delta_path)

                pbar.write(f"Saved: {ckpt_path} + delta {delta_path}")

                with open(os.path.join(output_dir, "config.json"), "w") as f:
                    json.dump({
                        "n_layer": moe_config.n_layer,
                        "n_embd":  moe_config.n_embd,
                        "vocab_size": moe_config.vocab_size,
                        "head_size": moe_config.head_size,
                        "ctx_len": moe_config.ctx_len,
                        "moe_num_experts": moe_config.moe_num_experts,
                        "moe_top_k": moe_config.moe_top_k,
                        "moe_mode": moe_config.moe_mode,
                        "moe_lora_rank": moe_config.moe_lora_rank,
                        "moe_layers": list(moe_config.moe_layers),
                    }, f, indent=2)
                with open(os.path.join(output_dir, "training_state.json"), "w") as f:
                    json.dump({"step": step}, f)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MoE fine-tuning for RWKV-v7 MLX")

    # Base model
    parser.add_argument("--base_model", type=str, required=True,
                        help="Directory with base model config.json")
    parser.add_argument("--resume", type=str, required=True,
                        help="Base checkpoint .safetensors to inject experts into")

    # Data & output
    parser.add_argument("--data",   type=str, required=True,  help="Training data .bin")
    parser.add_argument("--output", type=str, required=True,  help="Output directory")

    # MoE config
    parser.add_argument("--n_experts",  type=int, default=4,    help="Number of experts")
    parser.add_argument("--lora_rank",  type=int, default=64,   help="LoRA adapter rank (mode=lora)")
    parser.add_argument("--top_k",      type=int, default=2,    help="Experts per token")
    parser.add_argument("--moe_mode",   type=str, default="lora", choices=["lora", "full"])
    parser.add_argument("--moe_layers", type=str, default="",
                        help="Comma-separated layer indices to add MoE (default: all)")
    parser.add_argument("--aux_coeff",  type=float, default=0.01,  help="Load-balance loss weight")
    parser.add_argument("--z_coeff",    type=float, default=0.001, help="Z-loss weight")
    parser.add_argument("--ortho_coeff",type=float, default=0.001, help="Orthogonality loss weight")

    # Training
    parser.add_argument("--micro_bsz",    type=int,   default=2,     help="Micro batch size")
    parser.add_argument("--lr_init",      type=float, default=1e-4,  help="Initial LR")
    parser.add_argument("--lr_final",     type=float, default=1e-5,  help="Final LR")
    parser.add_argument("--warmup_steps", type=int,   default=50)
    parser.add_argument("--total_steps",  type=int,   default=10000)
    parser.add_argument("--save_every",   type=int,   default=1000)
    parser.add_argument("--log_every",    type=int,   default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip",    type=float, default=1.0)

    args = parser.parse_args()

    # Load base config
    config_path = os.path.join(args.base_model, "config.json")
    with open(config_path) as f:
        base_cfg = json.load(f)

    moe_layers = tuple(int(x) for x in args.moe_layers.split(",") if x.strip()) \
                 if args.moe_layers else ()

    moe_config = RWKVConfig(
        n_layer=base_cfg["n_layer"],
        n_embd=base_cfg["n_embd"],
        vocab_size=base_cfg.get("vocab_size", 65568),
        head_size=base_cfg.get("head_size", 64),
        ctx_len=base_cfg.get("ctx_len", 512),
        moe_num_experts=args.n_experts,
        moe_top_k=args.top_k,
        moe_mode=args.moe_mode,
        moe_lora_rank=args.lora_rank,
        moe_layers=moe_layers,
        moe_aux_loss_coeff=args.aux_coeff,
        moe_z_loss_coeff=args.z_coeff,
        moe_ortho_coeff=args.ortho_coeff,
    )

    finetune_moe(
        base_checkpoint=args.resume,
        moe_config=moe_config,
        data_path=args.data,
        output_dir=args.output,
        lr_init=args.lr_init,
        lr_final=args.lr_final,
        warmup_steps=args.warmup_steps,
        micro_bsz=args.micro_bsz,
        total_steps=args.total_steps,
        save_every=args.save_every,
        log_every=args.log_every,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )


if __name__ == "__main__":
    main()
