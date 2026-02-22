# RWKV-MLX: RWKV-v7 for Apple Silicon

A complete reimplementation of the **RWKV-v7 "Goose"** language model architecture using [Apple MLX](https://github.com/ml-explore/mlx), designed to fully utilize Metal GPU on Apple Silicon Macs.

Supports training from scratch, converting pretrained PyTorch weights, and both RNN-mode and GPT-mode inference — all natively on Metal with custom WKV-7 kernels.

> **Motivation**: This project is a platform for small language model research — exploring whether architectural innovations (mixture of experts, neuroevolution, polynomial activations) can push 10-50M parameter models toward capabilities that normally require scale.

## Status

**Training run active** (out-try-4): L12 D768 (184M params), training on OpenWebText2 (~100M tokens).

- ~1.6 Kt/s throughput on Mac mini M4
- At 71k steps (loss≈3.8, ppl≈45): coherent English text, checkpoint at `models/out-try-4/rwkv-71000.safetensors`

Sample output at 1k steps (temperature=1.0, top_p=0.9):

```text
> "Perfect temperature for"
Perfect temperature for over-passful resources.
Despite the developedization of the combatants to assure the public
policy of the stage in question around the world," said Hobbes said.
```

## Target Platform

- **Mac mini M4** (16GB unified memory, ~11GB accessible as VRAM)
- Native **Metal** acceleration via MLX (no CUDA dependency)
- **float16** precision for optimal Metal performance

## Features

- Full RWKV-v7 (x070) architecture: TimeMix with WKV-7 state update, ChannelMix with squared ReLU
- Custom Metal kernel for WKV-7 forward + backward pass (1100x faster than MLX autograd)
- Training from scratch with orthogonal initialization matching the reference implementation
- Selective weight decay (2D+ matrices only) and 2x LR for decay bias (`w0`)
- L2Wrap logit regularization matching reference backward pass
- RNN-mode inference for efficient autoregressive generation (~32-34 tok/s)
- GPT-mode inference for parallel prefill (~22-23 tok/s)
- PyTorch weight conversion (load pretrained RWKV-7 models)
- RWKV World tokenizer with 10 special tokens for chat/tool-use (65568 total vocab)
- Data preprocessing pipeline for JSONL/JSONL.ZST datasets
- **Curriculum learning**: rank documents by perplexity, train in easy-first or hard-first order (`--sequential`)
- **Supervised fine-tuning**: masked CE loss on output tokens, JSONL input/output pairs (`--sft_data`)
- **LoRA-MoE**: sparse expert adapters injected into ChannelMix, with load-balancing and orthogonality regularization

## Installation

```bash
cd RWKV-MLX
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Dependencies: `mlx>=0.22.0`, `numpy`, `safetensors`, `tqdm`

## Quick Start

### 1. Preprocess training data

Convert JSONL or JSONL.ZST datasets into the binary token format:

```bash
# Process a directory of .jsonl.zst files (e.g., OpenWebText2)
python scripts/data_preprocess.py /path/to/openwebtext2/ \
    -o data/owt2.bin \
    --min-words 50

# Process a single .jsonl file
python scripts/data_preprocess.py data/my_corpus.jsonl -o data/train.bin

# Download TinyStories from HuggingFace (~500MB tokenized)
python scripts/download_tinystories.py --output data/
```

Output: uint32 binary file with documents separated by `<|eos|>` tokens (ID 65536).

### 2. Train from scratch

```bash
python -m rwkv_mlx.train \
    --data data/owt2.bin \
    --output models/my-run \
    --n_layer 12 --n_embd 768 \
    --ctx_len 512 \
    --micro_bsz 2 \
    --grad_accum 4 \
    --lr_init 6e-4 --lr_final 6e-5 \
    --warmup_steps 100 \
    --total_steps 50000 \
    --save_every 5000

# Resume from checkpoint
python -m rwkv_mlx.train \
    --data data/owt2.bin \
    --output models/my-run \
    --resume models/my-run/rwkv-10000.safetensors \
    --total_steps 50000

# Curriculum learning — train on ranked data in order
python -m rwkv_mlx.train \
    --data data/tinystories_asc.bin \
    --output models/ts-curriculum \
    --n_layer 8 --n_embd 512 \
    --sequential \
    --total_steps 20000

# Supervised fine-tuning (loss on output tokens only)
python -m rwkv_mlx.train \
    --data data/owt2.bin \
    --sft_data data/my_sft.jsonl \
    --output models/my-sft \
    --resume models/my-run/rwkv-50000.safetensors \
    --lr_init 1e-4 --lr_final 1e-5
```

### 3. Generate text

```bash
# RNN mode (faster for generation)
python -m rwkv_mlx.generate \
    --model models/my-run \
    --prompt "The Eiffel tower is in the city of" \
    --max_tokens 200 \
    --temperature 1.0 --top_p 0.9 \
    --mode rnn

# GPT mode (parallel prefill)
python -m rwkv_mlx.generate \
    --model models/my-run \
    --prompt "Once upon a time" \
    --mode gpt
```

### 4. Convert pretrained PyTorch weights

```bash
# Download from https://huggingface.co/BlinkDL/rwkv-7-world
python -m rwkv_mlx.convert \
    path/to/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth \
    -o models/converted/
```

## Curriculum Learning

Rank training documents by perplexity under a trained model, then train a new model in easy-first or hard-first order.

Based on ["Influence-driven Curriculum Learning for Pre-training on Limited Data"](https://arxiv.org/abs/2508.15475) (Schoenegger et al., 2025).

```bash
# 1. Score TinyStories documents, sort easy-first
python scripts/rank_dataset.py \
    --model models/out-try-4 \
    --checkpoint models/out-try-4/rwkv-71000.safetensors \
    --data data/tinystories_train.bin \
    --output data/tinystories_asc.bin \
    --save_scores data/tinystories_scores.npy \
    --direction asc

# 2. MATTR-only ranking (no model needed — vocabulary diversity heuristic)
python scripts/rank_dataset.py \
    --data data/tinystories_train.bin \
    --mattr_only \
    --output data/tinystories_mattr.bin

# 3. Train with curriculum (--sequential reads batches in order)
python -m rwkv_mlx.train \
    --data data/tinystories_asc.bin \
    --output models/ts-curriculum-asc \
    --n_layer 8 --n_embd 512 \
    --sequential \
    --total_steps 20000 --lr_init 3e-4 --lr_final 3e-5
```

## Supervised Fine-Tuning

Pass a JSONL file with `{"input": "...", "output": "..."}` records. Loss is computed only on output tokens.

```bash
python -m rwkv_mlx.train \
    --data data/owt2.bin \
    --sft_data data/instructions.jsonl \
    --output models/my-sft \
    --resume models/out-try-4/rwkv-71000.safetensors \
    --lr_init 1e-4 --lr_final 1e-5 \
    --total_steps 5000
```

## LoRA-MoE Fine-Tuning

Inject sparse expert adapters (LoRA-style) into ChannelMix layers of a pretrained base model.

```bash
python -m rwkv_mlx.finetune_moe \
    --base_model models/out-try-4 \
    --resume models/out-try-4/rwkv-71000.safetensors \
    --data data/tinystories_train.bin \
    --output models/moe-try-1 \
    --n_experts 4 --lora_rank 64 --top_k 2 \
    --moe_layers 6,7,8,9,10,11 \
    --total_steps 10000 --lr_init 1e-4 --lr_final 1e-5
```

The base model is frozen. Only LoRA adapters (~9MB delta file) and router weights are saved.

## Architecture

```text
Input tokens
  -> Embedding (65568 vocab)
    -> LayerNorm (block 0 only)
      -> N x Block:
        |-- LayerNorm -> TimeMix (WKV-7 attention) -> residual add
        +-- LayerNorm -> ChannelMix (squared ReLU FFN) -> residual add
      -> LayerNorm
        -> Linear head
          -> Logits
```

### Key Design Decisions

1. **Custom Metal WKV kernel**: The core WKV-7 recurrent state update is implemented as a fused Metal kernel with both forward and backward passes, using `mx.custom_function` with VJP. This is 1100x faster than letting MLX autograd trace through 512+ sequential operations.

2. **CHUNK_LEN=16 backward**: The backward kernel uses state checkpointing every 16 timesteps (ported from the CUDA reference). This bounds gradient approximation error to <0.1% while using only ~12MB extra memory.

3. **float16 training**: Metal has excellent float16 support. The decay bias (`w0`) is kept in float32 for numerical stability.

4. **Orthogonal initialization**: Weight matrices use orthogonal init matching the reference RWKV-v7 implementation — critical for training stability.

## Training Details

### Initialization

Matches the reference `RWKV-v7/train_temp/src/model.py`:

- Embedding: uniform(-1e-4, 1e-4)
- receptance, value: orthogonal, gain=1.0
- key: orthogonal, gain=0.1
- output, ffn.value: zeros
- ffn.key: orthogonal, gain=1.0
- LoRA W1: zeros; W2: orthogonal, scale=0.1
- Head: orthogonal, scale=0.5*sqrt(V/C)

### Optimizer

- AdamW with betas=(0.9, 0.99), eps=1e-18
- Weight decay applied manually to 2D+ weight matrices only (NOT to 1D params like mixing coefficients, decay biases, or normalization weights)
- `att.w0` parameters receive 2x effective learning rate via gradient scaling
- Cosine LR decay with linear warmup

### Special Tokens

10 special tokens at IDs 65536-65545 for chat templates and tool use:

| Token | ID |
| --- | --- |
| `<\|eos\|>` | 65536 |
| `<\|pad\|>` | 65537 |
| `<\|im_start\|>` | 65538 |
| `<\|im_end\|>` | 65539 |
| `<\|think\|>` / `<\|/think\|>` | 65540-65541 |
| `<\|tool_call\|>` / `<\|/tool_call\|>` | 65542-65543 |
| `<\|tool_response\|>` / `<\|/tool_response\|>` | 65544-65545 |

Vocab padded to 65568 for 32-byte GPU alignment.

## Memory Estimates

| Model Size | Params | Model (fp16) | Optimizer | Activations* | Total |
| --- | --- | --- | --- | --- | --- |
| 0.1B (L12-D768) | ~184M | 0.35 GB | 1.40 GB | ~0.5 GB | ~2.3 GB |
| 0.4B (L24-D1024) | ~420M | 0.80 GB | 3.20 GB | ~1.0 GB | ~5.0 GB |
| 1.5B (L24-D2048) | ~1.5B | 2.86 GB | 11.4 GB | N/A | > 11 GB |

*Activations for micro_bsz=2, ctx_len=512. 1.5B exceeds M4 16GB and would require gradient checkpointing.

## Performance (L12 D768, Mac mini M4)

| Metric | Value |
| --- | --- |
| Training throughput | 1.3-1.6 Kt/s (B=2, T=512) |
| Metal WKV fwd+bwd per layer | ~7ms |
| RNN inference | ~32-34 tok/s |
| GPT inference | ~22-23 tok/s |
| Training step time | ~650ms |

## File Structure

```text
RWKV-MLX/
├── rwkv_mlx/
│   ├── __init__.py
│   ├── model.py           # Core RWKV-v7 model (TimeMix, ChannelMix, Block, RWKV)
│   ├── train.py           # Training pipeline (CLM, sequential, SFT)
│   ├── generate.py        # Inference & text generation (RNN + GPT mode)
│   ├── convert.py         # PyTorch -> MLX weight conversion
│   ├── tokenizer.py       # RWKV World tokenizer + special tokens
│   ├── moe.py             # LoRA-MoE channel-mix expert adapters
│   ├── finetune_moe.py    # MoE fine-tuning loop
│   └── kernels/
│       ├── __init__.py
│       └── wkv7_metal.py  # Custom Metal forward+backward kernels
├── scripts/
│   ├── data_preprocess.py      # JSONL/JSONL.ZST -> binary token file
│   ├── download_tinystories.py # Download TinyStories from HuggingFace
│   └── rank_dataset.py         # Score & sort .bin by perplexity/MATTR
├── docs/
│   └── plan.md            # Research plan (Phases 1-6)
├── rwkv_vocab_v20230424.txt # RWKV World tokenizer vocabulary
├── pyproject.toml
└── README.md
```

## Comparison with Original

| Feature | Original (CUDA) | RWKV-MLX (Metal) |
| --- | --- | --- |
| Framework | PyTorch + DeepSpeed | MLX |
| GPU | NVIDIA CUDA | Apple Metal |
| Precision | bf16/fp16 | fp16 |
| WKV kernel | CUDA custom kernel | Metal custom kernel |
| Training | PyTorch Lightning | Native MLX loop with mx.compile |
| Data format | binidx (mmap) | uint32 binary (.bin) |
| Distributed | DeepSpeed multi-GPU | Single GPU (unified memory) |
| Vocab size | 65536 | 65568 (+ special tokens) |

## Known Limitations

- **Single GPU only**: No distributed training (Apple Silicon unified memory)
- **1.5B+ models don't fit**: L24-D2048 exceeds M4 16GB without gradient checkpointing (and `mx.checkpoint` makes RWKV *slower* due to recurrent recomputation cost)
- **Pretrained small models are weak**: Converted 0.1B and 0.4B pretrained weights generate correctly but produce low-quality text — the smallest capable RWKV-7 model in the official family is ~2.9B
- **No bfloat16**: Metal has limited bf16 support; training uses float16 with float32 for numerically sensitive params (`w0`)

## References

- [RWKV-LM Repository](https://github.com/BlinkDL/RWKV-LM)
- [RWKV Paper (v4)](https://arxiv.org/abs/2305.13048)
- [MLX Framework](https://github.com/ml-explore/mlx)
- [Pretrained Models](https://huggingface.co/BlinkDL/rwkv-7-world)
