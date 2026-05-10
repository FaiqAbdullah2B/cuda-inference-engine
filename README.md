# cuda-inference-engine

> **CE323 — General Purpose Programming with GPU (CUDA)**  
> Forward pass implementation of [Karpathy's llm.c](https://github.com/karpathy/llm.c) — from scratch in C and CUDA.

---

## Overview

This project implements the **GPT-2 (124M) forward pass** in three stages: a reference CPU implementation in plain C, a naive GPU port in CUDA, and an optimized CUDA version with memory and compute improvements. The goal is to understand what actually happens inside a transformer at the kernel level, not just in PyTorch.

The math and tensor shapes are explored in a companion Python notebook, built while following Karpathy's GPT-2 video walkthrough.

---

## Architecture

**Model:** GPT-2 124M  
**Dataset:** TinyShakespeare  

```
Input Tokens
    │
    ├── Token Embeddings  (vocab_size × n_embd)
    └── Positional Embeddings  (seq_len × n_embd)
         │
         └── [×12 Transformer Blocks]
              │
              ├── LayerNorm
              ├── QKV Projection  (matmul)
              ├── Causal Self-Attention
              ├── Residual Add
              ├── LayerNorm
              ├── MLP  (Linear → GELU → Linear)
              └── Residual Add
                   │
                   └── Final LayerNorm → Logits
```

## Implementation Progress

### The Real Work

Before a single line of C was written, this project required roughly *15-20+ hours* of groundwork:
reading the GPT-2 paper, tracing Karpathy's llm.c line by line, watching the build video, and
manually tracking tensor shapes through every layer until the architecture actually made sense.

### CPU (C)

| Component | Status |
|---|---|
| Tokenizer (`.bin` loader) | Done |
| Dataloader | Done |
| GPT-2 structs + weight loading | Done |
| Encoder (token emb + positional emb) | Done |
| LayerNorm forward | Done |
| QKV matmul | Done |
| Masked multi-headed attention | Done |
| GELU activation | Done |
| Residual connections | Done |
| Full forward pass on TinyShakespeare | Done |

**Batch config:** 4 batches × 64 tokens

---

### GPU — Naive CUDA

| Component | Status |
|---|---|
| Encoder kernel | Done |
| LayerNorm kernel | Done |
| MatMul kernel | Done |
| Attention kernel | Done |
| GELU kernel | Done |
| Softmax kernel | Done |
| Full forward pass on TinyShakespeare | Done |

**Batch config:** 1 batches x 128 tokens

**Block Dim:** Hardcoded to 512 threads per block

---

### GPU — Optimized CUDA

| Component | Status |
|---|---|
| Tiled matmul (shared memory) | Planned |
| Fused LayerNorm | Planned |
| Attention with reduced global mem reads | Planned |
| Full pass + benchmarks vs naive | Planned |

---

## Running the Code

### Prerequisites

- CUDA Toolkit 12.x
- GCC / Clang
- Python 3.x + Jupyter (for the notebook)
- GPT-2 124M weights (download from `./dev/download_starter_pack.sh` by karpathy)


## Concepts Covered

**From scratch:**
- Transformer block anatomy at the CUDA kernel level
- Byte-pair encoding via pre-tokenized binary files
- Layer normalization: mean, variance, and the gamma/beta affine transform
- QKV projection and scaled dot-product attention
- GELU approximation used in GPT-2
- GPU memory hierarchy: global, shared, registers
- Thread block tiling for matmul

**From the notebook:**
- Tensor shape tracking through every layer
- Attention pattern visualization

---

## References

- [Karpathy — llm.c](https://github.com/karpathy/llm.c)
- [Karpathy — Let's build GPT-2 (video)](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2 paper)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

## Course Info

**Course:** CE323 — General Purpose Programming with GPU  
**Institute:** Ghulam Ishaq Khan Institute of Engineering Sciences and Technology (GIKI)  

---
