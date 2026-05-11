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
| WMMA matmul (Tensor Core calls) |  Done |
| Attention (running softmax, shared mem + reduction) |  Done |
| Texture cache for cached reads (LayerNorm & Attention) |  Done |
| Full pass + benchmarks vs naive |  Done |
---

---

### Benchmarks

> `B = 1, C = 88, Words Gen = 24` (GPU runs); `B = 1, C = 768, Words Gen = 10` (CPU run)

| Metric | CPU | Naive GPU | Optimized GPU |
|---|---|---|---|
| Total execution time | 22522.83 ms | 7322.66 ms | 330.08 ms |
| Total compute time | 22519.08 ms | 7320.25 ms | 327.71 ms |
| Avg time / word | 2252.28 ms | 305.11 ms | **13.75 ms** |
| Avg time / forward pass | 2251.91 ms | 305.01 ms | **13.65 ms** |
| Activations memory | 6 MB | 98 MB | 98 MB |
| Half-precision memory | — | — | 78 MB |

**Optimized GPU is ~22× faster than Naive GPU, and ~164× faster than CPU** *(note: CPU run used C = 768 vs C = 88 for GPU runs)*

<details>
<summary>Raw benchmark output</summary>

```
NAIVE GPU
B = 1, T = 64 to 768, C = 88, Words Gen = 24
Activations memory used: 98 MB
Total execution time:          7322.656840 ms
Allocation time overhead:         0.288876 ms
Total GPU execution time:      7320.252472 ms
Average time per word generation:  305.110702 ms
Average time per GPU forward pass: 305.010520 ms

OPTIMIZED GPU
B = 1, T = 64 to 768, C = 88, Words Gen = 24
Activations memory used: 98 MB
Halfs memory used:       78 MB
Total execution time:          330.084411 ms
Allocation time overhead:        0.259933 ms
Total GPU execution time:      327.711839 ms
Average time per word generation:  13.753517 ms
Average time per GPU forward pass: 13.654660 ms

CPU
B = 1, T = 8, C = 768, Words Gen = 10
Activations memory used: 6 MB
Total execution time:        22522.83 ms
Allocation time overhead:        0.02 ms
Total CPU execution time:    22519.08 ms
Average time per word generation:  2252.28 ms
Average time per forward pass:     2251.91 ms
```

</details>
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
