# ternary-transformer

[![crates.io](https://img.shields.io/crates/v/ternary-transformer.svg)](https://crates.io/crates/ternary-transformer)
[![docs.rs](https://docs.rs/ternary-transformer/badge.svg)](https://docs.rs/ternary-transformer)

**Ternary transformer components operating in ℤ₃ = {-1, 0, 1}.**

A research-grade Rust library implementing the full transformer architecture—attention, encoding, decoding, position encoding, feed-forward layers, and residual connections—where every weight, activation, and intermediate value is a **trit** (ternary digit) ∈ {-1, 0, 1}. All arithmetic uses explicit ℤ₃ modular addition and multiplication.

## Why Ternary Transformers?

Binary neural networks (BNNs) with weights in {-1, +1} have been extensively studied as a path to ultra-efficient inference. **Ternary neural networks** extend this by introducing a zero state, enabling:

- **Sparser activations** — The zero state acts as a natural "gating" mechanism
- **Better representational capacity** than binary (log₂3 ≈ 1.58× more information per parameter)
- **Hardware efficiency** — Ternary arithmetic maps well to integer ALUs and can be encoded in 2 bits
- **Theoretical elegance** — ℤ₃ is a finite field, enabling clean algebraic analysis

This library provides the building blocks to experiment with fully ternary transformers, where the entire computation graph operates in ℤ₃.

## Architecture Overview

```
Input → Position Encoding → Encoder Block(s) → Decoder Block(s) → Output
                                    │                  │
                         ┌──────────┴──────────┐       │
                         │  Self-Attention      │       │
                         │  (Q, K, V in ℤ₃)    │       │
                         │  + Residual          │       │
                         │  + FFN               │       │
                         └──────────────────────┘       │
                                              ┌─────────┴──────────┐
                                              │  Masked Self-Attn   │
                                              │  + Cross-Attention  │
                                              │  + Residual + FFN   │
                                              └─────────────────────┘
```

## Core Concepts

### ℤ₃ Arithmetic

All operations use explicit pattern matching over the 9 possible trit pairs:

| + | -1 | 0 | +1 |
|---|----|---|-----|
| **-1** | +1 | -1 | 0 |
| **0** | -1 | 0 | +1 |
| **+1** | 0 | +1 | -1 |

Note: 1 + 1 = 2 ≡ -1 (mod 3), and -1 + -1 = -2 ≡ +1 (mod 3). This is proper modular arithmetic, not saturation.

### Trit Type

```rust
pub enum Trit { NegOne, Zero, One }
```

Every value in the network is a `Trit`. Conversion from continuous values uses simple sign-based quantization.

## Components

### Ternary Self-Attention

Computes attention as: `output[i] = Σⱼ αᵢⱼ · Vⱼ` where attention weights αᵢⱼ are derived from ternary dot products of query and key vectors. No softmax normalization is needed—the ternary dot product naturally produces bounded results.

### Multi-Head Attention

Splits the feature dimension across `H` heads, applies attention independently per head, concatenates results, and projects through an output weight matrix. All projections are ternary matrix multiplications.

### Position Encoding

Ternary sinusoidal position encoding produces unique trit patterns for each position and dimension using modular arithmetic, analogous to the sinusoidal encoding in "Attention Is All You Need" but entirely in ℤ₃.

### Feed-Forward Block

`FFN(x) = W₂ · (W₁ · x)` — a two-layer ternary linear transformation. In pure ℤ₃, no activation function is needed between layers since the field structure already provides non-linearity through modular arithmetic.

### Residual Connection

`output = input ⊕ sublayer_output` — element-wise ℤ₃ addition. Note that residual connections in ℤ₃ are **not** identity-preserving in the traditional sense: x ⊕ x wraps around. This creates an interesting inductive bias.

### Encoder Block

`x → Residual(x, SelfAttention(x)) → Residual(·, FFN(·))`

### Decoder Block

`x → Residual(x, MaskedSelfAttention(x)) → Residual(·, CrossAttention(·, memory)) → Residual(·, FFN(·))`

## Usage

```rust
use ternary_transformer::*;

// Create input matrices (rows = sequence length, cols = d_model)
let source = TernaryMatrix::from_flat(4, 8, /* ... trits ... */);
let target = TernaryMatrix::from_flat(3, 8, /* ... trits ... */);

// Weight matrices (typically learned/quantized from a pre-trained model)
let w_q = TernaryMatrix::from_flat(8, 8, /* ... */);
// ... other weight matrices ...

// Forward pass
let output = ternary_transformer_forward(
    &source, &target,
    &w_q, &w_k, &w_v, &w_o, &w1, &w2,        // encoder weights
    &w_q_self, &w_k_self, &w_v_self, &w_o_self, // decoder self-attention
    &w_q_cross, &w_k_cross, &w_v_cross, &w_o_cross, // decoder cross-attention
    &dec_w1, &dec_w2,                            // decoder FFN
);
```

## Testing

The library includes comprehensive tests covering:

- **ℤ₃ arithmetic**: All 9 addition pairs and 9 multiplication pairs
- **Attention weight computation**: Verified against manual calculation
- **Multi-head parallelism**: Correct head splitting and concatenation
- **Position encoding uniqueness**: No two positions share the same encoding
- **Residual connections**: Zero-sublayer passthrough, doubling behavior
- **Matrix operations**: Multiplication, transpose
- **Full transformer forward pass**: End-to-end on small inputs

```bash
cargo test
```

## Performance Characteristics

Ternary operations have several advantages:

- **Memory**: 2 bits per parameter (vs 32 for float32) → 16× compression
- **Multiplication**: Trivial lookup table (9 entries) or branch prediction
- **Dot products**: No floating-point unit needed
- **Matrix multiply**: Can use bitwise operations and population counts

## Research Directions

- Gradient estimation through the ternary quantization bottleneck (Straight-Through Estimator variants)
- Mixed-precision training: continuous weights with ternary activations
- Scaling laws for ternary transformers
- ℤ₃-specific phenomena: exploiting the cyclic group structure for equivariant architectures
- Knowledge distillation from full-precision teachers

## License

MIT
