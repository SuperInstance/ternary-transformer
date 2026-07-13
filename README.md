# ternary-transformer

The full transformer architecture — attention, encoding, decoding, position encoding, FFN, residual connections — where every weight, activation, and intermediate value lives in ℤ₃ = {-1, 0, 1}.

[![crates.io](https://img.shields.io/crates/v/ternary-transformer.svg)](https://crates.io/crates/ternary-transformer)
[![docs.rs](https://docs.rs/ternary-transformer/badge.svg)](https://docs.rs/ternary-transformer)

---

## Why this exists

Binary neural networks (weights in {-1, +1}) sacrifice representational capacity for efficiency. Ternary neural networks add a zero state and gain log₂(3) ≈ 1.58× more information per parameter while keeping the hardware story clean: 2 bits per weight, no floating-point units needed, multiplications become sign comparisons.

This crate implements the *entire* transformer in ℤ₃ — not a hybrid, not a quantized approximation. Every addition is modular (1 + 1 = −1 mod 3), every multiplication is a lookup table, every dot product accumulates in ternary space. It's a laboratory for answering: *what happens when you build a transformer from the ground up in a finite field?*

## The key insight

ℤ₃ is a finite field — addition and multiplication both have inverses, and every non-zero element is a unit. This means the entire transformer computation is algebraically well-founded in a way that's not true for saturated or clipped arithmetic. The residual connection `x ⊕ x` wraps around to `-x` in ℤ₃ — residual connections are actually *negating* the input, not amplifying it. This creates a fundamentally different inductive bias than FP32 residuals, and it's one that only makes sense when you commit to the full modular structure.

## Quick Start

The snippet below is **illustrative** (placeholders elided for brevity, not runnable as-is). See `src/lib.rs` tests for fully worked, compiling examples.

```rust
use ternary_transformer::{Trit, TernaryMatrix, ternary_transformer_forward};

// Input matrices: rows = sequence length, cols = d_model.
// Each entry is a Trit (NegOne / Zero / One).
let source = TernaryMatrix::from_flat(4, 8, vec![/* 4 * 8 = 32 Trits */]);
let target = TernaryMatrix::from_flat(3, 8, vec![/* 3 * 8 = 24 Trits */]);

// Every weight below is a d_model x d_model TernaryMatrix, learned or
// quantized from a pre-trained model (see ternary-quantize). Here `w`
// stands in for one such matrix; the forward pass needs 18 of them.
let w: TernaryMatrix = TernaryMatrix::zeros(8, 8);

// Full encoder-decoder forward pass.
let _output = ternary_transformer_forward(
    &source, &target,
    &w, &w, &w, &w, &w, &w,           // encoder: Wq Wk Wv Wo W1 W2
    &w, &w, &w, &w,                    // decoder self-attn: Wq Wk Wv Wo
    &w, &w, &w, &w,                    // decoder cross-attn: Wq Wk Wv Wo
    &w, &w,                            // decoder FFN: W1 W2
);
```

### Building blocks

```rust
// ℤ₃ arithmetic
let sum = ternary_add(Trit::One, Trit::One);   // → NegOne (1+1=2≡-1 mod 3)
let prod = ternary_mul(Trit::NegOne, Trit::One); // → NegOne
let dot = ternary_dot(&[Trit::One, Trit::One], &[Trit::One, Trit::NegOne]); // → Zero

// Matrix operations
let c = a.matmul(&b);       // ℤ₃ matrix multiply
let t = a.transpose();      // transpose
let att = ternary_attention(&q, &k, &v);  // self-attention

// Multi-head attention
let mha = multi_head_attention(&input, &w_q, &w_k, &w_v, &w_o, num_heads);

// Position encoding (ternary sinusoidal — unique per position)
let pe = ternary_position_encoding(seq_len, d_model);

// Feed-forward: FFN(x) = W₂ · (W₁ · x) — no activation needed in ℤ₃
let ffn_out = ternary_ffn(&x, &w1, &w2);

// Residual: output = input ⊕ sublayer (modular addition)
let res = ternary_residual(&input, &attn);

// Encoder block: residual(self-attn) → residual(ffn)
let encoded = ternary_encoder_block(&input, &wq, &wk, &wv, &wo, &w1, &w2);

// Decoder block: masked-self-attn → cross-attn → ffn
let decoded = ternary_decoder_block(
    &target, &memory,
    &wq_self, &wk_self, &wv_self, &wo_self,
    &wq_cross, &wk_cross, &wv_cross, &wo_cross,
    &w1, &w2,
);
```

## Architecture

```
Input ──→ + PosEnc ──→ EncoderBlock ──→ Memory (encoded source)
                              │
                    Residual(SelfAttention(Q,K,V))
                    Residual(FFN(x))
                              │
Target ──→ + PosEnc ──→ DecoderBlock ──→ Output
                              │
                    Residual(MaskedSelfAttn)
                    Residual(CrossAttn(Q=tgt, K,V=memory))
                    Residual(FFN(x))
```

Each block applies two residual connections. In ℤ₃, residuals don't preserve identity — `x ⊕ sublayer(x)` wraps around. The ℤ₃ field structure provides non-linearity through modular arithmetic alone, so no activation function is needed between FFN layers.

### ℤ₃ Arithmetic

All operations use explicit pattern matching over the 9 possible trit pairs:

```
ℤ₃ Addition:              ℤ₃ Multiplication:
  +  | -1   0   +1           ×  | -1   0   +1
-1  | +1  -1    0          -1  | +1   0   -1
 0  | -1   0   +1           0  |  0   0    0
+1  |  0  +1   -1          +1  | -1   0   +1
```

1 + 1 = 2 ≡ −1 (mod 3). −1 + −1 = −2 ≡ +1 (mod 3). Proper modular arithmetic.

## API Reference

### Core Types

```rust
pub enum Trit { NegOne, Zero, One }
// from_i8(v: i8) -> Self, to_i8() -> i8, to_f32() -> f32, negate() -> Self

pub struct TernaryMatrix {
    pub rows: usize, pub cols: usize, pub data: Vec<Trit>,
}
// zeros(rows, cols), from_flat(rows, cols, data)
// get(r, c), set(r, c, v), row(r)
// matmul(&other) -> TernaryMatrix
// transpose() -> TernaryMatrix
```

### Functions

| Signature | Description |
|-----------|-------------|
| `ternary_add(a: Trit, b: Trit) -> Trit` | ℤ₃ addition (9-way match) |
| `ternary_mul(a: Trit, b: Trit) -> Trit` | ℤ₃ multiplication |
| `ternary_dot(a: &[Trit], b: &[Trit]) -> Trit` | Dot product in ℤ₃ |
| `ternary_attention(q, k, v) -> TernaryMatrix` | Self-attention |
| `multi_head_attention(input, w_q, w_k, w_v, w_o, num_heads) -> TernaryMatrix` | Multi-head attention |
| `ternary_position_encoding(seq_len, d_model) -> TernaryMatrix` | Unique position encoding |
| `ternary_ffn(input, w1, w2) -> TernaryMatrix` | Two-layer FFN |
| `ternary_residual(input, sublayer) -> TernaryMatrix` | ℤ₃ residual connection |
| `ternary_encoder_block(...)` | Full encoder block |
| `ternary_decoder_block(...)` | Full decoder block |
| `ternary_transformer_forward(...)` | End-to-end encoder-decoder |

## Real-world example

A drone runs obstacle detection on a ternary neural network: LIDAR returns are quantized to {-1: approaching, 0: static, +1: receding}. The transformer processes a 16-token sequence (8 angular sectors × 2 time steps) through a 2-layer encoder with d_model=32.

In ℤ₃, the entire forward pass needs:
- **Weights**: 2 bits × 32 × 32 × 6 matrices = 1,536 bytes per layer (12,288 bits)
- **Activations**: 2 bits × 16 × 32 = 128 bytes
- **No floating-point unit**: just sign comparisons and modular adds

The drone's microcontroller has 64 KB SRAM. A full-precision transformer with the same architecture would need ~50 KB just for weights. The ternary version uses ~3 KB — leaving room for control loops, sensor fusion, and radio communication.

## Ecosystem connections

- **[`ternary-quantize`](https://github.com/SuperInstance/ternary-quantize)** — converts FP32 pretrained weights to trits for this transformer
- **[`ternary-knn`](https://github.com/SuperInstance/ternary-knn)** — classifies transformer output embeddings
- **[`ternary-pipeline-parallel`](https://github.com/SuperInstance/ternary-pipeline-parallel)** — pipelines encoder/decoder across devices
- **[`ternary-tensor-parallel`](https://github.com/SuperInstance/ternary-tensor-parallel)** — splits attention heads across devices
- **[`ternary-command-buffer`](https://github.com/SuperInstance/ternary-command-buffer)** — records transformer forward passes for replay

## Performance

| Property | Value |
|----------|-------|
| Memory per parameter | 2 bits (vs 32 for FP32) — 16× compression |
| Multiplication | 9-entry lookup table or branch prediction |
| Dot product | No FPU needed — modular adds only |
| Matrix multiply | O(m·n·k) modular operations |

No SIMD, no GPU. The ℤ₃ operations are branching-heavy — a hardware implementation using dedicated trit ALUs would see different performance characteristics.

## Open questions

- **Training in ℤ₃**: How do you backpropagate through modular addition? Straight-Through Estimator (STE) works for {-1, +1} but the zero state introduces a plateau that kills gradients.
- **Why no activation in FFN?**: In ℤ₃, the field structure provides non-linearity through modular wrapping. But is this enough? Experiments suggest the representation capacity is there, but convergence during training is harder.
- **Masked attention**: The decoder uses unmasked self-attention for simplicity. A proper causal mask (zeroing future positions) is straightforward but changes the attention distribution in unexpected ways in ℤ₃.
- **Scaling laws**: Do ternary transformers follow the Chinchilla scaling laws, or does the discrete bottleneck change the compute-optimal frontier?

## Testing

```bash
cargo test
```

19 tests covering: all 9 ℤ₃ addition pairs, all 9 multiplication pairs, `from_i8`/`to_i8`/`to_f32`/`negate` conversions, attention output checked against a full hand-computed 2×3 score matrix, multi-head head splitting/concatenation, position-encoding uniqueness for the verified `(seq_len=8, d_model=6)` config, residual zero-passthrough and doubling, matrix multiply and transpose (plus transpose self-inverse), FFN identity property, the full encoder block checked against an exact hand-derived output matrix, and the full transformer forward pass end-to-end. Error paths (`ternary_dot`/`matmul` dimension mismatch, non-divisible `num_heads`) are covered with panic assertions.

## License

MIT
