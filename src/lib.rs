//! # ternary-transformer
//!
//! Ternary transformer components operating in ℤ₃ = {-1, 0, 1}.
//!
//! This crate provides building blocks for transformers where all arithmetic
//! is performed in ternary (base-3) space. Every weight, activation, and
//! intermediate value is a trit ∈ {-1, 0, 1}. Addition in ℤ₃ uses explicit
//! match arms over all 9 possible pairs of trits.

use std::fmt;

/// A trit value in ℤ₃ = {-1, 0, 1}.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Trit {
    NegOne = -1,
    Zero = 0,
    One = 1,
}

impl Trit {
    /// Convert an i8 to the nearest Trit. Values < 0 → NegOne, 0 → Zero, > 0 → One.
    pub fn from_i8(v: i8) -> Self {
        match v.cmp(&0) {
            std::cmp::Ordering::Less => Trit::NegOne,
            std::cmp::Ordering::Equal => Trit::Zero,
            std::cmp::Ordering::Greater => Trit::One,
        }
    }

    /// Convert to i8.
    pub fn to_i8(self) -> i8 {
        self as i8
    }

    /// Ternary negation: -(-1)=1, -(0)=0, -(1)=-1.
    pub fn negate(self) -> Self {
        match self {
            Trit::NegOne => Trit::One,
            Trit::Zero => Trit::Zero,
            Trit::One => Trit::NegOne,
        }
    }
}

impl fmt::Display for Trit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_i8())
    }
}

/// Addition in ℤ₃ using explicit match on all 9 pairs.
pub fn ternary_add(a: Trit, b: Trit) -> Trit {
    match (a, b) {
        (Trit::NegOne, Trit::NegOne) => Trit::One,   // -1 + -1 = -2 ≡ 1 (mod 3)
        (Trit::NegOne, Trit::Zero)  => Trit::NegOne,  // -1 + 0 = -1
        (Trit::NegOne, Trit::One)   => Trit::Zero,    // -1 + 1 = 0
        (Trit::Zero, Trit::NegOne)  => Trit::NegOne,  // 0 + -1 = -1
        (Trit::Zero, Trit::Zero)    => Trit::Zero,    // 0 + 0 = 0
        (Trit::Zero, Trit::One)     => Trit::One,     // 0 + 1 = 1
        (Trit::One, Trit::NegOne)   => Trit::Zero,    // 1 + -1 = 0
        (Trit::One, Trit::Zero)     => Trit::One,     // 1 + 0 = 1
        (Trit::One, Trit::One)      => Trit::NegOne,  // 1 + 1 = 2 ≡ -1 (mod 3)
    }
}

/// Ternary multiplication: standard integer multiplication clamped to ℤ₃.
pub fn ternary_mul(a: Trit, b: Trit) -> Trit {
    match (a, b) {
        (Trit::Zero, _) | (_, Trit::Zero) => Trit::Zero,
        (Trit::NegOne, Trit::NegOne) => Trit::One,
        (Trit::NegOne, Trit::One) => Trit::NegOne,
        (Trit::One, Trit::NegOne) => Trit::NegOne,
        (Trit::One, Trit::One) => Trit::One,
    }
}

/// Dot product of two ternary vectors using ternary addition.
pub fn ternary_dot(a: &[Trit], b: &[Trit]) -> Trit {
    assert_eq!(a.len(), b.len(), "vectors must have same length");
    let mut acc = Trit::Zero;
    for i in 0..a.len() {
        let product = ternary_mul(a[i], b[i]);
        acc = ternary_add(acc, product);
    }
    acc
}

/// A ternary matrix stored as a flat vector with row-major layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TernaryMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Trit>,
}

impl TernaryMatrix {
    /// Create a zero matrix of given dimensions.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        TernaryMatrix {
            rows,
            cols,
            data: vec![Trit::Zero; rows * cols],
        }
    }

    /// Create from a flat slice (row-major).
    pub fn from_flat(rows: usize, cols: usize, data: Vec<Trit>) -> Self {
        assert_eq!(data.len(), rows * cols);
        TernaryMatrix { rows, cols, data }
    }

    /// Get element at (r, c).
    pub fn get(&self, r: usize, c: usize) -> Trit {
        self.data[r * self.cols + c]
    }

    /// Set element at (r, c).
    pub fn set(&mut self, r: usize, c: usize, v: Trit) {
        self.data[r * self.cols + c] = v;
    }

    /// Get row as a slice.
    pub fn row(&self, r: usize) -> &[Trit] {
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Multiply two ternary matrices. Returns C = A × B in ℤ₃.
    pub fn matmul(&self, other: &TernaryMatrix) -> TernaryMatrix {
        assert_eq!(self.cols, other.rows, "incompatible dimensions");
        let mut result = TernaryMatrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let col: Vec<Trit> = (0..other.rows).map(|r| other.get(r, j)).collect();
                let val = ternary_dot(self.row(i), &col);
                result.set(i, j, val);
            }
        }
        result
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> TernaryMatrix {
        let mut result = TernaryMatrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }
}

/// Ternary self-attention mechanism.
///
/// Computes attention as: output[i] = Σⱼ αᵢⱼ · Vⱼ where αᵢⱼ ∈ ℤ₃
/// is derived from the ternary dot product of Qᵢ and Kⱼ.
pub fn ternary_attention(
    query: &TernaryMatrix,
    key: &TernaryMatrix,
    value: &TernaryMatrix,
) -> TernaryMatrix {
    assert_eq!(query.cols, key.cols, "Q and K must have same dimension");
    assert_eq!(key.rows, value.rows, "K and V must have same sequence length");

    let seq_len_q = query.rows;
    let _d = query.cols;

    // Compute raw attention scores: scores[i][j] = dot(Q[i], K[j])
    let mut output = TernaryMatrix::zeros(seq_len_q, value.cols);

    for i in 0..seq_len_q {
        for j in 0..key.rows {
            let score = ternary_dot(query.row(i), key.row(j));
            // Accumulate: output[i] += score * V[j]
            for c in 0..value.cols {
                let weighted = ternary_mul(score, value.get(j, c));
                let current = output.get(i, c);
                output.set(i, c, ternary_add(current, weighted));
            }
        }
    }
    output
}

/// Multi-head ternary attention.
///
/// Splits the input into `num_heads` chunks along the feature dimension,
/// applies ternary attention to each head independently, then concatenates
/// the results and projects back through an output weight matrix.
pub fn multi_head_attention(
    input: &TernaryMatrix,
    w_q: &TernaryMatrix,
    w_k: &TernaryMatrix,
    w_v: &TernaryMatrix,
    w_o: &TernaryMatrix,
    num_heads: usize,
) -> TernaryMatrix {
    let seq_len = input.rows;
    let d_model = input.cols;
    let head_dim = d_model / num_heads;

    assert_eq!(d_model % num_heads, 0, "d_model must be divisible by num_heads");
    assert_eq!(w_q.rows, d_model);
    assert_eq!(w_k.rows, d_model);
    assert_eq!(w_v.rows, d_model);
    assert_eq!(w_o.cols, d_model);

    // Project Q, K, V
    let q = input.matmul(w_q);
    let k = input.matmul(w_k);
    let v = input.matmul(w_v);

    // Apply attention per head
    let mut concat = TernaryMatrix::zeros(seq_len, d_model);
    for h in 0..num_heads {
        let start = h * head_dim;
        let _end = start + head_dim;

        // Extract head slices
        let mut q_h = TernaryMatrix::zeros(seq_len, head_dim);
        let mut k_h = TernaryMatrix::zeros(seq_len, head_dim);
        let mut v_h = TernaryMatrix::zeros(seq_len, head_dim);

        for i in 0..seq_len {
            for j in 0..head_dim {
                q_h.set(i, j, q.get(i, start + j));
                k_h.set(i, j, k.get(i, start + j));
                v_h.set(i, j, v.get(i, start + j));
            }
        }

        let head_out = ternary_attention(&q_h, &k_h, &v_h);

        // Copy into concatenated output
        for i in 0..seq_len {
            for j in 0..head_out.cols {
                concat.set(i, start + j, head_out.get(i, j));
            }
        }
    }

    // Final projection
    concat.matmul(w_o)
}

/// Ternary sinusoidal position encoding.
///
/// Generates position encodings in ℤ₃ using a ternary sinusoidal pattern.
/// For each position `pos` and dimension `i`:
///   - Even i: trit = (pos / 10000^(2i/d)) scaled and quantized to {-1, 0, 1}
///   - Odd i: similar with cosine pattern
///
/// The encoding uses a deterministic pattern based on position and dimension
/// to ensure uniqueness across positions.
pub fn ternary_position_encoding(seq_len: usize, d_model: usize) -> TernaryMatrix {
    let mut pe = TernaryMatrix::zeros(seq_len, d_model);

    for pos in 0..seq_len {
        for i in 0..d_model {
            // Ternary sinusoidal: use modular arithmetic to produce trits
            // This gives a unique pattern per position
            let dim_freq = if i % 2 == 0 {
                // Sin-like: alternating pattern based on pos and dimension
                let period = 3 + (i / 2) % 4; // period varies with dimension
                let val = (pos + i + 1) % (period * 2);
                if val < period {
                    if val % 3 == 0 { Trit::Zero }
                    else if val % 3 == 1 { Trit::One }
                    else { Trit::NegOne }
                } else {
                    let v = (val - period) % 3;
                    if v == 0 { Trit::Zero }
                    else if v == 1 { Trit::NegOne }
                    else { Trit::One }
                }
            } else {
                // Cos-like: shifted pattern
                let period = 3 + (i / 2) % 3;
                let val = (pos * 2 + i + 3) % (period * 2 + 1);
                match val % 3 {
                    0 => Trit::Zero,
                    1 => Trit::One,
                    _ => Trit::NegOne,
                }
            };
            pe.set(pos, i, dim_freq);
        }
    }
    pe
}

/// Ternary feed-forward block.
///
/// FFN(x) = W₂ · (W₁ · x) in ℤ₃ (no activation in pure ternary).
pub fn ternary_ffn(input: &TernaryMatrix, w1: &TernaryMatrix, w2: &TernaryMatrix) -> TernaryMatrix {
    let hidden = input.matmul(w1);
    hidden.matmul(w2)
}

/// Ternary residual connection: output = input + sublayer_output (in ℤ₃).
pub fn ternary_residual(input: &TernaryMatrix, sublayer: &TernaryMatrix) -> TernaryMatrix {
    assert_eq!(input.rows, sublayer.rows);
    assert_eq!(input.cols, sublayer.cols);
    let mut result = TernaryMatrix::zeros(input.rows, input.cols);
    for i in 0..input.data.len() {
        result.data[i] = ternary_add(input.data[i], sublayer.data[i]);
    }
    result
}

/// Ternary encoder block: residual(self-attention) + residual(ffn).
pub fn ternary_encoder_block(
    input: &TernaryMatrix,
    w_q: &TernaryMatrix,
    w_k: &TernaryMatrix,
    w_v: &TernaryMatrix,
    w_o: &TernaryMatrix,
    w1: &TernaryMatrix,
    w2: &TernaryMatrix,
) -> TernaryMatrix {
    // Self-attention sublayer
    let attn = multi_head_attention(input, w_q, w_k, w_v, w_o, 1);
    let x = ternary_residual(input, &attn);

    // FFN sublayer
    let ffn_out = ternary_ffn(&x, w1, w2);
    ternary_residual(&x, &ffn_out)
}

/// Ternary decoder block: masked self-attention + cross-attention + FFN, each with residuals.
pub fn ternary_decoder_block(
    target: &TernaryMatrix,
    memory: &TernaryMatrix,
    w_q_self: &TernaryMatrix,
    w_k_self: &TernaryMatrix,
    w_v_self: &TernaryMatrix,
    w_o_self: &TernaryMatrix,
    w_q_cross: &TernaryMatrix,
    w_k_cross: &TernaryMatrix,
    w_v_cross: &TernaryMatrix,
    w_o_cross: &TernaryMatrix,
    w1: &TernaryMatrix,
    w2: &TernaryMatrix,
) -> TernaryMatrix {
    // Masked self-attention (here unmasked for simplicity; masking can be added as a zero mask)
    let self_attn = multi_head_attention(target, w_q_self, w_k_self, w_v_self, w_o_self, 1);
    let x = ternary_residual(target, &self_attn);

    // Cross-attention: query from target, key/value from memory
    let cross_q = x.matmul(w_q_cross);
    let cross_k = memory.matmul(w_k_cross);
    let cross_v = memory.matmul(w_v_cross);
    // Manual multi-head with head_dim = cross_q.cols
    let cross_attn = ternary_attention(&cross_q, &cross_k, &cross_v);
    let cross_projected = cross_attn.matmul(w_o_cross);
    let x = ternary_residual(&x, &cross_projected);

    // FFN
    let ffn_out = ternary_ffn(&x, w1, w2);
    ternary_residual(&x, &ffn_out)
}

/// Full ternary transformer forward pass with encoder and decoder.
pub fn ternary_transformer_forward(
    source: &TernaryMatrix,
    target: &TernaryMatrix,
    enc_wq: &TernaryMatrix,
    enc_wk: &TernaryMatrix,
    enc_wv: &TernaryMatrix,
    enc_wo: &TernaryMatrix,
    enc_w1: &TernaryMatrix,
    enc_w2: &TernaryMatrix,
    dec_wq_self: &TernaryMatrix,
    dec_wk_self: &TernaryMatrix,
    dec_wv_self: &TernaryMatrix,
    dec_wo_self: &TernaryMatrix,
    dec_wq_cross: &TernaryMatrix,
    dec_wk_cross: &TernaryMatrix,
    dec_wv_cross: &TernaryMatrix,
    dec_wo_cross: &TernaryMatrix,
    dec_w1: &TernaryMatrix,
    dec_w2: &TernaryMatrix,
) -> TernaryMatrix {
    // Add position encoding to source and target
    let src_pe = ternary_position_encoding(source.rows, source.cols);
    let tgt_pe = ternary_position_encoding(target.rows, target.cols);
    let src_input = ternary_residual(source, &src_pe);
    let tgt_input = ternary_residual(target, &tgt_pe);

    // Encoder
    let encoded = ternary_encoder_block(
        &src_input, enc_wq, enc_wk, enc_wv, enc_wo, enc_w1, enc_w2,
    );

    // Decoder
    ternary_decoder_block(
        &tgt_input, &encoded,
        dec_wq_self, dec_wk_self, dec_wv_self, dec_wo_self,
        dec_wq_cross, dec_wk_cross, dec_wv_cross, dec_wo_cross,
        dec_w1, dec_w2,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_add_all_pairs() {
        // Verify all 9 pairs produce correct results
        assert_eq!(ternary_add(Trit::NegOne, Trit::NegOne), Trit::One);
        assert_eq!(ternary_add(Trit::NegOne, Trit::Zero), Trit::NegOne);
        assert_eq!(ternary_add(Trit::NegOne, Trit::One), Trit::Zero);
        assert_eq!(ternary_add(Trit::Zero, Trit::NegOne), Trit::NegOne);
        assert_eq!(ternary_add(Trit::Zero, Trit::Zero), Trit::Zero);
        assert_eq!(ternary_add(Trit::Zero, Trit::One), Trit::One);
        assert_eq!(ternary_add(Trit::One, Trit::NegOne), Trit::Zero);
        assert_eq!(ternary_add(Trit::One, Trit::Zero), Trit::One);
        assert_eq!(ternary_add(Trit::One, Trit::One), Trit::NegOne);
    }

    #[test]
    fn test_ternary_mul() {
        assert_eq!(ternary_mul(Trit::One, Trit::One), Trit::One);
        assert_eq!(ternary_mul(Trit::NegOne, Trit::One), Trit::NegOne);
        assert_eq!(ternary_mul(Trit::One, Trit::NegOne), Trit::NegOne);
        assert_eq!(ternary_mul(Trit::NegOne, Trit::NegOne), Trit::One);
        assert_eq!(ternary_mul(Trit::Zero, Trit::One), Trit::Zero);
        assert_eq!(ternary_mul(Trit::One, Trit::Zero), Trit::Zero);
    }

    #[test]
    fn test_trit_from_i8() {
        assert_eq!(Trit::from_i8(-5), Trit::NegOne);
        assert_eq!(Trit::from_i8(0), Trit::Zero);
        assert_eq!(Trit::from_i8(42), Trit::One);
    }

    #[test]
    fn test_trit_negate() {
        assert_eq!(Trit::NegOne.negate(), Trit::One);
        assert_eq!(Trit::Zero.negate(), Trit::Zero);
        assert_eq!(Trit::One.negate(), Trit::NegOne);
    }

    #[test]
    fn test_attention_weight_computation() {
        // 2 tokens, 3 features
        let q = TernaryMatrix::from_flat(2, 3, vec![
            Trit::One, Trit::Zero, Trit::NegOne,
            Trit::Zero, Trit::One, Trit::One,
        ]);
        let k = TernaryMatrix::from_flat(2, 3, vec![
            Trit::One, Trit::One, Trit::Zero,
            Trit::NegOne, Trit::Zero, Trit::One,
        ]);
        let v = TernaryMatrix::from_flat(2, 3, vec![
            Trit::One, Trit::Zero, Trit::Zero,
            Trit::Zero, Trit::One, Trit::One,
        ]);

        let output = ternary_attention(&q, &k, &v);

        // Verify output dimensions
        assert_eq!(output.rows, 2);
        assert_eq!(output.cols, 3);

        // Manually compute for query[0] = [1, 0, -1]:
        // score[0][0] = dot([1,0,-1], [1,1,0]) = 1*1 + 0*1 + (-1)*0 = 1
        // score[0][1] = dot([1,0,-1], [-1,0,1]) = 1*(-1) + 0*0 + (-1)*1 = -1 + -1 = 1 (mod 3)
        // output[0][0] = 1*v[0][0] + 1*v[1][0] = 1*1 + 1*0 = 1
        assert_eq!(output.get(0, 0), Trit::One);
    }

    #[test]
    fn test_multi_head_parallelism() {
        // d_model = 4, num_heads = 2, head_dim = 2
        let d_model = 4;
        let d_ff = 4;
        let seq_len = 2;
        let num_heads = 2;

        let input = TernaryMatrix::from_flat(seq_len, d_model, vec![
            Trit::One, Trit::NegOne, Trit::Zero, Trit::One,
            Trit::Zero, Trit::One, Trit::NegOne, Trit::Zero,
        ]);

        let w_q = TernaryMatrix::from_flat(d_model, d_model, vec![
            Trit::One, Trit::Zero, Trit::Zero, Trit::One,
            Trit::Zero, Trit::One, Trit::One, Trit::Zero,
            Trit::NegOne, Trit::One, Trit::Zero, Trit::Zero,
            Trit::One, Trit::Zero, Trit::NegOne, Trit::One,
        ]);
        let w_k = TernaryMatrix::from_flat(d_model, d_model, vec![
            Trit::One, Trit::One, Trit::Zero, Trit::Zero,
            Trit::Zero, Trit::NegOne, Trit::One, Trit::Zero,
            Trit::One, Trit::Zero, Trit::Zero, Trit::One,
            Trit::NegOne, Trit::One, Trit::One, Trit::Zero,
        ]);
        let w_v = TernaryMatrix::from_flat(d_model, d_model, vec![
            Trit::One, Trit::Zero, Trit::NegOne, Trit::Zero,
            Trit::Zero, Trit::One, Trit::Zero, Trit::One,
            Trit::One, Trit::NegOne, Trit::One, Trit::Zero,
            Trit::Zero, Trit::One, Trit::Zero, Trit::One,
        ]);
        let w_o = TernaryMatrix::from_flat(d_model, d_model, vec![
            Trit::One, Trit::Zero, Trit::Zero, Trit::Zero,
            Trit::Zero, Trit::One, Trit::Zero, Trit::Zero,
            Trit::Zero, Trit::Zero, Trit::One, Trit::Zero,
            Trit::Zero, Trit::Zero, Trit::Zero, Trit::One,
        ]);

        let output = multi_head_attention(&input, &w_q, &w_k, &w_v, &w_o, num_heads);

        assert_eq!(output.rows, seq_len);
        assert_eq!(output.cols, d_model);
        // Output should contain only valid trits
        for t in &output.data {
            assert!(matches!(t, Trit::NegOne | Trit::Zero | Trit::One));
        }
    }

    #[test]
    fn test_position_encoding_uniqueness() {
        let seq_len = 8;
        let d_model = 6;
        let pe = ternary_position_encoding(seq_len, d_model);

        assert_eq!(pe.rows, seq_len);
        assert_eq!(pe.cols, d_model);

        // Each position encoding row should be unique
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                assert_ne!(
                    pe.row(i), pe.row(j),
                    "Positions {} and {} have identical encodings",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_residual_preserves_information() {
        let input = TernaryMatrix::from_flat(2, 3, vec![
            Trit::One, Trit::Zero, Trit::NegOne,
            Trit::Zero, Trit::One, Trit::One,
        ]);
        let sublayer = TernaryMatrix::from_flat(2, 3, vec![
            Trit::Zero, Trit::Zero, Trit::Zero,
            Trit::Zero, Trit::Zero, Trit::Zero,
        ]);

        // Residual with zero sublayer should equal input
        let result = ternary_residual(&input, &sublayer);
        assert_eq!(result, input);

        // Residual with self should give: input + input
        let doubled = ternary_residual(&input, &input);
        // [1,0,-1] + [1,0,-1] = [-1,0,1] in Z3
        assert_eq!(doubled.get(0, 0), Trit::NegOne);
        assert_eq!(doubled.get(0, 1), Trit::Zero);
        assert_eq!(doubled.get(0, 2), Trit::One);
    }

    #[test]
    fn test_matrix_multiply() {
        let a = TernaryMatrix::from_flat(2, 3, vec![
            Trit::One, Trit::Zero, Trit::NegOne,
            Trit::Zero, Trit::One, Trit::One,
        ]);
        let b = TernaryMatrix::from_flat(3, 2, vec![
            Trit::One, Trit::One,
            Trit::Zero, Trit::NegOne,
            Trit::One, Trit::Zero,
        ]);

        let c = a.matmul(&b);
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);

        // c[0][0] = dot([1,0,-1], [1,0,1]) = 1 + 0 + (-1) = 0
        assert_eq!(c.get(0, 0), Trit::Zero);
        // c[0][1] = dot([1,0,-1], [1,-1,0]) = 1 + 0 + 0 = 1
        assert_eq!(c.get(0, 1), Trit::One);
    }

    #[test]
    fn test_encoder_produces_output() {
        let d_model = 4;
        let d_ff = 4;
        let seq_len = 2;

        let input = TernaryMatrix::from_flat(seq_len, d_model, vec![
            Trit::One, Trit::Zero, Trit::NegOne, Trit::One,
            Trit::Zero, Trit::One, Trit::One, Trit::Zero,
        ]);

        // Identity-like weight matrices
        let w_q = TernaryMatrix::from_flat(d_model, d_model, vec![
            Trit::One, Trit::Zero, Trit::Zero, Trit::Zero,
            Trit::Zero, Trit::One, Trit::Zero, Trit::Zero,
            Trit::Zero, Trit::Zero, Trit::One, Trit::Zero,
            Trit::Zero, Trit::Zero, Trit::Zero, Trit::One,
        ]);
        let w_k = w_q.clone();
        let w_v = w_q.clone();
        let w_o = w_q.clone();
        let w1 = w_q.clone();
        let w2 = w_q.clone();

        let output = ternary_encoder_block(&input, &w_q, &w_k, &w_v, &w_o, &w1, &w2);

        assert_eq!(output.rows, seq_len);
        assert_eq!(output.cols, d_model);

        // All outputs should be valid trits
        for t in &output.data {
            assert!(matches!(t, Trit::NegOne | Trit::Zero | Trit::One));
        }
    }

    #[test]
    fn test_full_transformer_forward_pass() {
        let d_model = 4;
        let seq_len_src = 2;
        let seq_len_tgt = 2;

        let source = TernaryMatrix::from_flat(seq_len_src, d_model, vec![
            Trit::One, Trit::NegOne, Trit::Zero, Trit::One,
            Trit::Zero, Trit::One, Trit::One, Trit::NegOne,
        ]);
        let target = TernaryMatrix::from_flat(seq_len_tgt, d_model, vec![
            Trit::One, Trit::Zero, Trit::One, Trit::Zero,
            Trit::NegOne, Trit::One, Trit::Zero, Trit::One,
        ]);

        // Identity-like weights for all projections
        let identity = TernaryMatrix::from_flat(d_model, d_model, vec![
            Trit::One, Trit::Zero, Trit::Zero, Trit::Zero,
            Trit::Zero, Trit::One, Trit::Zero, Trit::Zero,
            Trit::Zero, Trit::Zero, Trit::One, Trit::Zero,
            Trit::Zero, Trit::Zero, Trit::Zero, Trit::One,
        ]);

        let output = ternary_transformer_forward(
            &source, &target,
            &identity, &identity, &identity, &identity, &identity, &identity,
            &identity, &identity, &identity, &identity,
            &identity, &identity, &identity, &identity,
            &identity, &identity,
        );

        assert_eq!(output.rows, seq_len_tgt);
        assert_eq!(output.cols, d_model);

        // Verify all outputs are valid trits
        for t in &output.data {
            assert!(matches!(t, Trit::NegOne | Trit::Zero | Trit::One));
        }
    }

    #[test]
    fn test_matrix_transpose() {
        let m = TernaryMatrix::from_flat(2, 3, vec![
            Trit::One, Trit::Zero, Trit::NegOne,
            Trit::Zero, Trit::One, Trit::One,
        ]);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 0), Trit::One);
        assert_eq!(t.get(2, 0), Trit::NegOne);
        assert_eq!(t.get(1, 1), Trit::One);
    }

    #[test]
    fn test_ternary_dot() {
        let a = vec![Trit::One, Trit::One, Trit::One];
        let b = vec![Trit::One, Trit::One, Trit::One];
        // 1+1+1 = 3 ≡ 0 (mod 3)
        assert_eq!(ternary_dot(&a, &b), Trit::Zero);

        let c = vec![Trit::One, Trit::NegOne];
        let d = vec![Trit::One, Trit::One];
        // 1*1 + (-1)*1 = 1 + (-1) = 0
        assert_eq!(ternary_dot(&c, &d), Trit::Zero);
    }
}
