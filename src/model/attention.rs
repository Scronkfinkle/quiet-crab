use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Tensor, activation::softmax, backend::Backend},
};

/// Multi-head attention supporting self-attention and cross-attention.
///
/// Note: these values are the field names
/// inside of .safetensor files and changing them
/// will break loading
///
/// Used in three ways within Whisper:
/// - Encoder self-attention: all tokens attend to all other tokens
/// - Decoder causal self-attention: each token only attends to previous tokens (mask applied)
/// - Decoder cross-attention: decoder queries attend to encoder key/values (context provided)
#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    /// Number of attention heads
    n_heads: usize,
    /// Dimension per head: d_model / n_heads
    d_head: usize,
    /// QKV Matrices
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    /// Output Layer
    out_proj: Linear<B>,
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Create a new multi-head attention module.
    ///
    /// # Arguments
    /// * `d_model` - Model dimensionality (must be divisible by n_heads)
    /// * `n_heads` - Number of attention heads
    /// * `device` - Device to place parameters on
    pub fn new(d_model: usize, n_heads: usize, device: &B::Device) -> Self {
        assert!(
            d_model % n_heads == 0,
            "d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        );
        let d_head = d_model / n_heads;

        let query = LinearConfig::new(d_model, d_model).init(device);
        // Key matrix has no bias in the original Whisper implementation
        let key = LinearConfig::new(d_model, d_model)
            .with_bias(false)
            .init(device);
        let value = LinearConfig::new(d_model, d_model).init(device);
        let out = LinearConfig::new(d_model, d_model).init(device);

        Self {
            n_heads,
            d_head,
            q_proj: query,
            k_proj: key,
            v_proj: value,
            out_proj: out,
        }
    }

    /// Forward pass through multi-head attention.
    ///
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        context: Option<Tensor<B, 3>>,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let dims = x.shape().dims;
        let batch = dims[0];
        let seq_len = dims[1];

        let q = self.q_proj.forward(x.clone());

        // K and V come from context (cross-attention) or x itself (self-attention)
        let kv_src = context.unwrap_or(x);
        let k = self.k_proj.forward(kv_src.clone());
        let v = self.v_proj.forward(kv_src);

        let ctx_len = k.shape().dims[1];

        // Reshape from [batch, seq, d_model] -> [batch, n_heads, seq, d_head]
        // This splits the model dimension into parallel attention heads
        let q = q
            .reshape([batch, seq_len, self.n_heads, self.d_head])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, ctx_len, self.n_heads, self.d_head])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, ctx_len, self.n_heads, self.d_head])
            .swap_dims(1, 2);

        // Attention formula
        // pre-calculate 1/sqrt(d_k)
        let scale = 1.0 / (self.d_head as f64).sqrt();
        // This is the (QK^T)/sqrt(d_k)
        let scores = q.matmul(k.transpose()) * scale;

        // Add mask if provided (e.g. -inf at future positions for some of the decoding blocks)
        // unsqueeze broadcasts [seq, ctx] -> [1, 1, seq, ctx] to match scores dims
        let scores = if let Some(m) = mask {
            scores + m.unsqueeze::<4>()
        } else {
            scores
        };

        // Softmax over the key/context dimension to get attention weights
        let weights = softmax(scores, 3);

        // Weighted sum of values: [batch, n_heads, seq_len, d_head]
        let attended = weights.matmul(v);

        // Concatenate heads back: [batch, seq_len, d_model]
        let output = attended
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.n_heads * self.d_head]);

        // Final linear projection
        self.out_proj.forward(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_self_attention_shape() {
        let device = Default::default();
        let attn = MultiHeadAttention::<TestBackend>::new(384, 6, &device);

        let x = Tensor::<TestBackend, 3>::random(
            [2, 10, 384],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = attn.forward(x, None, None);
        assert_eq!(output.shape().dims, [2, 10, 384]);
    }

    #[test]
    fn test_cross_attention_shape() {
        let device = Default::default();
        let attn = MultiHeadAttention::<TestBackend>::new(384, 6, &device);

        // Decoder queries (shorter sequence)
        let x = Tensor::<TestBackend, 3>::random(
            [2, 5, 384],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        // Encoder outputs (longer sequence)
        let context = Tensor::<TestBackend, 3>::random(
            [2, 20, 384],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = attn.forward(x, Some(context), None);
        // Output follows the query sequence length, not the context length
        assert_eq!(output.shape().dims, [2, 5, 384]);
    }
}
