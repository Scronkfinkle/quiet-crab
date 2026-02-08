use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};

use crate::model::{attention::MultiHeadAttention, feed_forward::FeedForward};

/// Build a causal attention mask of shape [seq_len, seq_len].
///
/// Positions where j > i (future tokens) are filled with -inf so that
/// softmax assigns them zero weight. The diagonal and lower triangle are 0.
fn causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_data(TensorData::new(data, [seq_len, seq_len]), device)
}

/// Single decoder transformer block.
///
/// Pre-norm structure:
///   LayerNorm → Causal Self-Attention → Residual
///   LayerNorm → Cross-Attention (encoder K, V) → Residual
///   LayerNorm → FFN → Residual
#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    norm1: LayerNorm<B>,
    self_attn: MultiHeadAttention<B>,
    norm2: LayerNorm<B>,
    cross_attn: MultiHeadAttention<B>,
    norm3: LayerNorm<B>,
    ffn: FeedForward<B>,
}

impl<B: Backend> DecoderBlock<B> {
    pub fn new(d_model: usize, n_heads: usize, device: &B::Device) -> Self {
        Self {
            norm1: LayerNormConfig::new(d_model).init(device),
            self_attn: MultiHeadAttention::new(d_model, n_heads, device),
            norm2: LayerNormConfig::new(d_model).init(device),
            cross_attn: MultiHeadAttention::new(d_model, n_heads, device),
            norm3: LayerNormConfig::new(d_model).init(device),
            ffn: FeedForward::new(d_model, d_model * 4, device),
        }
    }

    /// Forward pass through one decoder block.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        encoder_output: Tensor<B, 3>,
        mask: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        // self-attention sub-layer (masks future tokens)
        let residual = x.clone();
        let x = self.norm1.forward(x);
        let x = self.self_attn.forward(x, None, Some(mask));
        let x = x + residual;

        // 2. Cross-attention sub-layer
        let residual = x.clone();
        let x = self.norm2.forward(x);
        let x = self.cross_attn.forward(x, Some(encoder_output), None);
        let x = x + residual;

        // 3. MLP sub-layer
        let residual = x.clone();
        let x = self.norm3.forward(x);
        let x = self.ffn.forward(x);
        x + residual
    }
}

/// Whisper text decoder.
#[derive(Module, Debug)]
pub struct WhisperDecoder<B: Backend> {
    token_embedding: Embedding<B>,
    positional_embedding: Embedding<B>,
    blocks: Vec<DecoderBlock<B>>,
    norm: LayerNorm<B>,
}

impl<B: Backend> WhisperDecoder<B> {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        max_target_positions: usize,
        device: &B::Device,
    ) -> Self {
        let blocks = (0..n_layers)
            .map(|_| DecoderBlock::new(d_model, n_heads, device))
            .collect();
        Self {
            token_embedding: EmbeddingConfig::new(vocab_size, d_model).init(device),
            positional_embedding: EmbeddingConfig::new(max_target_positions, d_model).init(device),
            blocks,
            norm: LayerNormConfig::new(d_model).init(device),
        }
    }

    /// Forward pass.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>, encoder_output: Tensor<B, 3>) -> Tensor<B, 3> {
        // The number of tokens
        let seq_len = tokens.shape().dims[1];
        let device = tokens.device();

        // Token embeddings: [batch, seq_len, d_model]
        let tok_emb = self.token_embedding.forward(tokens);

        // Position indices [0, 1, ..., seq_len-1] shaped as [1, seq_len] for embedding lookup
        let pos_ids: Vec<i64> = (0..seq_len as i64).collect();
        let positions =
            Tensor::<B, 2, Int>::from_data(TensorData::new(pos_ids, [1, seq_len]), &device);
        // pos_emb: [1, seq_len, d_model] — broadcasts over batch dimension
        let pos_emb = self.positional_embedding.forward(positions);

        let x = tok_emb + pos_emb;

        // mask with -inf above the diagonal
        let mask = causal_mask::<B>(seq_len, &device);

        // Decoder blocks
        let x = self.blocks.iter().fold(x, |x, block| {
            block.forward(x, encoder_output.clone(), mask.clone())
        });

        // Final norm
        let x = self.norm.forward(x);

        // Weight-tied output projection: x @ token_embedding.weight^T
        // token_embedding.weight: [vocab_size, d_model]
        // Result: [batch, seq_len, vocab_size]
        let dims = x.shape().dims;
        let (batch, seq, d) = (dims[0], dims[1], dims[2]);
        let vocab = self.token_embedding.weight.val().shape().dims[0];
        let w = self.token_embedding.weight.val().transpose(); // [d_model, vocab_size]
        x.reshape([batch * seq, d])
            .matmul(w)
            .reshape([batch, seq, vocab])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_decoder_output_shape() {
        let device = Default::default();
        // Tiny-model dimensions, 2 layers to keep test fast
        let decoder = WhisperDecoder::<TestBackend>::new(
            51865, // vocab_size
            384,   // d_model
            6,     // n_heads
            2,     // n_layers
            448,   // max_target_positions
            &device,
        );

        // Fake token IDs: [batch=1, seq_len=5]
        let tokens = Tensor::<TestBackend, 2, Int>::zeros([1, 5], &device);
        // Fake encoder output: [batch=1, enc_len=50, d_model=384]
        let encoder_out = Tensor::<TestBackend, 3>::random(
            [1, 50, 384],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let logits = decoder.forward(tokens, encoder_out);

        // [batch=1, seq_len=5, vocab_size=51865]
        assert_eq!(logits.shape().dims, [1, 5, 51865]);
    }
}
