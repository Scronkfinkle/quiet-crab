use burn::{
    module::Module,
    nn::{
        Gelu, LayerNorm, LayerNormConfig, PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{Tensor, backend::Backend},
};

use crate::model::{
    attention::MultiHeadAttention, feed_forward::FeedForward, positional::sinusoids,
};

/// Single encoder transformer block.
///
/// Pre-norm structure:
///   LayerNorm → Self-Attention → Residual
///   LayerNorm → FFN → Residual
#[derive(Module, Debug)]
pub struct EncoderBlock<B: Backend> {
    norm1: LayerNorm<B>,
    self_attn: MultiHeadAttention<B>,
    norm2: LayerNorm<B>,
    ffn: FeedForward<B>,
}

impl<B: Backend> EncoderBlock<B> {
    pub fn new(d_model: usize, n_heads: usize, device: &B::Device) -> Self {
        Self {
            norm1: LayerNormConfig::new(d_model).init(device),
            self_attn: MultiHeadAttention::new(d_model, n_heads, device),
            norm2: LayerNormConfig::new(d_model).init(device),
            // FFN intermediate dimension is 4x d_model (standard transformer ratio)
            ffn: FeedForward::new(d_model, d_model * 4, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // self-attention sub-layer
        let residual = x.clone();
        let x = self.norm1.forward(x);
        let x = self.self_attn.forward(x, None, None);
        let x = x + residual;

        // MLP sub-layer
        let residual = x.clone();
        let x = self.norm2.forward(x);
        let x = self.ffn.forward(x);
        x + residual
    }
}

/// Whisper audio encoder.
///
/// Converts a log-mel spectrogram into encoder hidden states for the decoder.
#[derive(Module, Debug)]
pub struct WhisperEncoder<B: Backend> {
    conv1: Conv1d<B>,
    conv2: Conv1d<B>,
    gelu: Gelu,
    blocks: Vec<EncoderBlock<B>>,
    norm: LayerNorm<B>,
    d_model: usize,
}

impl<B: Backend> WhisperEncoder<B> {
    pub fn new(
        n_mels: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        device: &B::Device,
    ) -> Self {
        // All the values here were eyeballed from the Whisper python repo
        let conv1 = Conv1dConfig::new(n_mels, d_model, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .init(device);

        let conv2 = Conv1dConfig::new(d_model, d_model, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1))
            .init(device);

        let blocks = (0..n_layers)
            .map(|_| EncoderBlock::new(d_model, n_heads, device))
            .collect();

        Self {
            conv1,
            conv2,
            gelu: Gelu::new(),
            blocks,
            norm: LayerNormConfig::new(d_model).init(device),
            d_model,
        }
    }

    /// Forward pass.
    ///
    /// Encoder hidden states [num_batches, time_steps/2, d_model]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // We start with 2 × Conv1D + GELU
        let x = self.gelu.forward(self.conv1.forward(x));
        let x = self.gelu.forward(self.conv2.forward(x));

        // Rearrange to [num_batches, time/2, d_model] for the transformer blocks
        let x = x.swap_dims(1, 2);

        // Add sinusoidal positional embeddings
        let n_samples = x.shape().dims[1];
        let device = x.device();
        let pos_emb = sinusoids::<B>(n_samples, self.d_model, &device);
        // sinusoids returns [seq_len, d_model]; unsqueeze adds batch dim -> [1, seq_len, d_model]
        let x = x + pos_emb.unsqueeze::<3>();

        // Stack of transformer encoder blocks
        let x = self.blocks.iter().fold(x, |x, block| block.forward(x));

        // After the encoder blocks are run, whisper runs a final LayerNorm
        self.norm.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_encoder_output_shape() {
        let device = Default::default();
        // Use tiny-model params: d_model=384, 6 heads, but only 2 layers to keep test fast
        let encoder = WhisperEncoder::<TestBackend>::new(80, 384, 6, 2, &device);

        // Fake mel spectrogram: [batch=1, n_mels=80, time=100]
        let mel = Tensor::<TestBackend, 3>::random(
            [1, 80, 100],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let out = encoder.forward(mel);

        // stride=2 halves the time: 100 -> 50
        assert_eq!(out.shape().dims, [1, 50, 384]);
    }
}
