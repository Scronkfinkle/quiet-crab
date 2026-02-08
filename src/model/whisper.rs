use std::path::PathBuf;

use crate::model::{config::WhisperConfig, decoder::WhisperDecoder, encoder::WhisperEncoder};
use burn::{
    module::Module,
    tensor::{Int, Tensor, backend::Backend},
};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

/// Top-level Whisper model.
///
/// Wires the audio encoder and text decoder together:
///
///   mel spectrogram → encoder → hidden states
///   token ids + hidden states → decoder → logits
#[derive(Module, Debug)]
pub struct WhisperModel<B: Backend> {
    pub encoder: WhisperEncoder<B>,
    pub decoder: WhisperDecoder<B>,
}

impl<B: Backend> WhisperModel<B> {
    /// Construct a WhisperModel from a config.
    pub fn new(config: &WhisperConfig, device: &B::Device) -> Self {
        Self {
            encoder: WhisperEncoder::new(
                config.num_mel_bins,
                config.d_model,
                config.num_heads,
                config.encoder_layers,
                device,
            ),
            decoder: WhisperDecoder::new(
                config.vocab_size,
                config.d_model,
                config.num_heads,
                config.decoder_layers,
                config.max_target_positions,
                device,
            ),
        }
    }

    /// Full forward pass: mel spectrogram → token logits.
    ///
    /// # Arguments
    /// * `mel` - Log-mel spectrogram [batch, n_mels, time_steps]
    /// * `tokens` - Decoder input token IDs [batch, tgt_len]
    ///
    /// # Returns
    /// Logits over vocabulary [batch, tgt_len, vocab_size]
    pub fn forward(&self, mel: Tensor<B, 3>, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let encoder_out = self.encoder.forward(mel);
        self.decoder.forward(tokens, encoder_out)
    }

    /// Encode audio only (useful to cache encoder output during inference).
    pub fn encode(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.forward(mel)
    }

    /// Decode one step given cached encoder output.
    pub fn decode(&self, tokens: Tensor<B, 2, Int>, encoder_out: Tensor<B, 3>) -> Tensor<B, 3> {
        self.decoder.forward(tokens, encoder_out)
    }

    /// Load pre-trained weights from a HuggingFace safetensors file.
    ///
    /// Remaps PyTorch/HF key names to Burn module paths.
    pub fn from_safetensors(
        config: &WhisperConfig,
        path: impl Into<PathBuf>,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        let mut model = Self::new(config, device);
        let mut store = SafetensorsStore::from_file(path.into())
            .with_from_adapter(PyTorchToBurnAdapter)
            // Remove "model." prefix
            .with_key_remapping("^model\\.", "")
            // encoder.layers.N -> encoder.blocks.N
            .with_key_remapping("encoder\\.layers\\.(\\d+)\\.", "encoder.blocks.$1.")
            // decoder.layers.N -> decoder.blocks.N
            .with_key_remapping("decoder\\.layers\\.(\\d+)\\.", "decoder.blocks.$1.")
            // encoder.layer_norm -> encoder.norm
            .with_key_remapping("encoder\\.layer_norm\\.", "encoder.norm.")
            // decoder.layer_norm -> decoder.norm
            .with_key_remapping("decoder\\.layer_norm\\.", "decoder.norm.")
            // self_attn_layer_norm -> norm1
            .with_key_remapping("\\.self_attn_layer_norm\\.", ".norm1.")
            // encoder_attn_layer_norm -> norm2
            .with_key_remapping("\\.encoder_attn_layer_norm\\.", ".norm2.")
            // encoder blocks: final_layer_norm -> norm2
            .with_key_remapping("(encoder\\.blocks\\.\\d+)\\.final_layer_norm", "$1.norm2")
            // decoder blocks: final_layer_norm -> norm3
            .with_key_remapping("(decoder\\.blocks\\.\\d+)\\.final_layer_norm", "$1.norm3")
            // encoder_attn -> cross_attn
            .with_key_remapping("\\.encoder_attn\\.", ".cross_attn.")
            // fc1/fc2 -> ffn.fc1/ffn.fc2
            .with_key_remapping("\\.fc1\\.", ".ffn.fc1.")
            .with_key_remapping("\\.fc2\\.", ".ffn.fc2.")
            // embed_tokens -> token_embedding
            .with_key_remapping("decoder\\.embed_tokens\\.", "decoder.token_embedding.")
            // embed_positions -> positional_embedding
            .with_key_remapping(
                "decoder\\.embed_positions\\.",
                "decoder.positional_embedding.",
            );

        model
            .load_from(&mut store)
            .expect("Failed to load safetensors file");
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{NdArray, ndarray::NdArrayDevice};

    type TestBackend = NdArray;

    #[test]
    fn test_tiny_model_forward() {
        let device = Default::default();
        let config = WhisperConfig::tiny();
        let model = WhisperModel::<TestBackend>::new(&config, &device);

        // Fake mel spectrogram: [batch=1, n_mels=80, time=100]
        let mel = Tensor::<TestBackend, 3>::random(
            [1, 80, 100],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        // Fake decoder tokens: [batch=1, tgt_len=5]
        let tokens = Tensor::<TestBackend, 2, Int>::zeros([1, 5], &device);

        let logits = model.forward(mel, tokens);

        // [batch=1, tgt_len=5, vocab_size=51865]
        assert_eq!(logits.shape().dims, vec![1, 5, 51865]);
    }

    #[test]
    #[ignore = "requires whisper-small.safetensors on disk"]
    fn test_load_safetensors() {
        let device = NdArrayDevice::Cpu;
        let config = WhisperConfig::small();
        let model = WhisperModel::<NdArray>::from_safetensors(
            &config,
            "whisper-small.safetensors",
            &device,
        )
        .expect("should load weights");

        // Verify a basic forward pass works with the loaded weights
        let mel = Tensor::<NdArray, 3>::zeros([1, 80, 100], &device);
        let tokens = Tensor::<NdArray, 2, Int>::zeros([1, 1], &device);
        let logits = model.forward(mel, tokens);
        assert_eq!(logits.shape().dims[2], 51865);
    }

    #[test]
    fn test_encode_then_decode() {
        let device = Default::default();
        let config = WhisperConfig::tiny();
        let model = WhisperModel::<TestBackend>::new(&config, &device);

        let mel = Tensor::<TestBackend, 3>::random(
            [1, 80, 100],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        // Encode once, decode multiple times (simulates autoregressive generation)
        let encoder_out = model.encode(mel);
        assert_eq!(encoder_out.shape().dims[2], 384); // d_model

        let tokens = Tensor::<TestBackend, 2, Int>::zeros([1, 3], &device);
        let logits = model.decode(tokens, encoder_out);
        assert_eq!(logits.shape().dims, vec![1, 3, 51865]);
    }
}
