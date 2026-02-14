use std::path::PathBuf;

use crate::model::{config::WhisperModelParams, decoder::WhisperDecoder, encoder::WhisperEncoder};
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
    pub fn new(params: &WhisperModelParams, device: &B::Device) -> Self {
        Self {
            encoder: WhisperEncoder::new(
                params.num_mel_bins,
                params.d_model,
                params.num_heads,
                params.encoder_layers,
                device,
            ),
            decoder: WhisperDecoder::new(
                params.vocab_size,
                params.d_model,
                params.num_heads,
                params.decoder_layers,
                params.max_target_positions,
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
        params: &WhisperModelParams,
        path: impl Into<PathBuf>,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        let mut model = Self::new(params, device);
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
