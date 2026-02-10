use std::path::Path;

use anyhow::Result;
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

use crate::{
    audio::{
        constants::{N_FRAMES, N_SAMPLES},
        mel_spectrogram::log_mel_spectrogram,
        preprocessing::load_audio,
    },
    model::{config::WhisperConfig, whisper::WhisperModel},
    tokenizer::whisper_tokenizer::WhisperTokenizer,
};

/// End-to-end transcription pipeline.
pub struct Transcriber<B: Backend> {
    model: WhisperModel<B>,
    tokenizer: WhisperTokenizer,
    config: WhisperConfig,
    device: B::Device,
}

impl<B: Backend> Transcriber<B> {
    pub fn new(
        model: WhisperModel<B>,
        tokenizer: WhisperTokenizer,
        config: WhisperConfig,
        device: B::Device,
    ) -> Self {
        Self {
            model,
            tokenizer,
            config,
            device,
        }
    }

    /// Transcribe an audio file to text.
    ///
    /// Audio is split into non-overlapping 30-second chunks and decoded independently.
    pub fn transcribe<P: AsRef<Path>>(&self, path: P, language: Option<&str>) -> Result<String> {
        let samples = load_audio(path)?;
        let mut parts = Vec::new();
        for chunk in samples.chunks(N_SAMPLES) {
            let text = self.transcribe_chunk(chunk, language)?;
            let trimmed = text.trim().to_string();
            if !trimmed.is_empty() {
                parts.push(trimmed);
            }
        }
        Ok(parts.join(" "))
    }

    fn transcribe_chunk(&self, samples: &[f32], language: Option<&str>) -> Result<String> {
        // Mel spectrogram: [n_mels, n_frames]
        let mel = log_mel_spectrogram(samples, self.config.num_mel_bins)?;
        assert_eq!(mel.len(), self.config.num_mel_bins);
        assert_eq!(mel.first().unwrap().len(), N_FRAMES);

        // Convert to tensor [1, n_mels, n_frames]
        let n_mels = mel.len();
        let flat: Vec<f32> = mel.into_iter().flatten().collect();
        let mel_tensor =
            Tensor::<B, 3>::from_data(TensorData::new(flat, [1, n_mels, N_FRAMES]), &self.device);

        // Encode once; reuse across all decoder steps
        let encoder_out = self.model.encode(mel_tensor);

        // Greedy decode → token ids
        let tokens = self.greedy_decode(encoder_out, language)?;

        // Tokens → text
        self.tokenizer.decode(&tokens)
    }

    /// Greedy (argmax) decoding.
    ///
    /// Feeds the encoder output through the decoder one token at a time,
    /// always picking the highest-probability next token until `<|endoftext|>`.
    fn greedy_decode(&self, encoder_out: Tensor<B, 3>, language: Option<&str>) -> Result<Vec<u32>> {
        let initial = self.tokenizer.initial_tokens(language);
        let prompt_len = initial.len();
        let mut tokens: Vec<i64> = initial.iter().map(|&t| t as i64).collect();

        for _ in 0..(self.config.max_target_positions - prompt_len) {
            let seq_len = tokens.len();
            let token_tensor = Tensor::<B, 2, Int>::from_data(
                TensorData::new(tokens.clone(), [1, seq_len]),
                &self.device,
            );

            // logits: [1, seq_len, vocab_size]
            let logits = self.model.decode(token_tensor, encoder_out.clone());
            let vocab = logits.shape().dims[2];

            // Slice out the last position's logits and find argmax on CPU.
            // Avoids Int element type mismatch across backends (i32 vs i64).
            let last_logits = logits
                .slice([0..1, (seq_len - 1)..seq_len, 0..vocab])
                .into_data();
            let mut vals: Vec<f32> = last_logits.to_vec::<f32>().unwrap();
            // Never predict <|notimestamps|>
            vals[50363] = 0.0;

            let next_tok = vals
                .iter()
                .copied()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
                    if v > bv { (i, v) } else { (bi, bv) }
                })
                .0 as u32;

            if next_tok == self.tokenizer.eot {
                break;
            }
            tokens.push(next_tok as i64);
        }

        // Strip the initial prompt, return only generated tokens
        Ok(tokens[prompt_len..].iter().map(|&t| t as u32).collect())
    }
}
