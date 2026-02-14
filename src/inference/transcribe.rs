use core::f32;
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
    pub fn transcribe<P: AsRef<Path>>(&self, path: P) -> Result<String> {
        let samples = load_audio(path)?;
        let mut parts = Vec::new();
        for chunk in samples.chunks(N_SAMPLES) {
            let text = self.transcribe_chunk(chunk)?;
            let trimmed = text.trim().to_string();
            if !trimmed.is_empty() {
                parts.push(trimmed);
            }
        }
        Ok(parts.join(" "))
    }

    fn transcribe_chunk(&self, samples: &[f32]) -> Result<String> {
        // Mel spectrogram: [n_mels, n_frames]
        let model_params = self.config.model_params;
        let mel = log_mel_spectrogram(samples, model_params.num_mel_bins)?;
        assert_eq!(mel.len(), model_params.num_mel_bins);
        assert_eq!(mel.first().unwrap().len(), N_FRAMES);

        // Convert to tensor [1, n_mels, n_frames]
        let n_mels = mel.len();
        let flat: Vec<f32> = mel.into_iter().flatten().collect();
        let mel_tensor =
            Tensor::<B, 3>::from_data(TensorData::new(flat, [1, n_mels, N_FRAMES]), &self.device);

        // Encode once; reuse across all decoder steps
        let encoder_out = self.model.encode(mel_tensor);

        // Greedy decode → token ids
        let tokens = self.greedy_decode(encoder_out)?;

        // Tokens → text
        self.tokenizer.decode(&tokens)
    }

    fn softmax(x: &[f32]) -> Vec<f32> {
        let e_sum = x.iter().fold(0.0, |acc, val| acc + val.exp());
        x.iter().map(|val| val.exp() / e_sum).collect()
    }

    fn apply_timestamp_rules(&self, vals: &[f32], tokens: &[i64]) -> Vec<f32> {
        let mut new_vals = vals.to_vec();
        let no_timestamps = self.tokenizer.no_timestamps as usize;
        let begin_time = self.tokenizer.begin_time as usize;

        // Never predict <|notimestamps|>
        new_vals[no_timestamps] = f32::NEG_INFINITY;

        // <|notimestamps|> specified, ignore further processing
        if tokens.contains(&(no_timestamps as i64)) {
            return new_vals;
        }

        // If the sum of probabilities over timestamps is more than non-timestamps
        // then always return a timestamp
        let sm_vals = Self::softmax(&new_vals);
        let timestamps = &sm_vals[(begin_time)..];
        let sum_ts = timestamps.iter().sum();
        let ts_prob_is_max = !(&sm_vals[0..begin_time].iter().any(|val| val > &sum_ts));
        if ts_prob_is_max {
            new_vals = new_vals
                .iter()
                .enumerate()
                .map(|(index, val)| {
                    if index < begin_time {
                        f32::NEG_INFINITY
                    } else {
                        *val
                    }
                })
                .collect()
        }

        // Force timestamps to be in pairs
        let last_token = tokens
            .last()
            .expect("There should always be at least a <|startoftranscipt|> token");
        if *last_token >= begin_time as i64 {
            // This should be fine since it would mean that we have both the start token
            // and now a timestamp token
            let last_last_token = tokens[tokens.len() - 2];
            // If not a timestamp token, force timestamp sampling
            if last_last_token < begin_time as i64 {
                new_vals = new_vals
                    .iter()
                    .enumerate()
                    .map(|(index, val)| {
                        if index < begin_time {
                            f32::NEG_INFINITY
                        } else {
                            *val
                        }
                    })
                    .collect();
            }
        }

        // Never let timestamps decrease
        let max_token = tokens.iter().max().unwrap();
        if *max_token >= begin_time as i64 {
            new_vals = new_vals
                .iter()
                .enumerate()
                .map(|(i, val)| {
                    if i >= begin_time && ((i as i64) < *max_token) {
                        f32::NEG_INFINITY
                    } else {
                        *val
                    }
                })
                .collect();
        }
        new_vals
    }

    /// Greedy (argmax) decoding.
    ///
    /// Feeds the encoder output through the decoder one token at a time,
    /// always picking the highest-probability next token until `<|endoftext|>`.
    fn greedy_decode(&self, encoder_out: Tensor<B, 3>) -> Result<Vec<u32>> {
        let initial = self.tokenizer.initial_tokens(&self.config);
        let prompt_len = initial.len();
        let mut tokens: Vec<i64> = initial.iter().map(|&t| t as i64).collect();

        for _ in 0..(self.config.model_params.max_target_positions - prompt_len) {
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
            let mut logits: Vec<f32> = last_logits.to_vec::<f32>().unwrap();
            logits = self.apply_timestamp_rules(&logits, &tokens);

            let next_tok = logits
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

        Ok(tokens.iter().map(|&t| t as u32).collect())
    }
}
