use anyhow::{Result, anyhow};
use std::path::Path;
use tokenizers::Tokenizer;

use crate::model::config::WhisperConfig;

/// Wrapper around the HuggingFace BPE tokenizer with Whisper-specific helpers.
pub struct WhisperTokenizer {
    tokenizer: Tokenizer,
    pub sot: u32,           // <|startoftranscript|>
    pub eot: u32,           // <|endoftext|>
    pub transcribe: u32,    // <|transcribe|>
    pub translate: u32,     // <|translate|>
    pub no_speech: u32,     // <|nospeech|>
    pub no_timestamps: u32, // <|notimestamps|>
    pub begin_time: u32,    // <|0.00|>
}

impl WhisperTokenizer {
    /// Load tokenizer from a `tokenizer.json` file (HuggingFace format).
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(path).map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;

        // Look up each special token by its string representation
        let lookup = |name: &str| -> Result<u32> {
            tokenizer
                .token_to_id(name)
                .ok_or_else(|| anyhow!("special token not found: {name}"))
        };

        // Older Whisper checkpoints use <|nocaptions|>; newer use <|nospeech|>
        let no_speech = lookup("<|nospeech|>").or_else(|_| lookup("<|nocaptions|>"))?;

        Ok(Self {
            sot: lookup("<|startoftranscript|>")?,
            eot: lookup("<|endoftext|>")?,
            transcribe: lookup("<|transcribe|>")?,
            translate: lookup("<|translate|>")?,
            no_speech,
            no_timestamps: lookup("<|notimestamps|>")?,
            begin_time: lookup("<|0.00|>")?,
            tokenizer,
        })
    }

    /// Get the token ID for a language code (e.g. `"es"` for Spanish).
    pub fn language_token(&self, lang: &str) -> Option<u32> {
        self.tokenizer.token_to_id(&format!("<|{lang}|>"))
    }

    /// Build the initial decoder prompt tokens.
    ///
    /// For Spanish transcription without timestamps:
    ///   `[<|startoftranscript|>, <|es|>, <|transcribe|>, <|notimestamps|>]`
    pub fn initial_tokens(&self, config: &WhisperConfig) -> Vec<u32> {
        let mut tokens = vec![self.sot];
        if let Some(lang) = &config.language
            && let Some(id) = self.language_token(lang)
        {
            tokens.push(id);
        }
        tokens.push(self.transcribe);
        if config.no_timestamps {
            tokens.push(self.no_timestamps);
        } else {
            tokens.push(self.begin_time);
        }
        tokens
    }

    /// Encode text into token IDs (without adding special tokens automatically).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("encode error: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back to text, skipping special tokens.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow!("decode error: {e}"))
    }

    /// Decode a single token ID to its string (useful for debugging).
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }
}
