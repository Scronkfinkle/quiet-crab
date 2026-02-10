/// Different Whisper model sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
    LargeV2,
    LargeV3,
}

/// Configuration for Whisper model architecture
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    /// Model size variant
    pub model_size: ModelSize,

    /// Vocabulary size (number of tokens)
    pub vocab_size: usize,

    /// Number of mel frequency bins
    pub num_mel_bins: usize,

    /// Model dimensionality (hidden size)
    pub d_model: usize,

    /// Number of encoder layers
    pub encoder_layers: usize,

    /// Number of decoder layers
    pub decoder_layers: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Feed-forward network dimension
    pub ffn_dim: usize,

    /// Dropout probability
    pub dropout: f64,

    /// Maximum number of encoder positions (audio frames)
    pub max_source_positions: usize,

    /// Maximum number of decoder positions (text tokens)
    pub max_target_positions: usize,
}

impl WhisperConfig {
    /// Create configuration for Tiny model (39M parameters)
    pub fn tiny() -> Self {
        Self {
            model_size: ModelSize::Tiny,
            vocab_size: 51865,
            num_mel_bins: 80,
            d_model: 384,
            encoder_layers: 4,
            decoder_layers: 4,
            num_heads: 6,
            ffn_dim: 1536,
            dropout: 0.0,
            max_source_positions: 1500,
            max_target_positions: 448,
        }
    }

    /// Create configuration for Base model (74M parameters)
    pub fn base() -> Self {
        Self {
            model_size: ModelSize::Base,
            vocab_size: 51865,
            num_mel_bins: 80,
            d_model: 512,
            encoder_layers: 6,
            decoder_layers: 6,
            num_heads: 8,
            ffn_dim: 2048,
            dropout: 0.0,
            max_source_positions: 1500,
            max_target_positions: 448,
        }
    }

    /// Create configuration for Small model (244M parameters)
    pub fn small() -> Self {
        Self {
            model_size: ModelSize::Small,
            vocab_size: 51865,
            num_mel_bins: 80,
            d_model: 768,
            encoder_layers: 12,
            decoder_layers: 12,
            num_heads: 12,
            ffn_dim: 3072,
            dropout: 0.0,
            max_source_positions: 1500,
            max_target_positions: 448,
        }
    }

    /// Create configuration for Medium model (769M parameters)
    pub fn medium() -> Self {
        Self {
            model_size: ModelSize::Medium,
            vocab_size: 51865,
            num_mel_bins: 80,
            d_model: 1024,
            encoder_layers: 24,
            decoder_layers: 24,
            num_heads: 16,
            ffn_dim: 4096,
            dropout: 0.0,
            max_source_positions: 1500,
            max_target_positions: 448,
        }
    }

    pub fn large() -> Self {
        Self {
            model_size: ModelSize::Large,
            vocab_size: 51865,
            num_mel_bins: 80,
            d_model: 1280,
            encoder_layers: 32,
            decoder_layers: 32,
            num_heads: 20,
            ffn_dim: 5120,
            dropout: 0.0,
            max_source_positions: 1500,
            max_target_positions: 448,
        }
    }

    /// Create configuration for Large model (1550M parameters)
    ///
    /// Note: The HuggingFace whisper-large checkpoint uses 128 mel bins (vs 80
    /// for smaller models) and a vocab of 51866 (51865 + 1 padding token).
    pub fn large_v3() -> Self {
        Self {
            model_size: ModelSize::LargeV3,
            vocab_size: 51866,
            num_mel_bins: 128,
            d_model: 1280,
            encoder_layers: 32,
            decoder_layers: 32,
            num_heads: 20,
            ffn_dim: 5120,
            dropout: 0.0,
            max_source_positions: 1500,
            max_target_positions: 448,
        }
    }

    pub fn from_size(size: ModelSize) -> Self {
        match size {
            ModelSize::Base => Self::base(),
            ModelSize::Tiny => Self::tiny(),
            ModelSize::Small => Self::small(),
            ModelSize::Medium => Self::medium(),
            ModelSize::Large | ModelSize::LargeV2 => Self::large(),
            ModelSize::LargeV3 => Self::large_v3(),
        }
    }

    /// Get the dimension of each attention head
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }
}
