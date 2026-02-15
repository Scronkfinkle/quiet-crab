# Quiet Crab - Run Whisper Models in Rust Natively

Quiet Crab is a project to let developers run Whisper transcription models against various audio files in pure Rust.

I was working on a flashcard app but couldn't get Whisper.cpp Rust bindings to build on NixOS so it was easier to rebuilt it.

# Installation
At the moment this project is not publishing a crate. 

To install, add the following to your `Cargo.toml`

```toml
[dependencies]
quiet-crab = { version = "0.1.1"}
# Use wgpu for GPU support, and ndarray for CPU-only
burn = { version = "0.20.1", features = ["wgpu", "ndarray"] }
```

# Quick Start

## Grab A Model And Tokenizer
Get yourself a `tokenizer.json` from https://huggingface.co/openai/whisper-large-v3/blob/main/tokenizer.json

Also grab a medium/small model from huggingface (large is currently unsupported):
* small: https://huggingface.co/openai/whisper-small/blob/main/model.safetensors
* medium: https://huggingface.co/openai/whisper-medium/blob/main/model.safetensors

## Run A Transcription
```rust
use burn::backend::Wgpu;
use quiet_crab::{
    inference::Transcriber,
    model::{
        ModelSize, WhisperModel,
        config::{WhisperConfig, WhisperModelParams},
    },
    tokenizer::whisper_tokenizer::WhisperTokenizer,
};

type Backend = Wgpu;

fn main() -> anyhow::Result<()> {
    let device = Default::default();

    // Specify Options
    let config = WhisperConfig {
        model_params: WhisperModelParams::from_size(ModelSize::Medium),
        no_timestamps: false,
        language: Some(String::from("en")),
    };

    println!("Loading model weights...");
    let model = WhisperModel::<Backend>::from_safetensors(
        &config.model_params,
        "whisper-medium.safetensors",
        &device,
    )?;

    println!("Loading tokenizer...");
    let tokenizer = WhisperTokenizer::from_file("/home/jesse/code/rust/quiet_crab/tokenizer.json")?;

    let transcriber = Transcriber::new(model, tokenizer, config, device);

    println!("Transcribing...");
    let text = transcriber.transcribe("sample.mp3")?;

    println!("{text}");
    Ok(())
}
```

# Model Support

| Model                      | NdArray | Wgpu    |
| -------------------------- | ------  | ------- |
| `whisper-tiny`             | ❓      | ❓      |
| `whisper-small`            | ✅      | ✅      |
| `whisper-medium`           | ✅      | ✅      |
| `whisper-large`            | ✅      | ✅      |
| `whisper-largev2`          | ✅      | ✅      |
| `whisper-largev3`          | ❌      | ❌      |
