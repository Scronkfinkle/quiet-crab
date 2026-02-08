use burn::backend::Wgpu;
use burn::backend::{NdArray, ndarray::NdArrayDevice};
use quiet_crab::{
    inference::Transcriber,
    model::{config::WhisperConfig, whisper::WhisperModel},
    tokenizer::whisper_tokenizer::WhisperTokenizer,
};

//type Backend = NdArray<f32>;
type Backend = Wgpu;

fn main() -> anyhow::Result<()> {
    let device = Default::default();
    let config = WhisperConfig::medium();

    println!("Loading model weights...");
    let model =
        WhisperModel::<Backend>::from_safetensors(&config, "whisper-medium.safetensors", &device)?;

    println!("Loading tokenizer...");
    let tokenizer = WhisperTokenizer::from_file("tokenizer.json")?;

    let transcriber = Transcriber::new(model, tokenizer, config, device);

    println!("Transcribing...");
    let text = transcriber.transcribe("enamoré.mp3", Some("es"))?;

    println!("{text}");
    Ok(())
}
