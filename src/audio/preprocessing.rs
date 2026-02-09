// Disclaimer I didn't want to figure this out, so I had claude code generate this file
use anyhow::{Context, Result};
use rubato::{FftFixedIn, Resampler};
use std::path::Path;
use symphonia::core::{
    audio::SampleBuffer, codecs::DecoderOptions, formats::FormatOptions, io::MediaSourceStream,
    meta::MetadataOptions, probe::Hint,
};

use crate::audio::constants::SAMPLE_RATE;

/// Load an audio file (MP3, WAV, FLAC, …) and return normalized mono PCM at 16kHz.
///
/// The pipeline is:
///   1. Decode compressed audio via symphonia
///   2. Convert stereo → mono (average channels)
///   3. Resample to 16000 Hz via rubato
///   4. Normalize amplitude to [-1, 1]
///
///
/// Returns a `Vec<f32>` of samples ready for mel spectrogram computation.
pub fn load_audio<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
    // --- Step 1: decode ---
    let (samples, sample_rate, channels) = decode_audio(path)?;

    // --- Step 2: stereo → mono ---
    let mono = to_mono(&samples, channels);

    // --- Step 3: resample to 16kHz ---
    let resampled = if sample_rate != (SAMPLE_RATE as u32) {
        resample(&mono, sample_rate, SAMPLE_RATE as u32)?
    } else {
        mono
    };

    // --- Step 4: normalize ---
    Ok(normalize(resampled))
}

/// Decode any symphonia-supported audio file into interleaved f32 samples.
/// Returns (samples, sample_rate, num_channels).
fn decode_audio<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, u32, usize)> {
    let file = std::fs::File::open(path.as_ref())?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Give symphonia a hint about the file format from the extension
    let mut hint = Hint::new();
    if let Some(ext) = path.as_ref().extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let meta_opts = MetadataOptions::default();
    let fmt_opts = FormatOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .context("unsupported audio format")?;

    let mut format = probed.format;

    // Pick the first audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .context("no audio track found")?;

    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .context("unknown sample rate")?;
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

    let dec_opts = DecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .context("creating decoder")?;

    let mut all_samples: Vec<f32> = Vec::new();

    // Decode packet by packet
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(_)) => break,
            Err(symphonia::core::errors::Error::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                sample_buf.copy_interleaved_ref(decoded);
                all_samples.extend_from_slice(sample_buf.samples());
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        }
    }

    Ok((all_samples, sample_rate, channels))
}

/// Convert interleaved multi-channel samples to mono by averaging channels.
fn to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }
    samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Resample mono audio from `src_rate` to `dst_rate` using a high-quality FFT resampler.
fn resample(samples: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    // Rubato works on chunks; we process in one shot by treating input as one chunk
    let chunk_size = samples.len().max(1);
    let mut resampler =
        FftFixedIn::<f32>::new(src_rate as usize, dst_rate as usize, chunk_size, 2, 1)
            .context("creating resampler")?;

    // Rubato expects Vec<Vec<f32>> (one Vec per channel)
    let input = vec![samples.to_vec()];
    let output = resampler.process(&input, None).context("resampling")?;

    Ok(output.into_iter().next().unwrap_or_default())
}

/// Normalize amplitude: divides by the maximum absolute value (or leaves as-is if silent).
fn normalize(mut samples: Vec<f32>) -> Vec<f32> {
    let max_amp = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if max_amp > 1e-6 {
        samples.iter_mut().for_each(|s| *s /= max_amp);
    }
    samples
}
