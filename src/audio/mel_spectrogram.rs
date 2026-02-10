use anyhow::Result;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;

use crate::audio::constants::{
    F_MAX, F_MIN, FRAME_SIZE, HOP_LENGTH, N_FFT_BINS, N_FRAMES, N_SAMPLES, SAMPLE_RATE,
};

/// Compute log-mel spectrogram from mono 16kHz audio samples.
///
/// Humans hear sound logarithmically. To model this, we use a unit called "mels".
/// Mels convert Hz to logarithmic scale, and is calibrated such that 1000 Mel = 1000 Hz
///
/// Converting the spectrogram to mels allows us to create a perceptually relevant frequency representation
pub fn log_mel_spectrogram(samples: &[f32], n_mels: usize) -> Result<Vec<Vec<f32>>> {
    assert!(samples.len() <= N_SAMPLES);
    let hann = hann_window(FRAME_SIZE);
    let mel_filters = mel_filterbank(FRAME_SIZE, n_mels, SAMPLE_RATE, F_MIN, F_MAX);
    // Pad to 30 seconds
    let mut padded = vec![0.0f32; N_SAMPLES];
    padded[..samples.len()].clone_from_slice(samples);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FRAME_SIZE);

    // Output: [n_mels][n_frames]
    let mut mel_spec = vec![vec![0.0f32; N_FRAMES]; n_mels];

    let mut fft_buf = vec![Complex::new(0.0, 0.0); FRAME_SIZE];

    for frame_idx in 0..N_FRAMES {
        let start = frame_idx * HOP_LENGTH;

        // Apply Hann window to this frame so the signal is periodic
        for i in 0..FRAME_SIZE {
            fft_buf[i] = Complex::new(padded[start + i] * hann[i], 0.0);
        }

        // Apply FFT
        fft.process(&mut fft_buf);

        // for a complex number like z = a + bi
        // We can measure the magnitude (norm) like this
        // sqrt(a^2 + b^2)
        // sqrt is expensive though so instead we take the norm squared which is
        // sqrt(a^2 + b^2)^2 = a^2 + b^2
        // and don't care about the fact that it's not the actualy magnitude
        let power: Vec<f32> = (0..N_FFT_BINS).map(|k| fft_buf[k].norm_sqr()).collect();

        // Apply mel filterbank: each mel bin is a weighted sum of FFT bins
        for mel_idx in 0..n_mels {
            let val: f32 = mel_filters[mel_idx]
                .iter()
                .zip(power.iter())
                .map(|(w, p)| w * p)
                .sum();
            mel_spec[mel_idx][frame_idx] = val;
        }
    }

    // Whisper does this next to make a log mel spec
    // log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    for row in &mut mel_spec {
        for v in row.iter_mut() {
            *v = (*v).max(1e-10).log10();
        }
    }

    // Next it does this:
    // log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    // This finds the loudest point in the sample, and then
    // sets an 80 dB range from that. All smaller background sounds
    // get clipped to the floor which Whisper is trained to ignore
    let global_max = mel_spec
        .iter()
        .flatten()
        .fold(f32::NEG_INFINITY, |arg0, other| arg0.max(*other));
    let floor = global_max - 8.0;
    for row in &mut mel_spec {
        for v in row.iter_mut() {
            *v = v.max(floor);
        }
    }

    // Finally they shift the values around
    // log_spec = (log_spec + 4.0) / 4.0
    for row in &mut mel_spec {
        for v in row.iter_mut() {
            *v = (*v + 4.0) / 4.0;
        }
    }

    Ok(mel_spec)
}

/// Generate a Hann window of the given length.
///
/// When you take a FFT of audio, it assumes that the signal
/// it is decomposing is periodic. That means that the sound sample starts and ends
/// at the same magnitude.
///
/// In practice, taking a ~10ms sample of someone speaking means
/// this never happens. Because of this, we apply a windowing function over the signal, which
/// squishes the signal down to zero at the start and end of the sample. This forces the sample
/// to be periodic, which lets us run the fourier transform over it.
///
/// Running the fourier transform over a non-periodic signal creates "spectral leakage" where
/// it will detect higher frequencies that don't exist
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n as f32 - 1.0)).cos()))
        .collect()
}

/// Convert frequency in Hz to mel scale.
///
/// Formula from: https://en.wikipedia.org/wiki/Mel_scale
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel value back to Hz.
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Build the mel filterbank matrix: shape [n_mels][n_fft_bins].
///
/// When we run the FFT on an audio sample, it measures the magnitude of thousands of different
/// frequencies. This is many more than we need, so we need to essentially turn down the resolution
/// and put all the frequencies into smaller bins.
///
/// Naively we could specify a set number of bins, and for the values in between them portion out the magnitude.
/// For example, if we have a bin at 1000 Hz, another at 2000 Hz, and then a signal at 1750 Hz we could divide the
/// magnitude of that signal and weight it by the distance from each value e.g.
///
/// bin 1 += M*0.25
/// bin 2 += M*0.75
///
/// The problem with that is humans hear sound logarithmically. Therefore, to generate the filters that divide up
/// the frequencies, we first get the highest and lower frequencies of the sample in Hz, and convert to mels.
/// Now that we have a range that's logarithmic, we create evenly spaced bins. Then, we convert the bins back to Hz
/// and we end up with a set of values that appropriately divides the frequencies into their corresponding bins, scaled
/// logarithmically
fn mel_filterbank(
    frame_size: usize,
    n_mels: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
) -> Vec<Vec<f32>> {
    // The number of frequency bins the FFT gives
    // We divide by two because the second half of the signal
    // is redundant (a copy of the first half mirrroed into negative).
    // FFT libs throw out the second half for this reason
    // we then add 1 because we want to encapsulate the range (0, frame_size/2)
    let n_fft_bins = (frame_size / 2) + 1;

    // Evenly spaced mel points from f_min to f_max
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // n_mels + 2 points: one extra on each side for the triangular filters
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert back to Hz
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    // The points we calculated above need to be rounded to the nearest bins that the FFT produced
    // FFT divides evenly, so the size of each bin is (sample_rate/frame_size)
    // thus, hz = (sample_rate/frame_size)*bin
    // we can re-arrange the formula to find the frequency bin like so:
    // bin = hz*(frame_size/sample_rate)
    let bin_points: Vec<f32> = hz_points
        .iter()
        // AI put frame_size+1, but may be a bug
        .map(|&hz| frame_size as f32 * hz / sample_rate)
        .collect();

    // Build triangular filters
    let mut filters = vec![vec![0.0f32; n_fft_bins]; n_mels];
    for m in 0..n_mels {
        // Triangle points
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        for k in 0..n_fft_bins {
            let k = k as f32;
            // Get the ratio of the triangle
            // e.g. if we have left = 1000Hz, center = 2000Hz, and k = 1750
            // (1750-1000)/(2000-1000) = 0.75

            // Left side of the triangle
            if k >= left && k <= center {
                filters[m][k as usize] = (k - left) / (center - left);
            // Right side of the triange
            } else if k > center && k <= right {
                filters[m][k as usize] = (right - k) / (right - center);
            }
        }
    }
    filters
}
