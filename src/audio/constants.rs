// Whisper's STFT parameters
//
// From the Whisper paper:
//    All audio is re-sampled to 16,000 Hz, and an 80-channel
//    log-magnitude Mel spectrogram representation is computed on
//    25-millisecond windows with a stride of 10 milliseconds.

/// 16 KHz sample rate
pub const SAMPLE_RATE: f32 = 16_000.0;
// whisper processes chunks in 30 second intervals
pub const N_SAMPLES: usize = 30 * 16_000;

/// 25ms frame (window) at 16kHz
///
/// This is how many samples to capture per frame
/// before running the fourier transform
///
/// Note: the paper refers to this as the "window" but everywhere
/// I read about this stuff they call it the frame, and use the term
/// "windowing" to refer to a form of signal processing (i.e. hann window)
pub const FRAME_SIZE: usize = 400;

pub const N_FFT_BINS: usize = (FRAME_SIZE / 2) + 1;

/// 10ms stride at 16kHz
///
/// It's not stated in the paper, but we shift a smaller number than the window size
/// likely to ensure overlapping frames. This is useful to restore the data that is
/// trimmed out by the hann windowing function
pub const HOP_LENGTH: usize = 160;

// The number of frames in a sample
// since all samples are 30 seconds, the number of frames is constant
pub const N_FRAMES: usize = ((N_SAMPLES - FRAME_SIZE) / HOP_LENGTH) + 1;

pub const F_MIN: f32 = 0.0;
/// Nyquist for 16kHz (just half the sample rate, 8KHz)
///
/// The Nyquist frequency is the maxmimum frequency of a signal
/// that can be played without aliasing. I don't entirely understand
/// aliasing, other than the frequencies above the Nyquist sound like garbage.
///
/// See: https://en.wikipedia.org/wiki/Nyquist_frequency
pub const F_MAX: f32 = 8_000.0;
