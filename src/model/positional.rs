use burn::tensor::{Tensor, backend::Backend};

/// Compute sinusoidal positional embeddings used by the Whisper encoder.
///
/// Creates a square matrix that can be added element-wise to another tensor
/// to encode the positions of each element
pub fn sinusoids<B: Backend>(length: usize, channels: usize, device: &B::Device) -> Tensor<B, 2> {
    assert!(
        channels.is_multiple_of(2),
        "channels must be even, got {channels}"
    );
    assert_ne!(channels, 2);

    let half = channels / 2;
    // How different the frequency of each subsequent sinusoidal wave will be
    let log_timescale_increment = 10000.0_f64.ln() / (half as f64 - 1.0);

    // Create a flat vector to build the positional encodings
    let mut data = vec![0.0f32; length * channels];
    for pos in 0..length {
        for i in 0..half {
            // Generates the frequency for the wave
            let inv_timescale = (-log_timescale_increment * i as f64).exp();
            // Generate the value of the wave at position pos in time
            let angle = (pos as f64 * inv_timescale) as f32;
            // The first half of the channels are sine waves, the second are the cosine waves
            // sin goes into the first half of the channel dimension
            data[pos * channels + i] = angle.sin();
            // cos goes into the second half
            data[pos * channels + half + i] = angle.cos();
        }
    }

    // Once all the positional embeddings are created, we reshape to fit the dimensions of
    // the model
    Tensor::from_data(
        burn::tensor::TensorData::new(data, [length, channels]),
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_sinusoids_shape() {
        let device = Default::default();
        let emb = sinusoids::<TestBackend>(1500, 512, &device);
        assert_eq!(emb.shape().dims, [1500, 512]);
    }

    #[test]
    fn test_sinusoids_first_position_is_zero_sin() {
        // At pos=0: sin(0 * anything) = 0, cos(0 * anything) = 1
        let device = Default::default();
        let emb = sinusoids::<TestBackend>(10, 4, &device);
        let data = emb.to_data();
        let vals = data.to_vec::<f32>().unwrap();

        // Position 0: [sin(0), sin(0), cos(0), cos(0)] = [0, 0, 1, 1]
        assert!((vals[0] - 0.0).abs() < 1e-6, "sin at pos=0 should be 0");
        assert!((vals[1] - 0.0).abs() < 1e-6, "sin at pos=0 should be 0");
        assert!((vals[2] - 1.0).abs() < 1e-6, "cos at pos=0 should be 1");
        assert!((vals[3] - 1.0).abs() < 1e-6, "cos at pos=0 should be 1");
    }
}
