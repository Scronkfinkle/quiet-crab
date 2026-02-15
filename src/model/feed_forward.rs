use burn::{
    module::Module,
    nn::{Gelu, Linear, LinearConfig},
    tensor::{Tensor, backend::Backend},
};

/// Feed-forward network used in transformer blocks
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    /// First linear layer: d_model -> ffn_dim
    fc1: Linear<B>,
    /// GELU activation function
    activation: Gelu,
    /// Second linear layer: ffn_dim -> d_model
    fc2: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    /// Create a new feed-forward network
    pub fn new(d_model: usize, ffn_dim: usize, device: &B::Device) -> Self {
        let fc1 = LinearConfig::new(d_model, ffn_dim).init(device);
        let fc2 = LinearConfig::new(ffn_dim, d_model).init(device);
        let activation = Gelu::new();

        Self {
            fc1,
            activation,
            fc2,
        }
    }

    /// Forward pass through the feed-forward network
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // First linear layer
        let x = self.fc1.forward(x);

        // GELU activation
        let x = self.activation.forward(x);

        // Second linear layer
        self.fc2.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_feed_forward_shape() {
        let device = Default::default();
        let ffn = FeedForward::<TestBackend>::new(384, 1536, &device);

        // Create input: [batch=2, seq_len=10, d_model=384]
        let input = Tensor::<TestBackend, 3>::random(
            [2, 10, 384],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = ffn.forward(input.clone());

        // Output should have same shape as input
        assert_eq!(output.shape().dims, [2, 10, 384]);
    }
}
