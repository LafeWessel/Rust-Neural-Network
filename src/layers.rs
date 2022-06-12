pub mod layers{
    use std::vec;

    use crate::activations::activation_fns::Activate;

    pub trait Layer{
        fn forward_propagate(&mut self, prev_layer: &[f32]) -> Vec<f32>;

        fn back_propagate(&mut self, output_error: &[f32], learning_rate: f32) -> Vec<f32>;

    }

    pub struct ActivationLayer<'a, T> where T: Activate {
        pub func: &'a T,
        pub input_data: Vec<f32>
    }

    impl<'a, T> Layer for ActivationLayer<'a, T> where T: Activate{

        /// Apply activation function to all inputs
        fn forward_propagate(&mut self, prev_layer: &[f32]) -> Vec<f32>{
            self.input_data = prev_layer.to_vec();
            prev_layer.iter().map(|i| self.func.activate(*i)).collect()
        }

        /// Apply activation function derivative to input * output error
        fn back_propagate(&mut self, output_error: &[f32], learning_rate: f32) -> Vec<f32>{
            self.input_data.iter()
                .map(|i| self.func.derivative(*i))
                .zip(output_error.iter())
                .map(|(v,e)| e * v)
                .collect()
        }
        
    }

    impl<'a, T> ActivationLayer<'a,T> where T: Activate{
        pub fn new(func: &'a T) -> Self
        where T: Activate
        {
            ActivationLayer { func: func, input_data: vec![] }
        }
    }

    pub struct FCLayer{
        pub weights: Vec<Vec<f32>>, // Vec[neuron][inputs]
        pub biases: Vec<f32>,
        pub input_data: Vec<f32>
    }
    impl Layer for FCLayer{

        /// Forward propagate values
        fn forward_propagate(&mut self, prev_layer: &[f32]) -> Vec<f32>{
            self.input_data = prev_layer.to_vec();
            // calculate dot products
            // For each neuron, multiply the values of the input neurons times the weights for this neuron
            // calculate sum of weight * each previous connection
            // add bias
            self.weights.iter()
                .map(|v|
                    v.iter()
                    .zip(prev_layer.iter())
                    .fold(0f32,|p, (cw, ci)| p + (cw * ci)))
                .zip(self.biases.iter())
                .map(|(w, b)| w + b)
                .collect()
            }
            
            /// calculate backpropagation
            fn back_propagate(&mut self, output_error: &[f32], learning_rate: f32) -> Vec<f32>{
                // calculate and return input error product of output error and weights
                // for each neuron, multiply each weight by its corresponding output error
                // should have same length as number of inputs
                let input_errors : Vec<f32> = (0..self.input_data.len())
                    .into_iter()
                    .map(|i| 
                        self.weights.iter()
                            .zip(output_error.iter())
                            .map(|(w,o)| w[i] * o)
                            .sum::<f32>())
                        .collect();

                // calculate weight error with dot product of input * output error
                // for each input, multiply by each of the outputs
                // should be same size as weights
                let weight_errors: Vec<Vec<f32>> = output_error.iter()
                    .map(|i| self.input_data.iter()
                        .map(|e| e * i).collect()).collect();

                // adjust weights with -= lr * weight error
                self.weights = self.weights.iter()
                    .zip(weight_errors.iter())
                    .map(|(n,we)| n.iter()
                        .zip(we.iter())
                        .map(|(nw, nwe)| nw - (learning_rate * nwe))
                        .collect())
                    .collect();

                // adjust biases with -= lr * output error
                self.biases = self.biases.iter()
                    .zip(output_error.iter())
                    .map(|(b,e)| b - (learning_rate * e))
                    .collect();

            input_errors
        }
    }

    impl FCLayer{
        /// Create a new FCLayer, neurons is the same as output size, optionally initialize bias for all neurons
        pub fn new(input_size: usize, neurons: usize) -> Self{

            // create random weights
            let mut w = vec![];
            for i in 0..neurons{
                w.push(vec![]);
                for k in 0..input_size{
                    // TODO convert to random
                    w[i].push(0.0);
                }
            }

            // init biases
            let mut b = vec![];
            for i in 0..neurons{
                // TODO convert to random
                b.push(0.0);
            }

            FCLayer { weights: w, biases: b, input_data: vec![] }
        }
    }

}

#[cfg(test)]
mod tests{
    use crate::layers::layers::{ActivationLayer, Layer, FCLayer};
    use crate::activations::activation_fns::Relu;


    #[test]
    /// Test Activation Layer forward propagation
    fn activation_layer_forward(){
        // use Relu for simple activation function
        let a = Relu{};
        let mut l = ActivationLayer{
            func: &a,
            input_data: vec![]
        };

        let forward: Vec<f32> = vec![1.0, 0.0, -2.0];
        let res = l.forward_propagate(&forward);
        assert_eq!(vec![1.0, 0.0, 0.0], res);
    }

    #[test]
    /// Test FCLayer forward propagation
    fn fc_layer_forward(){
        // three inputs
        let i : Vec<f32> = vec![0.0, 1.0, 2.0];
        // two neurons
        let w : Vec<Vec<f32>> = vec![vec![0.5, 0.75, 1.0], vec![0.0, 0.25, 0.5]];
        // biases
        let b : Vec<f32> = vec![0.5, 0.25];

        let mut fc = FCLayer{
            weights: w,
            biases: b,
            input_data: vec![]
        };

        let res = fc.forward_propagate(&i);
        // neuron 1 output = (0.5 * 0) + (0.75 * 1) + (1 * 2) + 0.5 = 3.25
        // neuron 2 output = (0 * 0) + (0.25 * 1) + (0.5 * 2) + 0.25 = 1.5
        assert_eq!(vec![3.25, 1.5], res);

    }

    #[test]
    fn activation_layer_backprop(){
        // Arrange
        let output_error = vec![0.0, 1.0, 2.0];
        // learning rate, not actually used for activation function backprop
        let lr : f32 = 1.0;

        let func = Relu{};
        let mut l = ActivationLayer{
            func: &func,
            input_data: vec![1.0, 1.0, 1.0],
        };

        // Act
        let res = l.back_propagate(&output_error, lr);

        // Assert
        // should be Relu derivative of input data * output error
        assert_eq!(vec![0.0, 1.0, 2.0], res);

    }

    #[test]
    fn fc_layer_backprop(){
        // Arrange
        // learning rate
        let lr: f32 = 1.0;
        // output error
        let output_error = vec![0.1, 0.5, 1.0];
        // layer, simulating 2 inputs to 3 neurons
        let mut l = FCLayer{
            weights: vec![vec![0.0, 1.0], vec![2.0, 3.0], vec![4.0, 5.0]],
            biases: vec![1.0, 1.0, 1.0],
            input_data: vec![0.5, 1.0]
        };

        // Act
        let res = l.back_propagate(&output_error, lr);

        // Assert
        // weight errors should be [[0.05, 0.1], [0.25, 0.5], [0.5, 1.0]]
        // since learning rate is 1.0, the resulting weights should be [[-0.05, 0.9], [1.75, 2.5], [3.5, 4.0]]
        assert_eq!(vec![vec![-0.05, 0.9], vec![1.75, 2.5], vec![3.5, 4.0]], l.weights);

        // bias errors are the output errors
        // since the learning rate is 1.0, the resulting biases should be [0.9, 0.5, 0.0]
        assert_eq!(vec![0.9, 0.5, 0.0], l.biases);

        // results should be output_error * weights and the same length as the inputs
        // sum of first of each "neuron" weight * corresponding output error
        // thus, should be [(0 * 0.1 + 2 * 0.5 + 4 * 1.0), (1 * 0.1 + 3 * 0.5 + 5 * 1)]
        // = [(0 + 1 + 4), (0.1 + 1.5 + 5)]
        // = [5, 6.6]
        assert_eq!(vec![5.0, 6.6], res);


    }

}