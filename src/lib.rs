
/*

TODO:

- create different optimizer functions
- create FC layer
- create Conv2d layer
- Use slices instead of Vec in forward, back, activation, and loss fns
- Implement Cross Entropy
- Create Dropout layer
- Implement regularization (L1, L2)
- use Vec<Vec<f32>> in loss fn instead of Vec<f32> to be able to easily implement cross entropy or other loss fn

 */

mod metrics;
mod activations;
mod loss;

pub mod neural_net{
    use crate::activations::activation_fns::Activate;
    use crate::metrics::metrics::Metric;

    pub struct Network{
        history: History,
    }


    pub struct History{
        pub loss: Vec<f32>,
        pub true_pos: Vec<u64>,
        pub true_neg: Vec<u64>,
        pub false_pos: Vec<u64>,
        pub false_neg: Vec<u64>,
        pub epochs: u64,
        pub metrics: Vec<Box<Metric>>,
    }

    impl History{

        /// Save an epoch's values
        pub fn save_epoch(&mut self, true_pos: u64, true_neg: u64, false_pos: u64, false_neg: u64, loss: f32){
            self.epochs += 1;
            self.true_pos.push(true_pos);
            self.true_neg.push(true_neg);
            self.false_pos.push(false_pos);
            self.false_neg.push(false_neg);
            self.loss.push(loss);

            for m in &mut self.metrics{
                m.calculate_and_save_metric(true_pos, true_neg, false_pos, false_neg);
            }
        }

    }

}


mod layers{
    use crate::activations::activation_fns::Activate;

    pub trait Layer{
        fn forward_propagate(&mut self, prev_layer: &[f32]) -> Vec<f32>;

        fn back_propagate(&mut self, output_error: &Vec<Vec<f32>>, learning_rate: f32) -> Vec<Vec<f32>>;
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
        fn back_propagate(&mut self, output_error: &Vec<Vec<f32>>, learning_rate: f32) -> Vec<Vec<f32>>{
            self.input_data.iter()
                .map(|i| self.func.derivative(*i))
                .zip(output_error.iter())
                .map(|(v,e)| v * e)
                .collect()
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

        /// calculate
        fn back_propagate(&mut self, output_error: &Vec<Vec<f32>>, learning_rate: f32) -> Vec<Vec<f32>>{
            // calculate weight error with dot product of input * output error
            // for each input, multiply by each of the outputs
            let weight_errors: Vec<Vec<f32>> = self.input_data.iter()
                .map(|i| output_error.iter()
                    .map(|e| e * i)
                    .collect())
                .collect();

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

            // calculate and return input error = dot product of output error and weights
            vec![]
        }
    }

}

#[cfg(test)]
mod tests{
    use crate::layers::{ActivationLayer, Layer, FCLayer};
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

}