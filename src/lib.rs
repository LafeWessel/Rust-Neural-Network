
/*

TODO:
- create linear algebra logic
- create different activations functions
- create different optimizer functions
- create FC layers
- create Conv layers

 */

pub mod neural_net{

    pub struct Network{
        history: History
    }

    impl Network{
        fn train(){

        }

        fn predict(){

        }

        fn score(){

        }

        fn add_layer(){

        }

        fn compile(){

        }
    }

    pub struct History{
        loss: Vec<f32>,
        true_pos: Vec<u64>,
        true_neg: Vec<u64>,
        false_pos: Vec<u64>,
        false_neg: Vec<u64>,
    }


}

pub mod metrics{
    
    pub trait Metric{
        fn calculate_metric(true_pos: u64, true_neg: u64, false_pos:u64, false_neg: u64) -> f32;
    }
    
    pub struct Precision{}
    impl Metric for Precision{
        /// Precision = TP / (TP + FP)
        fn calculate_metric(true_pos: u64, _true_neg: u64, false_pos: u64, _false_neg: u64) -> f32 {
            (true_pos as f32 / (true_pos + false_pos) as f32) as f32
        }
    }
    
    pub struct Recall{}
    impl Metric for Recall{
        /// Recall = TP / (TP + FN)
        fn calculate_metric(true_pos: u64, _true_neg: u64, _false_pos: u64, false_neg: u64) -> f32 {
            (true_pos as f32 / (true_pos + false_neg) as f32) as f32
        }
    }
    
    pub struct Accuracy{}
    impl Metric for Accuracy{
        /// Accuracy = (TP + TN) / (TP + TN + FP + FN)
        fn calculate_metric(true_pos: u64, true_neg: u64, false_pos: u64, false_neg: u64) -> f32 {
            ((true_neg + true_pos) as f32 / (true_neg + true_pos + false_neg + false_pos) as f32) as f32
        }
    }

}

pub mod loss_fns{

    pub trait Loss{
        fn calculate_loss(predictions: &Vec<f32>, targets: &Vec<f32>) -> f32;
    }

    /// Mean Squared Error
    pub struct MSE{}
    impl Loss for MSE{
        /// Mean squared error
        fn calculate_loss(predictions: &Vec<f32>, targets: &Vec<f32>) -> f32 {

            assert_eq!(predictions.len(), targets.len(), "Predictions and Responses are not the same length: {} =/= {}", predictions.len(), targets.len());
            predictions.iter()
                .zip(targets.iter())
                .map(|(p,r)| (p-r) * (p-r))
                .sum::<f32>() / predictions.len() as f32
        }
    }

    /// Mean Absolute Error
    pub struct MAE{}
    impl Loss for MAE{
        /// Mean absolute error
        fn calculate_loss(predictions: &Vec<f32>, targets: &Vec<f32>) -> f32 {
            assert_eq!(predictions.len(), targets.len(), "Predictions and Responses are not the same length: {} =/= {}", predictions.len(), targets.len());
            predictions.iter()
                .zip(targets.iter())
                .map(|(p,r)| (p - r).abs())
                .sum::<f32>() / predictions.len() as f32
        }
    }

}

mod lin_alg{


}

pub mod activation_fns{

    pub trait Activate{
        fn activate(&self, x: f32) -> f32;
    }

    pub struct Relu {}
    impl Activate for Relu {
        /// max(0, x)
        fn activate(&self, x: f32) -> f32 {
            f32::max(0f32,x)
        }
    }

    pub struct Sigmoid{}
    impl Activate for Sigmoid{
        /// 1 / (1 + e^-x)
        fn activate(&self, x: f32) -> f32 {
            1f32 / (1f32 + ((-1f32 * x).exp()))
        }
    }

    pub struct Tanh{}
    impl Activate for Tanh{
        /// tanh(x)
        fn activate(&self, x: f32) -> f32 {
            f32::tanh(x)
        }
    }

}

mod optimizers{

}

mod layers{
    use crate::activation_fns::{Activate};
    use std::ops::Deref;
    use std::borrow::Borrow;

    pub trait Layer{
        fn forward_propagate(&self, prev_layer: &Vec<f32>) -> Vec<f32>;

        fn back_propagate(&self, next_layer: &Vec<f32>);
    }

    pub struct ActivationLayer<'a, T> where T: Activate {
        pub func: &'a T
    }

    impl<'a, T> Layer for ActivationLayer<'a, T> where T: Activate{

        /// Apply activation function to all inputs
        fn forward_propagate(&self, prev_layer: &Vec<f32>) -> Vec<f32>{
            prev_layer.iter().map(|i| self.func.activate(*i)).collect()
        }

        /// Do nothing
        fn back_propagate(&self, next_layer: &Vec<f32>) {

        }
    }

    pub struct FCLayer{
        pub weights: Vec<Vec<f32>>, // Vec[neuron][inputs]
        pub biases: Vec<f32>,

    }
    impl Layer for FCLayer{

        fn forward_propagate(&self, prev_layer: &Vec<f32>) -> Vec<f32>{

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

        fn back_propagate(&self, next_layer: &Vec<f32>) {
            todo!()
        }
    }

}

#[cfg(test)]
mod tests{
    use crate::activation_fns::{Relu, Sigmoid, Activate};
    use crate::metrics::{Accuracy, Precision, Recall, Metric};
    use crate::loss_fns::{MSE, MAE, Loss};
    use crate::layers::{ActivationLayer, Layer, FCLayer};

    #[test]
    /// Rest Sigmoid activation function
    fn sigmoid(){
        let s = Sigmoid{};

        assert_eq!(0.006692851, s.activate(-5f32));
        assert_eq!(0.047425874, s.activate(-3f32));
        assert_eq!(0.5, s.activate(0f32));
        assert_eq!(0.95257413, s.activate(3f32));
        assert_eq!(0.9933072, s.activate(5f32));

    }

    #[test]
    /// Test ReLU activation function
    fn relu(){
        let r = Relu{};
        assert_eq!(0f32, r.activate(-1f32));
        assert_eq!(1f32, r.activate(1f32));
        assert_eq!(0f32, r.activate(0f32));
    }

    #[test]
    /// Test Accuracy metric
    fn accuracy(){

        assert_eq!(0.5f32, Accuracy::calculate_metric(5, 5, 5, 5));
        assert_eq!(0.25f32, Accuracy::calculate_metric(5, 0, 10, 5));
        assert_eq!(1f32, Accuracy::calculate_metric(10, 10, 0, 0));
    }

    #[test]
    /// Test Recall metric
    fn recall(){
        assert_eq!(0.5f32, Recall::calculate_metric(5, 0, 0, 5));
        assert_eq!(1f32, Recall::calculate_metric(10, 0, 0, 0));
        assert_eq!(0f32, Recall::calculate_metric(0, 0, 0, 10));

    }

    #[test]
    /// Test Precision metric
    fn precision(){
        assert_eq!(0.5f32, Precision::calculate_metric(5, 0, 5, 0));
        assert_eq!(1f32, Precision::calculate_metric(5, 0, 0, 0));
        assert_eq!(0f32, Precision::calculate_metric(0, 0, 5, 0));
    }

    #[test]
    /// Test MSE
    fn mse(){
        let pred = vec![0.0, 1.0, 2.0, 3.0];
        let resp = vec![2.0, 3.0, 4.0, 5.0];
        assert_eq!(4f32, MSE::calculate_loss(&pred, &resp));
    }

    #[test]
    /// Test MAE
    fn mae(){
        let pred = vec![0.0, 1.0, 2.0, 3.0];
        let resp = vec![2.0, 3.0, 4.0, 5.0];
        assert_eq!(2f32, MAE::calculate_loss(&pred, &resp));
    }

    #[test]
    /// Test Activation Layer forward propagation
    fn activation_layer_forward(){
        // use Relu for simple activation function
        let a = Relu{};
        let l = ActivationLayer{
            func: &a
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

        let fc = FCLayer{
            weights: w,
            biases: b
        };

        let res = fc.forward_propagate(&i);
        // neuron 1 output = (0.5 * 0) + (0.75 * 1) + (1 * 2) + 0.5 = 3.25
        // neuron 2 output = (0 * 0) + (0.25 * 1) + (0.5 * 2) + 0.25 = 1.5
        assert_eq!(vec![3.25, 1.5], res);

    }

}