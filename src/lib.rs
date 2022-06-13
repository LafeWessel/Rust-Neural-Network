
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
mod layers;

pub mod neural_net{
    use crate::activations::activation_fns::Activate;
    use crate::metrics::metrics::Metric;
    use crate::layers::layers::Layer;
    use crate::loss::loss_fns::Loss;

    pub struct Network{
        history: History,
        layers: Vec<Box<dyn Layer>>,
        loss_fn: Box<dyn Loss>,
        learning_rate: f32,
    }

    impl Network{
        pub fn add_layer(&mut self, layer: Box<dyn Layer>){
            self.layers.push(layer);
        }

        /// Fit the model to X and y for a specified number of epochs
        pub fn fit(&mut self, X: &Vec<Vec<f32>>, y: &Vec<Vec<f32>>, epochs: u32){

            // for each sample for each epoch
            for i in 1..=epochs{
                for (xi, yi) in X.iter().zip(y.iter()){
                    // forward propagate
                    let preds = self.forward_propagate(&xi);
        
                    // calculate metrics and loss
                    let loss = 
        
                    
                    // backward propagate


                }
            }

        }
        
        /// Make predictions based on input X
        pub fn predict(&mut self, X: &Vec<Vec<f32>>) -> Vec<f32>{
            X.iter().map(|x| self.forward_propagate(&x)).collect()
        }

        /// Get fit()/train history
        pub fn get_history(&self) -> &History{
            &self.history
        }

        /// Forward propagate through network
        fn forward_propagate(&mut self, X: &Vec<f32>) -> Vec<f32>{
            let mut t = X.clone();
            for layer in &mut self.layers.iter_mut(){
                t = layer.forward_propagate(&t);
            }
            t
        }

        /// Backward propagate through network
        fn backwards_propagate(&mut self, predictions: &[f32], targets: &[f32]){
            let mut loss = self.loss_fn.loss_derivative(predictions, targets);
            for layer in &mut self.layers.iter_mut().rev(){
                loss = layer.back_propagate(&loss, self.learning_rate)
            }
        }

    }


    pub struct History{
        pub loss: Vec<f32>,
        pub epochs: u64,
    }

    impl History{

        /// Save an epoch's values
        pub fn save_epoch(&mut self, loss: f32){
            self.epochs += 1;
            self.loss.push(loss);
        }

    }

}



