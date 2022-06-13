

mod activations;
mod loss;
mod layers;

pub mod neural_net{
    use crate::layers::layers::Layer;
    use crate::loss::loss_fns::Loss;

    pub struct Network{
        history: History,
        layers: Vec<Box<dyn Layer>>,
        loss_fn: Box<dyn Loss>,
        learning_rate: f32,
    }

    impl Network{
        pub fn new(loss_fn: Box<dyn Loss>, lr: f32) -> Self{
            Network { history: History::new(), layers: vec![], loss_fn: loss_fn, learning_rate: lr }
        }


        pub fn add_layer(&mut self, layer: Box<dyn Layer>){
            self.layers.push(layer);
        }

        /// Fit the model to X and y for a specified number of epochs
        pub fn fit(&mut self, x: &Vec<Vec<f32>>, y: &Vec<Vec<f32>>, epochs: u32){

            // for each sample for each epoch
            for i in 1..=epochs{
                let mut tot_loss = 0.0;
                for (xi, yi) in x.iter().zip(y.iter()){
                    // forward propagate
                    let preds = self.forward_propagate(&xi);
        
                    // calculate metrics and loss
                    let loss = self.loss_fn.loss(&preds, &yi);
                    tot_loss += loss;
                    
                    
                    // backward propagate
                    self.backwards_propagate(&preds, &yi);
                }
                self.history.save_epoch(tot_loss/x.len() as f32);
                println!("Epoch {}/{}: avg loss={}",i, epochs, tot_loss/x.len() as f32);
            }

        }
        
        /// Make predictions based on input X
        pub fn predict(&mut self, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>>{
            x.iter().map(|xi| self.forward_propagate(&xi)).collect()
        }

        /// Get fit()/train history
        pub fn get_history(&self) -> &History{
            &self.history
        }

        /// Forward propagate through network
        fn forward_propagate(&mut self, x: &Vec<f32>) -> Vec<f32>{
            let mut t = x.clone();
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

        pub fn new() -> Self{
            History { loss: vec![], epochs: 0 }
        }

        /// Save an epoch's values
        pub fn save_epoch(&mut self, loss: f32){
            self.epochs += 1;
            self.loss.push(loss);
        }

    }

}

#[cfg(test)]
mod tests{
    use crate::{neural_net::Network, loss::loss_fns::MSE, layers::layers::{FCLayer, ActivationLayer}, activations::activation_fns::Relu};

    #[test]
    /// test network forward propagation
    fn network_forward(){

    }

    #[test]
    /// test network backprop
    fn network_backprop(){

    }

    #[test]
    /// test network predict
    fn network_predict(){

    }

    #[test]
    /// test network fit
    fn network_fit(){

    }

    #[test]
    /// test XOR network example
    fn test_network_xor(){

        // Arrange
        // create base network
        let loss = MSE{};
        let mut network = Network::new(Box::new(loss), 0.005);
        
        // create layers
        // first layer
        let fc1 = FCLayer::new(2, 2);
        let relu1 = Relu{};
        let ac1 = ActivationLayer::new(relu1);
        network.add_layer(Box::new(fc1));
        network.add_layer(Box::new(ac1));

        // second layer
        let fc2 = FCLayer::new(2, 2);
        let relu2 = Relu{};
        let ac2 = ActivationLayer::new(relu2);
        network.add_layer(Box::new(fc2));
        network.add_layer(Box::new(ac2));

        // output layer
        let out = FCLayer::new(2, 1);
        network.add_layer(Box::new(out));


        // create X,y
        let x = vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 0.0], vec![0.0, 1.0]];
        let y = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        // Act
        network.fit(&x, &y, 100);

        // Assert
        // assert that loss decreased as network progressed
        assert!(network.get_history().loss[0] > network.get_history().loss[100-1]);
        

    }
}

