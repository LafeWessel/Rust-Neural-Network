
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
        pub metrics: Vec<Box<dyn Metric>>,
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



