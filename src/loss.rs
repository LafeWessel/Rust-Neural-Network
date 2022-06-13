

pub mod loss_fns{
    use core::panic;
    use std::cmp::Ordering;


    pub trait Loss{
        fn loss(&self, predictions: &[f32], targets: &[f32]) -> f32;

        fn loss_derivative(&self, predictions: &[f32], targets: &[f32]) -> Vec<f32>;
    }

    /// Mean Squared Error
    pub struct MSE{}
    impl Loss for MSE{
        /// Mean squared error
        fn loss(&self, predictions: &[f32], targets: &[f32]) -> f32 {
            predictions.iter()
                .zip(targets.iter())
                .map(|(p,t)| (p-t).powi(2))
                .sum::<f32>() / predictions.len() as f32
        }

        /// Derivative of mean squared error
        fn loss_derivative(&self, predictions: &[f32], targets: &[f32]) -> Vec<f32> {
            // 2 * (pred - targ) / targ.size
            predictions.iter()
                .zip(targets.iter())
                .map(|(p,t)| 2.0 * (p-t) / predictions.len() as f32)
                .collect()
        }
    }

    /// Mean Absolute Error
    pub struct MAE{}
    impl Loss for MAE{
        /// Mean absolute error
        fn loss(&self, predictions: &[f32], targets: &[f32]) -> f32 {
            predictions.iter()
                .zip(targets.iter())
                .map(|(p,r)| (p - r).abs())
                .sum::<f32>() / predictions.len() as f32
        }

        /// Derivative of mean absolute error
        fn loss_derivative(&self, predictions: &[f32], targets: &[f32]) -> Vec<f32> {
            predictions.iter().zip(targets.iter()).map(|(p,t)| match p.partial_cmp(t) {
                Some(Ordering::Greater) => 1.0,
                Some(Ordering::Equal) => 0.0,
                Some(Ordering::Less) => -1.0,
                None => panic!("Cannot compare {}, {}", p, t)
            }).collect()
        }
    }

}

#[cfg(test)]
mod tests{
    use crate::loss::loss_fns::{MSE, MAE, Loss};

    #[test]
    /// Test MSE
    fn mse(){
        let a = MSE{};
        let pred = vec![0.0, 1.0, 2.0, 3.0];
        let resp = vec![2.0, 3.0, 4.0, 5.0];
        assert_eq!(4.0, a.loss(&pred, &resp));
    }

    #[test]
    /// Test MAE
    fn mae(){
        let a = MAE{};
        let pred = vec![0.0, 1.0, 2.0, 3.0];
        let resp = vec![2.0, 3.0, 4.0, 5.0];
        assert_eq!(2.0, a.loss(&pred, &resp));
    }

    #[test]
    /// Test MSE derivative
    fn mse_derivative(){
        let a = MSE{};
        let pred = vec![0.0,1.0,2.0, 4.0];
        let resp = vec![1.0, 2.0, 4.0, 4.0];
        assert_eq!(vec![-0.5, -0.5, -1.0, 0.0], a.loss_derivative(&pred, &resp));
    }

    #[test]
    /// Test MAE derivative
    fn mae_derivative(){
        let a = MAE{};
        let pred = vec![0.0, 1.0, 2.0, 4.0];
        let resp = vec![-1.0, 2.0, 3.0, 4.0];

        assert_eq!(vec![1.0, -1.0, -1.0, 0.0], a.loss_derivative(&pred, &resp));
    }
}