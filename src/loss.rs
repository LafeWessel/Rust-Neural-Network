

pub mod loss_fns{

    pub trait Loss{
        fn calculate_loss(&self, predictions: &[f32], targets: &[f32]) -> f32;
    }

    /// Mean Squared Error
    pub struct MSE{}
    impl Loss for MSE{
        /// Mean squared error
        fn calculate_loss(&self, predictions: &[f32], targets: &[f32]) -> f32 {
            predictions.iter()
                .zip(targets.iter())
                .map(|(p,r)| (p-r).powi(2))
                .sum::<f32>() / predictions.len() as f32
        }
    }

    /// Mean Absolute Error
    pub struct MAE{}
    impl Loss for MAE{
        /// Mean absolute error
        fn calculate_loss(&self, predictions: &[f32], targets: &[f32]) -> f32 {
            predictions.iter()
                .zip(targets.iter())
                .map(|(p,r)| (p - r).abs())
                .sum::<f32>() / predictions.len() as f32
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
        assert_eq!(4f32, a.calculate_loss(&pred, &resp));
    }

    #[test]
    /// Test MAE
    fn mae(){
        let a = MAE{};
        let pred = vec![0.0, 1.0, 2.0, 3.0];
        let resp = vec![2.0, 3.0, 4.0, 5.0];
        assert_eq!(2f32, a.calculate_loss(&pred, &resp));
    }
}