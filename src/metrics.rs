
pub mod metrics{

    pub trait Metric{
        fn calculate_metric(&self,
                            true_pos: u64,
                            true_neg: u64,
                            false_pos:u64,
                            false_neg: u64) -> f32;
        fn calculate_and_save_metric(&mut self,
                                     true_pos: u64,
                                     true_neg: u64,
                                     false_pos:u64,
                                     false_neg: u64);
        fn get_value(&self, index: usize) -> f32;
        fn get_values(&self) -> &Vec<f32>;

    }

    pub struct Precision{
        pub values: Vec<f32>
    }
    impl Metric for Precision{
        /// Precision = TP / (TP + FP)
        /// Calculate and save value to values Vec<f32>
        fn calculate_metric(&self,
                            true_pos: u64,
                            _true_neg: u64,
                            false_pos: u64,
                            _false_neg: u64) -> f32 {
            (true_pos as f32 / (true_pos + false_pos) as f32) as f32
        }

        fn calculate_and_save_metric(&mut self,
                                     true_pos: u64,
                                     _true_neg: u64,
                                     false_pos:u64,
                                     _false_neg: u64) {
            self.values.push(
                self.calculate_metric(true_pos, _true_neg, false_pos, _false_neg));
        }

        fn get_value(&self, index: usize) -> f32 {
            *self.values.get(index).expect("Unable to access index")
        }

        fn get_values(&self) -> &Vec<f32>{
            &self.values
        }
    }

    pub struct Recall{
        pub values: Vec<f32>
    }
    impl Metric for Recall{
        /// Recall = TP / (TP + FN)
        /// Calculate and save value to values Vec<f32>
        fn calculate_metric(&self,
                            true_pos: u64,
                            _true_neg: u64,
                            _false_pos: u64,
                            false_neg: u64) -> f32 {
            (true_pos as f32 / (true_pos + false_neg) as f32) as f32
        }

        fn calculate_and_save_metric(&mut self,
                                     true_pos: u64,
                                     _true_neg: u64,
                                     false_pos:u64,
                                     _false_neg: u64) {
            self.values.push(
                self.calculate_metric(true_pos, _true_neg, false_pos, _false_neg));
        }

        fn get_value(&self, index: usize) -> f32 {
            *self.values.get(index).expect("Unable to access index")
        }

        fn get_values(&self) -> &Vec<f32>{
            &self.values
        }
    }

    pub struct Accuracy{
        pub values: Vec<f32>
    }
    impl Metric for Accuracy{
        /// Accuracy = (TP + TN) / (TP + TN + FP + FN)
        /// Calculate and save value to values Vec<f32>
        fn calculate_metric(&self,
                            true_pos: u64,
                            true_neg: u64,
                            false_pos: u64,
                            false_neg: u64) -> f32 {
            ((true_neg + true_pos) as f32 / (true_neg + true_pos + false_neg + false_pos) as f32) as f32
        }

        fn calculate_and_save_metric(&mut self,
                                     true_pos: u64,
                                     _true_neg: u64,
                                     false_pos:u64,
                                     _false_neg: u64) {
            self.values.push(
                self.calculate_metric(true_pos, _true_neg, false_pos, _false_neg));
        }

        fn get_value(&self, index: usize) -> f32 {
            *self.values.get(index).expect("Unable to access index")
        }

        fn get_values(&self) -> &Vec<f32>{
            &self.values
        }
    }

}

#[cfg(test)]
mod tests{
    use crate::metrics::metrics::{Accuracy, Recall, Metric, Precision};

    #[test]
    /// Test Accuracy metric
    fn accuracy(){
        let m  = Accuracy{values: vec![]};
        assert_eq!(0.5f32, m.calculate_metric(5, 5, 5, 5));
        assert_eq!(0.25f32, m.calculate_metric(5, 0, 10, 5));
        assert_eq!(1f32, m.calculate_metric(10, 10, 0, 0));
    }

    #[test]
    /// Test Recall metric
    fn recall(){
        let m = Recall{values: vec![]};
        assert_eq!(0.5f32, m.calculate_metric(5, 0, 0, 5));
        assert_eq!(1f32, m.calculate_metric(10, 0, 0, 0));
        assert_eq!(0f32, m.calculate_metric(0, 0, 0, 10));

    }

    #[test]
    /// Test Precision metric
    fn precision(){
        let m = Precision{values: vec![]};
        assert_eq!(0.5f32, m.calculate_metric(5, 0, 5, 0));
        assert_eq!(1f32, m.calculate_metric(5, 0, 0, 0));
        assert_eq!(0f32, m.calculate_metric(0, 0, 5, 0));
    }
}
