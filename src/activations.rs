
pub mod activation_fns{

    pub trait Activate{
        fn activate(&self, x: f32) -> f32;
        fn derivative(&self, x: f32) -> f32;
    }

    pub struct Relu {}
    impl Activate for Relu {
        /// max(0, x), length of x should be 1
        fn activate(&self, x: f32) -> f32 {
            f32::max(0f32,x)
        }

        /// Derivative of Relu
        fn derivative(&self, x: f32) -> f32 {
            if x > 0.0 { 1.0 } else { 0.0 }
        }
    }

    pub struct Sigmoid{}
    impl Activate for Sigmoid{
        /// 1 / (1 + e^-x)
        fn activate(&self, x: f32) -> f32 {
            1f32 / (1f32 + ((-1f32 * x).exp()))
        }

        /// sig *  (1 - sig)
        fn derivative(&self, x: f32) -> f32 {
            let s = self.activate(x);
            s * (1.0 - s)
        }
    }

    pub struct Tanh{}
    impl Activate for Tanh{
        /// tanh(x)
        fn activate(&self, x: f32) -> f32 {
            x.tanh()
        }

        /// 1 - tanh(x)^2
        fn derivative(&self, x: f32) -> f32 {
            1.0 - x.tanh().powi(2)
        }
    }

}

#[cfg(test)]
mod test{
    use crate::activations::activation_fns::{Sigmoid, Relu, Activate};

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

}