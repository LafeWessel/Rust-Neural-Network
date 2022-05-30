
pub mod activation_fns{

    pub trait Activate{
        fn activate(&self, x: f32) -> f32;
    }

    pub struct Relu {}
    impl Activate for Relu {
        /// max(0, x), length of x should be 1
        fn activate(&self, x: f32) -> f32 {
            f32::max(0f32,x)
        }
    }

    pub struct Sigmoid{}
    impl Activate for Sigmoid{
        /// 1 / (1 + e^-x), length of x should be 1
        fn activate(&self, x: f32) -> f32 {
            1f32 / (1f32 + ((-1f32 * x).exp()))
        }
    }

    pub struct Tanh{}
    impl Activate for Tanh{
        /// tanh(x), length of x should be 1
        fn activate(&self, x: f32) -> f32 {
            x.tanh()
        }
    }

    pub struct Softmax{
        pub dividend: Option<f32>,
    }
    impl Activate for Softmax{

        fn activate(&self, x: f32) -> f32 {
            x.exp() / self.dividend.unwrap()
        }

    }
    impl Softmax{
        /// Initialize dividend for calculating outputs
        pub fn init_div(&mut self, input: &[f32]){
            self.dividend = Some(input.iter().map(|i| i.exp()).sum::<f32>())
        }

        /// Clear dividend after calculating inputs
        pub fn clear_div(&mut self){
            self.dividend = None;
        }

    }

}

#[cfg(test)]
mod test{
    use crate::activations::activation_fns::{Sigmoid, Relu, Softmax, Activate};

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
    /// Test Softmax activation function
    fn softmax(){
        let mut s = Softmax{dividend: None};
        // initialization vec
        let v: Vec<f32> = vec![1.0, 2.0, 3.0];

        // hand calculated dividend:
        let d = v[0].exp() + v[1].exp() + v[2].exp();

        // init softmax
        s.init_div(&v);

        assert_eq!(v[0].exp()/d, s.activate(v[0]));
        assert_eq!(v[1].exp()/d, s.activate(v[1]));
        assert_eq!(v[2].exp()/d, s.activate(v[2]));
    }

}