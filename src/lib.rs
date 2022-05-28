
/*

TODO:
- create linear algebra logic
- create different activations functions
- create different optimizer functions
- create FC layers
- create Conv layers

 */

pub mod neural_net{


}

mod lin_alg{


}

pub mod activation_fns{
    use std::cmp::max;

    // TODO convert f32 to Float trait in nums crate to enable using all kinds of floats

    pub trait Activate{
        fn activate(x: f32) -> f32;
    }

    pub struct Relu {}
    impl Activate for Relu {
        fn activate(x: f32) -> f32 {
            f32::max(0f32,x)
        }
    }

    pub struct Sigmoid{}
    impl Activate for Sigmoid{
        fn activate(x: f32) -> f32 {
            // 1 / (1 + e^-x)
            1f32 / (1f32 + ((-1f32 * x).exp()))
        }
    }

    pub struct Tanh{}
    impl Activate for Tanh{
        fn activate(x: f32) -> f32 {
            f32::tanh(x)
        }
    }

}

mod optimizers{

}

mod layers{

}

#[cfg(test)]
mod tests{
    use crate::activation_fns::{Relu, Sigmoid, Activate};

    #[test]
    fn sigmoid(){
        assert_eq!(0.006692851, Sigmoid::activate(-5f32));
        assert_eq!(0.047425874, Sigmoid::activate(-3f32));
        assert_eq!(0.5, Sigmoid::activate(0f32));
        assert_eq!(0.95257413, Sigmoid::activate(3f32));
        assert_eq!(0.9933072, Sigmoid::activate(5f32));

    }

    #[test]
    fn relu(){
        assert_eq!(0f32, Relu::activate(-1f32));
        assert_eq!(1f32, Relu::activate(1f32));
        assert_eq!(0f32, Relu::activate(0f32));
    }

}