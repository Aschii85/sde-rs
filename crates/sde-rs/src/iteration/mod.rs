pub mod euler;
pub mod runge_kutta;

pub enum IterationScheme {
    Euler,
    RungeKutta,
}
