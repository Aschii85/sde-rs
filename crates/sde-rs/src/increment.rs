use crate::{
    filtration::{ScenarioFiltration, TimeStep},
    func::Function,
    rng::BaseRng,
};
use std::error::Error;

#[derive(Clone)]
pub struct TimeIncrementor {
    index: usize,
    name: String,
}

impl TimeIncrementor {
    pub fn new(index: usize) -> Self {
        Self {
            index,
            name: String::from("dt"),
        }
    }

    fn sample(
        &self,
        current_time_step: &TimeStep,
        next_time_step: &TimeStep,
    ) -> Result<f64, Box<dyn Error + Send + Sync>> {
        Ok(next_time_step.value.into_inner() - current_time_step.value.into_inner())
    }
}

#[derive(Clone)]
pub struct WienerIncrementor {
    index: usize,
    name: String,
}

impl WienerIncrementor {
    pub fn new(index: usize, name: String) -> Self {
        Self { index, name }
    }

    fn sample(
        &self,
        current_time_step: &TimeStep,
        next_time_step: &TimeStep,
        rng: &mut dyn BaseRng,
    ) -> Result<f64, Box<dyn Error + Send + Sync>> {
        let q = rng.sample(current_time_step.index, self.index)?;
        let sqrt_dt =
            (next_time_step.value.into_inner() - current_time_step.value.into_inner()).sqrt();
        Ok(sqrt_dt * fast_inverse_normal_cdf(q))
    }
}

#[derive(Clone)]
pub struct PoissonJumpIncrementor {
    index: usize,
    name: String,
    lambda: Function,
}

impl PoissonJumpIncrementor {
    pub fn new(index: usize, name: String, lambda: Function) -> Self {
        Self {
            index,
            name,
            lambda,
        }
    }

    pub fn sample(
        &self,
        current_time_step: &TimeStep,
        next_time_step: &TimeStep,
        filtration: &mut ScenarioFiltration,
        rng: &mut dyn BaseRng,
    ) -> Result<f64, Box<dyn Error + Send + Sync>> {
        let u = rng.sample(current_time_step.index, self.index)?;
        let dt = next_time_step.value.into_inner() - current_time_step.value.into_inner();
        let effective_lambda = self.lambda.eval(current_time_step, filtration)? * dt;
        Ok(fast_inverse_poisson_cdf(u, effective_lambda) as f64)
    }
}

#[derive(Clone)]
pub enum Incrementor {
    Time(TimeIncrementor),
    Wiener(WienerIncrementor),
    PoissonJump(PoissonJumpIncrementor),
}

impl Incrementor {
    pub fn name(&self) -> &String {
        match self {
            Incrementor::Time(t) => &t.name,
            Incrementor::Wiener(w) => &w.name,
            Incrementor::PoissonJump(pj) => &pj.name,
        }
    }

    pub fn sample(
        &self,
        current_time_step: &TimeStep,
        next_time_step: &TimeStep,
        filtration: &mut ScenarioFiltration,
        rng: &mut dyn BaseRng,
    ) -> Result<f64, Box<dyn Error + Send + Sync>> {
        match self {
            Incrementor::Time(t) => t.sample(current_time_step, next_time_step),
            Incrementor::Wiener(w) => w.sample(current_time_step, next_time_step, rng),
            Incrementor::PoissonJump(pj) => {
                pj.sample(current_time_step, next_time_step, filtration, rng)
            }
        }
    }

    pub fn index(&self) -> usize {
        match self {
            Incrementor::Time(inc) => inc.index,
            Incrementor::Wiener(inc) => inc.index,
            Incrementor::PoissonJump(inc) => inc.index,
        }
    }
}

impl std::fmt::Debug for Incrementor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(self.name().as_str()).finish()
    }
}

// Inverse cdff functions
fn fast_inverse_normal_cdf(p: f64) -> f64 {
    // High-precision approximation (Acklam's or similar)
    // For brevity, here is a standard efficient approximation
    // often used in high-performance simulators:
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let x = t - ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0);
    if p < 0.5 { -x } else { x }
}

fn fast_inverse_poisson_cdf(u: f64, lambda: f64) -> u64 {
    if lambda <= 0.0 {
        return 0;
    }
    // Initial probability P(X=0) = e^(-lambda)
    let mut p = (-lambda).exp();
    let mut f = p; // Cumulative distribution function value
    let mut k = 0;
    // Iterate until the cumulative probability exceeds our uniform sample
    // A cap of 200 is used for numerical safety, though for small lambda
    // it will resolve much earlier.
    while u > f && k < 200 {
        k += 1;
        // Recurrence: P(X=k) = P(X=k-1) * lambda / k
        p *= lambda / (k as f64);
        f += p;
    }
    k
}
