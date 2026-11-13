use crate::{
    filtration::{Filtration, TimeStep},
    func::Function,
};
use std::error::Error;

#[derive(Clone)]
pub struct TimeIncrementor {
    name: String,
}

impl TimeIncrementor {
    pub fn new() -> Self {
        Self {
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

impl Default for TimeIncrementor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct WienerIncrementor {
    pub index: usize,
    pub name: String,
}

impl WienerIncrementor {
    pub fn new(index: usize, name: String) -> Self {
        Self { index, name }
    }

    fn sample(
        &self,
        current_time_step: &TimeStep,
        next_time_step: &TimeStep,
        filtration: &Filtration,
    ) -> Result<f64, Box<dyn Error + Send + Sync>> {
        let q = filtration.rng_draw(current_time_step, self.index);
        let sqrt_dt =
            (next_time_step.value.into_inner() - current_time_step.value.into_inner()).sqrt();
        Ok(sqrt_dt * fast_inverse_normal_cdf(q))
    }
}

#[derive(Clone)]
pub struct PoissonJumpIncrementor {
    pub index: usize,
    pub name: String,
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
        filtration: &mut Filtration,
    ) -> Result<f64, Box<dyn Error + Send + Sync>> {
        let u = filtration.rng_draw(current_time_step, self.index);
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
        filtration: &mut Filtration,
    ) -> Result<f64, Box<dyn Error + Send + Sync>> {
        match self {
            Incrementor::Time(t) => t.sample(current_time_step, next_time_step),
            Incrementor::Wiener(w) => w.sample(current_time_step, next_time_step, filtration),
            Incrementor::PoissonJump(pj) => {
                pj.sample(current_time_step, next_time_step, filtration)
            }
        }
    }

    pub fn index(&self) -> Option<usize> {
        match self {
            Incrementor::Time(_) => None,
            Incrementor::Wiener(inc) => Some(inc.index),
            Incrementor::PoissonJump(inc) => Some(inc.index),
        }
    }
}

impl std::fmt::Debug for Incrementor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(self.name().as_str()).finish()
    }
}

/// Computes the inverse standard normal CDF (probit function) for `p` in `(0.0, 1.0)`.
///
/// Implements Michael J. Wichura's Algorithm AS 241 (`PPND16`) with ~16 digits of accuracy.
/// Returns `f64::NEG_INFINITY` for `p <= 0.0`, `f64::INFINITY` for `p >= 1.0`,
/// and `f64::NAN` for `NaN` inputs.
#[inline]
fn fast_inverse_normal_cdf(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Coefficients for central region: |p - 0.5| <= 0.425
    const A0: f64 = 3.387_132_872_796_366_5;
    const A1: f64 = 1.331_416_678_917_843_8e2;
    const A2: f64 = 1.971_590_950_306_551_3e3;
    const A3: f64 = 1.373_169_376_550_946e4;
    const A4: f64 = 4.592_195_393_154_987e4;
    const A5: f64 = 6.726_577_092_700_87e4;
    const A6: f64 = 3.343_057_558_358_813e4;
    const A7: f64 = 2.509_080_928_730_122_7e3;

    const B1: f64 = 4.231_333_070_160_091e1;
    const B2: f64 = 6.871_870_074_920_579e2;
    const B3: f64 = 5.394_196_021_424_751e3;
    const B4: f64 = 2.121_379_430_158_659_7e4;
    const B5: f64 = 3.930_789_580_009_271e4;
    const B6: f64 = 2.872_908_573_572_194_3e4;
    const B7: f64 = 5.226_495_278_852_854e3;

    // Coefficients for intermediate tails: 0.425 < |p - 0.5| and sqrt(-ln(min(p, 1-p))) <= 5.0
    const C0: f64 = 1.423_437_110_749_683_5;
    const C1: f64 = 4.630_337_846_156_546;
    const C2: f64 = 5.769_497_221_460_691;
    const C3: f64 = 3.647_848_324_763_204_5;
    const C4: f64 = 1.270_458_252_452_368_4;
    const C5: f64 = 2.417_807_251_774_506e-1;
    const C6: f64 = 2.272_384_498_926_918_4e-2;
    const C7: f64 = 7.745_450_142_783_414e-4;

    const D1: f64 = 2.053_191_626_637_759;
    const D2: f64 = 1.676_384_830_183_803_8;
    const D3: f64 = 6.897_673_349_851e-1;
    const D4: f64 = 1.481_039_764_274_800_8e-1;
    const D5: f64 = 1.519_866_656_361_645_7e-2;
    const D6: f64 = 5.475_938_084_995_345e-4;
    const D7: f64 = 1.050_750_071_644_416_9e-9;

    // Coefficients for extreme tails: sqrt(-ln(min(p, 1-p))) > 5.0
    const E0: f64 = 6.657_904_643_501_103;
    const E1: f64 = 5.463_784_911_164_114;
    const E2: f64 = 1.784_826_539_917_291_3;
    const E3: f64 = 2.965_605_718_285_048_7e-1;
    const E4: f64 = 2.653_218_952_657_612_4e-2;
    const E5: f64 = 1.242_660_947_388_078_4e-3;
    const E6: f64 = 2.711_555_568_743_487_6e-5;
    const E7: f64 = 2.010_334_399_292_288_1e-7;

    const F1: f64 = 5.998_322_065_558_88e-1;
    const F2: f64 = 1.369_298_809_227_358e-1;
    const F3: f64 = 1.487_536_129_085_061_5e-2;
    const F4: f64 = 7.868_691_311_456_133e-4;
    const F5: f64 = 1.846_318_317_510_054_8e-5;
    const F6: f64 = 1.421_511_758_316_446e-7;
    const F7: f64 = 2.044_263_103_389_939_7e-15;

    let q = p - 0.5;

    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        let num =
            (((((((A7 * r + A6) * r + A5) * r + A4) * r + A3) * r + A2) * r + A1) * r + A0) * q;
        let den = ((((((B7 * r + B6) * r + B5) * r + B4) * r + B3) * r + B2) * r + B1) * r + 1.0;
        return num / den;
    }

    let r = if q < 0.0 { p } else { 1.0 - p };
    let r = (-r.ln()).sqrt();

    let val = if r <= 5.0 {
        let r = r - 1.6;
        let num = ((((((C7 * r + C6) * r + C5) * r + C4) * r + C3) * r + C2) * r + C1) * r + C0;
        let den = ((((((D7 * r + D6) * r + D5) * r + D4) * r + D3) * r + D2) * r + D1) * r + 1.0;
        num / den
    } else {
        let r = r - 5.0;
        let num = ((((((E7 * r + E6) * r + E5) * r + E4) * r + E3) * r + E2) * r + E1) * r + E0;
        let den = ((((((F7 * r + F6) * r + F5) * r + F4) * r + F3) * r + F2) * r + F1) * r + 1.0;
        num / den
    };

    if q < 0.0 { -val } else { val }
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
