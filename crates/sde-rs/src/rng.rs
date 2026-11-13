use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sobol::params::JoeKuoD6;
use std::sync::{Arc, Mutex, OnceLock};

static SOBOL_LOCK: OnceLock<JoeKuoD6> = OnceLock::new();

pub enum RngMethod {
    Pseudo,
    Sobol,
}

/// Uniform(0,1) draws for one scenario, laid out [time step][incrementor].
pub struct RngDraws {
    values: Vec<f64>,
    num_incrementors: usize,
    pub seed: u64,
}

impl RngDraws {
    #[inline]
    pub fn get(&self, time_idx: usize, incrementor_idx: usize) -> f64 {
        self.values[time_idx * self.num_incrementors + incrementor_idx]
    }
}

#[derive(Clone)]
enum Source {
    Pseudo,
    Sobol {
        points: Arc<Mutex<Box<dyn Iterator<Item = Vec<f64>> + Send>>>,
        shift: Arc<Vec<f64>>, // one shift, shared by every point
    },
}

#[derive(Clone)]
pub struct SimulationRng {
    pub random_seed: u64,
    dims: usize,
    num_incrementors: usize,
    source: Source,
}

impl SimulationRng {
    pub fn new(
        method: RngMethod,
        num_incrementors: usize,
        num_time_steps: usize,
        seed: Option<u64>,
    ) -> Self {
        let random_seed = seed.unwrap_or_else(rand::random);
        let dims = num_time_steps.saturating_sub(1) * num_incrementors;
        let source = match method {
            RngMethod::Pseudo => Source::Pseudo,
            RngMethod::Sobol => {
                let params = SOBOL_LOCK.get_or_init(JoeKuoD6::extended);
                let points = sobol::Sobol::<f64>::new(dims, params).skip(5);
                let mut rng = ChaCha8Rng::seed_from_u64(random_seed);
                Source::Sobol {
                    points: Arc::new(Mutex::new(Box::new(points))),
                    shift: Arc::new((0..dims).map(|_| rng.random()).collect()),
                }
            }
        };
        Self {
            random_seed,
            dims,
            num_incrementors,
            source,
        }
    }

    pub fn draw(&self, scenario: u64) -> RngDraws {
        let seed = self.random_seed.wrapping_add(scenario);
        let values = match &self.source {
            Source::Pseudo => {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                (0..self.dims).map(|_| rng.random::<f64>()).collect()
            }
            Source::Sobol { points, shift } => {
                // lock only around next(); the guard is dropped at the end of this statement
                let mut v = points
                    .lock()
                    .unwrap()
                    .next()
                    .expect("Sobol sequence exhausted");
                for (x, s) in v.iter_mut().zip(shift.iter()) {
                    *x += s;
                    if *x >= 1.0 {
                        *x -= 1.0
                    }
                }
                v
            }
        };
        RngDraws {
            values,
            num_incrementors: self.num_incrementors,
            seed,
        }
    }
}
