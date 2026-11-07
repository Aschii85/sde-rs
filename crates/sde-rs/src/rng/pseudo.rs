use crate::rng::{BaseRng, StepCache};
use rand::{Rng as RandRng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::error::Error;

// --- Pseudo RNG ---

pub struct PseudoRng {
    last_step: Option<StepCache>,
    num_increments: usize,
    rng: ChaCha8Rng,
}

impl PseudoRng {
    pub fn new(seed: u64, num_increments: usize) -> Self {
        Self {
            last_step: None,
            num_increments,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    fn refresh_cache(&mut self, time_idx: usize) {
        let mut values = Vec::with_capacity(self.num_increments);
        for _ in 0..self.num_increments {
            values.push(self.rng.random::<f64>());
        }
        self.last_step = Some(StepCache {
            time_idx: Some(time_idx),
            values,
        });
    }
}

impl BaseRng for PseudoRng {
    fn sample(
        &mut self,
        time_idx: usize,
        increment_idx: usize,
    ) -> Result<f64, Box<dyn Error + Send + Sync>> {
        match self
            .last_step
            .as_ref()
            .is_some_and(|c| c.time_idx == Some(time_idx))
        {
            true => {}
            false => self.refresh_cache(time_idx),
        };
        match self.last_step.as_ref() {
            Some(cache) => {
                let val = *cache.values.get(increment_idx).ok_or("value not found")?;
                Ok(val)
            }
            None => Err("cache not created".into()),
        }
    }
}
