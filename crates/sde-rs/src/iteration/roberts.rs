//! Stochastic Runge-Kutta ("improved Euler") scheme for Ito SDEs (Roberts, 2012),
//! as described at https://en.wikipedia.org/wiki/Runge–Kutta_method_(SDE):
//!
//!   k1 = a(t_n, X_n) h + b(t_n, X_n) (dW − S√h)
//!   k2 = a(t_n+h, X_n + k1) h + b(t_n+h, X_n + k1) (dW + S√h)
//!   X_{n+1} = X_n + (k1 + k2) / 2,        S = ±1 with probability 1/2
//!
//! The scheme is only defined for Wiener incrementors.

use crate::{
    filtration::{Filtration, TimeStep},
    increment::Incrementor,
    process::ProcessUniverse,
};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

pub fn roberts_iteration(
    filtration: &mut Filtration,
    process_universe: &ProcessUniverse,
    current_time_step: &TimeStep,
    next_time_step: &TimeStep,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let num_processes = process_universe.num_processes();
    let dt = next_time_step.value.into_inner() - current_time_step.value.into_inner();
    let sqrt_dt = dt.sqrt();

    // Capture state at t_idx to avoid repetitive filtration lookups
    let mut x_t = vec![0.0; num_processes];
    for process in process_universe.levy_processes() {
        x_t[process.index] = filtration.get(current_time_step, process);
    }

    // 1. Generate the sk random variable (±1) for the stochastic correction
    let sk = {
        let sk_seed = filtration.rng_seed() ^ (current_time_step.index as u64);
        let mut rng = ChaCha20Rng::seed_from_u64(sk_seed);
        if rng.random_bool(0.5) { 1.0 } else { -1.0 }
    };

    // Compute k1
    let mut k1 = vec![0.0; num_processes];
    for process in process_universe.levy_processes() {
        for term in process.terms.iter() {
            k1[process.index] += term.coefficient.eval(current_time_step, filtration)?
                * match term.incrementor.as_ref() {
                    Some(inc) => match inc {
                        Incrementor::Time(_) => dt,
                        Incrementor::Wiener(_) => {
                            inc.sample(current_time_step, next_time_step, filtration)?
                                - sk * sqrt_dt
                        }
                        _ => return Err("Roberts scheme supports only dt and dW terms".into()),
                    },
                    None => 1.0,
                }
        }
    }

    // Set a temporary "probed" state in the filtration for t+1
    for process in process_universe.levy_processes() {
        filtration.set(
            next_time_step,
            process,
            x_t[process.index] + k1[process.index],
        );
    }
    // Settle Algebraic processes for the probe
    for process in process_universe.algebraic_processes() {
        let value = process.terms.iter().try_fold(0.0, |acc, t| {
            let val = t.coefficient.eval(next_time_step, filtration)?;
            Ok::<f64, fasteval::Error>(acc + val)
        })?;
        filtration.set(next_time_step, process, value);
    }

    // Compute k2 using the probed state
    let mut k2 = vec![0.0; num_processes];
    for process in process_universe.levy_processes() {
        for term in process.terms.iter() {
            k2[process.index] += term.coefficient.eval(next_time_step, filtration)?
                * match term.incrementor.as_ref() {
                    Some(inc) => match inc {
                        Incrementor::Time(_) => dt,
                        Incrementor::Wiener(_) => {
                            inc.sample(current_time_step, next_time_step, filtration)?
                                + sk * sqrt_dt
                        }
                        _ => return Err("Roberts scheme supports only dt and dW terms".into()),
                    },
                    None => 1.0,
                }
        }
    }

    // Settle Levy Processes
    for process in process_universe.levy_processes() {
        filtration.set(
            next_time_step,
            process,
            x_t[process.index] + 0.5 * (k1[process.index] + k2[process.index]),
        );
    }

    // Settle Algebraic processes
    // These must be calculated last so they see the final converged Levy values at t+1
    for process in process_universe.algebraic_processes() {
        let value = process.terms.iter().try_fold(0.0, |acc, t| {
            let val = t.coefficient.eval(next_time_step, filtration)?;
            Ok::<f64, fasteval::Error>(acc + val)
        })?;
        filtration.set(next_time_step, process, value);
    }

    Ok(())
}
