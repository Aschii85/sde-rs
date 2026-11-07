use crate::{
    filtration::{ScenarioFiltration, TimeStep},
    increment::Incrementor,
    process::ProcessUniverse,
    rng::BaseRng,
};

// This strictly only works for wiener process increments, not poison jump processses...
// But I still allow the user to run the algorithm for all implemented processes...
pub fn runge_kutta_iteration(
    filtration: &mut ScenarioFiltration,
    process_universe: &ProcessUniverse,
    current_time_step: &TimeStep,
    next_time_step: &TimeStep,
    rng: &mut dyn BaseRng,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let num_processes = process_universe.num_processes();
    let num_incrementors = process_universe.num_incrementors();
    let sqrt_dt = (next_time_step.value.into_inner() - current_time_step.value.into_inner()).sqrt();

    // Capture state at t_idx to avoid repetitive filtration lookups
    let mut x_t = vec![0.0; num_processes];
    for process in process_universe.levy_processes() {
        x_t[process.index] = filtration.get(current_time_step, process);
    }

    // 1. Generate the sk random variable (±1) for the stochastic correction
    let sk = if rng.sample(current_time_step.index, 0)? > 0.5 {
        1.0
    } else {
        -1.0
    };

    // 2. Pre-sample all increments for this step. k1 and k2 MUST use the same dW and dN values.
    // TODO: Below can be made more efficient, since we iter over the same incrementors...
    let mut step_increments = Vec::with_capacity(num_incrementors);
    for process in process_universe.levy_processes() {
        for incrementor in process.terms.iter().map(|x| x.incrementor.as_ref()) {
            match incrementor {
                Some(inc) => {
                    if step_increments.get(inc.index()).is_none() {
                        step_increments.insert(
                            inc.index(),
                            inc.sample(current_time_step, next_time_step, filtration, rng),
                        );
                    }
                }
                None => {}
            }
        }
    }

    // --- STAGE 1: Compute k1 ---
    let mut k1 = vec![0.0; num_processes];
    for process in process_universe.levy_processes() {
        for term in process.terms.iter() {
            k1[process.index] += term.coefficient.eval(current_time_step, filtration)?
                * match term.incrementor.as_ref() {
                    Some(inc) => match step_increments[inc.index()] {
                        Ok(val) => val,
                        Err(_) => return Err("Failed to evaluate step increment".into()),
                    },
                    None => 1.0,
                }
        }
    }

    // --- STAGE 2: Compute k2 ---
    // We evaluate coefficients at the "probed" state (t + dt, x + k1 + perturbation)
    let mut k2 = vec![0.0; num_processes];
    // Set a temporary "probed" state in the filtration for t+1
    for process in process_universe.levy_processes() {
        let mut perturbation = 0.0;
        for term in process.terms.iter() {
            if let Some(Incrementor::Wiener(_)) = term.incrementor.as_ref() {
                perturbation +=
                    term.coefficient.eval(current_time_step, filtration)? * sk * sqrt_dt;
            }
        }
        filtration.set(
            next_time_step,
            process,
            x_t[process.index] + k1[process.index] + perturbation,
        );
    }
    // Now compute k2 using the probed state
    for process in process_universe.levy_processes() {
        for term in process.terms.iter() {
            k2[process.index] += term.coefficient.eval(next_time_step, filtration)?
                * match term.incrementor.as_ref() {
                    Some(inc) => match step_increments[inc.index()] {
                        Ok(val) => val,
                        Err(_) => return Err("Failed to evaluate step increment".into()),
                    },
                    None => 1.0,
                }
        }
    }

    // --- FINAL UPDATE: Settle Levy Processes ---
    for process in process_universe.levy_processes() {
        let p_idx = process.index;
        let final_val = x_t[p_idx] + 0.5 * (k1[p_idx] + k2[p_idx]);
        filtration.set(next_time_step, process, final_val);
    }

    // --- FINAL UPDATE: Settle Algebraic processes ---
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
