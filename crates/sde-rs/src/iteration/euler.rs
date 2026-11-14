use crate::{
    filtration::{Filtration, TimeStep},
    process::ProcessUniverse,
};

pub fn euler_iteration(
    filtration: &mut Filtration,
    process_universe: &ProcessUniverse,
    current_time_step: &TimeStep,
    next_time_step: &TimeStep,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // PASS 1: Compute all SDE-based (Levy) updates
    for process in process_universe.levy_processes() {
        let mut value = filtration.get(current_time_step, process);
        for term in process.terms.iter() {
            let coeff = term.coefficient.eval(current_time_step, filtration)?;
            value += match &term.incrementor {
                Some(inc) => coeff * inc.sample(current_time_step, next_time_step, filtration)?,
                None => coeff,
            }
        }
        filtration.set(next_time_step, process, value);
    }
    // PASS 2: Evaluate Algebraic processes using next, t + 1, values
    for process in process_universe.algebraic_processes() {
        let value = process.terms.iter().try_fold(0.0, |acc, t| {
            let val = t.coefficient.eval(next_time_step, filtration)?;
            Ok::<f64, fasteval::Error>(acc + val)
        })?;
        filtration.set(next_time_step, process, value);
    }
    Ok(())
}
