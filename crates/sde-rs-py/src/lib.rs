use ::sde_rs::{parser::parse_equations, simulation::Simulation};
use sde_rs::{iteration::IterationScheme, rng::RngMethod};

use numpy::{PyArray1, PyArray3, PyArrayMethods};
use ordered_float::OrderedFloat;
use pyo3::{
    Bound,
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

enum Outcome {
    Done(),
    SimError(String),
    Interrupted(PyErr),
}

#[pyfunction]
#[pyo3(name = "_simulate")]
#[pyo3(signature = (processes_equations, time_steps, number_scenarios, initial_values, *, iteration_scheme="euler".to_string(), rng_method="pseudo".to_string(), rng_seed=None))]
pub fn simulate_py<'py>(
    py: Python<'py>,
    processes_equations: Vec<String>,
    time_steps: Vec<f64>,
    number_scenarios: i64,
    initial_values: HashMap<String, f64>,
    iteration_scheme: String,
    rng_method: String,
    rng_seed: Option<u64>,
) -> PyResult<(
    Bound<'py, PyArray3<f64>>, // (scenario, time_step, process)
    Vec<u64>,                  // scenarios
    Vec<f64>,                  // time steps
    Vec<String>,               // process names, in index order
)> {
    if number_scenarios <= 0 {
        return Err(PyValueError::new_err(
            "scenarios must be a positive integer",
        ));
    }
    if time_steps.is_empty() {
        return Err(PyValueError::new_err("time_steps must not be empty"));
    }
    let scenarios: Vec<u64> = (0..number_scenarios as u64).collect();
    let processes = parse_equations(&processes_equations)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse equations: {}", e)))?;
    let simulation = Simulation::new(
        Arc::new(processes),
        time_steps.iter().copied().map(OrderedFloat).collect(),
        scenarios.clone(),
        initial_values,
        match iteration_scheme.as_str() {
            "euler" => IterationScheme::Euler,
            "roberts" => IterationScheme::Roberts,
            other => Err(PyValueError::new_err(format!(
                "iteration_scheme must be 'euler' or 'roberts', got {other}"
            )))?,
        },
        match rng_method.as_str() {
            "pseudo" => RngMethod::Pseudo,
            "sobol" => RngMethod::Sobol,
            other => Err(PyValueError::new_err(format!(
                "rng_method must be 'pseudo' or 'sobol', got {other}"
            )))?,
        },
        rng_seed,
    );
    let mut simulation_result = simulation.empty_result();
    // parallel computation of scenarios put into results vector
    let cancel = AtomicBool::new(false);
    let block = simulation.time_steps.len() * simulation.process_universe.num_processes();
    let outcome = std::thread::scope(|s| {
        let handle = s.spawn(|| {
            simulation_result
                .data
                .par_chunks_mut(block)
                .enumerate()
                .try_for_each(|(scenario, chunk)| -> Result<(), String> {
                    if cancel.load(Ordering::Relaxed) {
                        return Err("cancelled".to_string());
                    }
                    simulation
                        .simulate_scenario(scenario as u64, chunk)
                        .map_err(|e| e.to_string())
                })
        });

        while !handle.is_finished() {
            if let Err(e) = py.check_signals() {
                cancel.store(true, Ordering::Relaxed);
                let _ = handle.join(); // workers exit quickly via the flag
                return Outcome::Interrupted(e);
            }
            py.detach(|| std::thread::sleep(Duration::from_millis(20)));
        }
        match handle.join().expect("simulation thread panicked") {
            Ok(_) => Outcome::Done(),
            Err(e) => Outcome::SimError(e),
        }
    });

    match outcome {
        Outcome::Done() => {
            let scenarios = simulation_result.scenarios().clone();
            let times: Vec<f64> = simulation_result
                .times()
                .clone()
                .iter()
                .map(|x| x.into_inner())
                .collect();
            let process_names = simulation_result.process_names().clone();
            Ok((
                // Shape: (scenario, time_step, process)
                PyArray1::from_vec(py, simulation_result.data).reshape([
                    scenarios.len(),
                    times.len(),
                    process_names.len(),
                ])?,
                scenarios,
                times,
                process_names,
            ))
        }
        Outcome::SimError(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {}", e))),
        Outcome::Interrupted(e) => Err(e), // raises KeyboardInterrupt in Python
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_py, m)?)?;
    Ok(())
}
