use sde_rs::parser::parse_equations;
use sde_rs::simulation::Simulation;

use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // ────── configuration ──────
    let initial_values = HashMap::from([("X1".to_string(), 100.0), ("X2".to_string(), 0.0)]);

    let processes_equations = vec![
        "dX1 = ( sin(t) ) * dt + (0.01 * X1) * dW1 + (0.001 * X1) * dN1(0.5 * cos(t))".to_string(),
        "X2 = max(X1 - 100.0, 0.0)".to_string(),
    ];

    let scheme = "euler"; // other valid value: "runge-kutta"
    let rng_method = "sobol"; // other valid value: "sobol"
    let num_scenarios: u64 = 10_000;

    // build a uniformly spaced time vector, identical to what the Python
    // wrapper accepts as `time_steps: Vec<OrderedFloat<f64>>`.
    let dt = 0.1;
    let t_start = 0.0;
    let t_end = 100.0;
    let time_steps: Vec<OrderedFloat<f64>> = (0..)
        .map(|i| OrderedFloat(t_start + i as f64 * dt))
        .take_while(|t| *t <= OrderedFloat(t_end))
        .collect();

    // parse the equations into a ProcessUniverse (same work done in Python)
    let universe = parse_equations(&processes_equations)?;
    let simulation = Simulation::new(
        Arc::new(universe),
        time_steps,
        (0..num_scenarios as u64).collect(),
        initial_values,
        &scheme,
        &rng_method,
        None,
    );
    let mut simulation_result = simulation.empty_result();
    // parallel computation of scenarios put into results vector
    let start = Instant::now();
    println!(
        "running {} scenarios with {} rng...",
        num_scenarios, rng_method
    );
    std::thread::scope(|s| {
        s.spawn(|| {
            simulation_result
                .data
                .par_chunks_mut(
                    simulation.time_steps.len() * simulation.process_universe.num_processes(),
                )
                .enumerate()
                .try_for_each(|(scenario, chunk)| -> Result<(), String> {
                    simulation
                        .simulate_scenario(scenario as u64, chunk)
                        .map_err(|e| e.to_string())
                })
        });
    });
    // let df = concat(
    //     results
    //         .into_iter()
    //         .map(|x| x.to_lazyframe())
    //         .collect::<Vec<_>>(),
    //     UnionArgs::default(),
    // )?
    // .collect()?;
    let elapsed = start.elapsed();
    println!("completed in {:.3}s", elapsed.as_secs_f64());

    // print a small portion of the output frame
    // println!("{:#?}", simulation_result);

    Ok(())
}
