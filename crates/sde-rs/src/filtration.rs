use crate::{
    process::{LagType, Process, ProcessUniverse, Variable},
    rng::RngDraws,
};

use fasteval::EvalNamespace;
use itertools::Itertools;
use ordered_float::OrderedFloat;
use std::{collections::HashMap, sync::Arc};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TimeStep {
    pub value: OrderedFloat<f64>,
    pub index: usize,
}

#[derive(Clone)]
pub struct TimeStepContainer {
    time_steps: Vec<TimeStep>,
}

impl TimeStepContainer {
    pub fn new(times: Vec<OrderedFloat<f64>>) -> Self {
        let mut sorted_time_steps = Vec::with_capacity(times.len());
        for (i, t) in times.into_iter().sorted().enumerate() {
            sorted_time_steps.push(TimeStep { value: t, index: i })
        }
        Self {
            time_steps: sorted_time_steps,
        }
    }

    pub fn next(&self, time_step: &TimeStep) -> Option<&TimeStep> {
        self.time_steps.get(time_step.index + 1)
    }

    pub fn get_by_index(&self, index: usize) -> Option<&TimeStep> {
        self.time_steps.get(index)
    }

    pub fn iter(&self) -> impl Iterator<Item = &TimeStep> {
        self.time_steps.iter()
    }

    pub fn len(&self) -> usize {
        self.time_steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.time_steps.is_empty()
    }

    pub fn lagged(&self, time_step: &TimeStep, lag: &LagType) -> Option<&TimeStep> {
        match lag {
            LagType::Size(s) => Some(self.get_by_index(time_step.index.saturating_sub(*s))?),
            LagType::Time(t) => {
                let target = time_step.value - t;
                match self
                    .time_steps
                    .binary_search_by(|item| item.value.cmp(&target))
                {
                    Ok(idx) => Some(&self.time_steps[idx]), // Exact match found
                    Err(idx) => {
                        // idx is where the target would be inserted.
                        // Check boundaries and compare the elements at idx and idx - 1.
                        let num_time_steps = self.time_steps.len();
                        match idx {
                            0 => Some(&self.time_steps[0]),
                            i if i >= num_time_steps => Some(&self.time_steps[num_time_steps - 1]),
                            _ => {
                                let prev = self.time_steps[idx - 1];
                                let next = self.time_steps[idx];
                                if (target - prev.value).abs() <= (next.value - target).abs() {
                                    Some(&self.time_steps[idx - 1])
                                } else {
                                    Some(&self.time_steps[idx])
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

pub struct Filtration<'a> {
    raw_data: &'a mut [f64],
    process_universe: Arc<ProcessUniverse>,
    time_steps: TimeStepContainer,
    rng_draws: RngDraws,
}

impl<'a> Filtration<'a> {
    pub fn new(
        raw_data: &'a mut [f64],
        process_universe: Arc<ProcessUniverse>,
        time_steps: TimeStepContainer,
        initial_values: &HashMap<String, f64>,
        rng_draws: RngDraws,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let initial_time_step = *time_steps.get_by_index(0).ok_or("no time steps")?;
        let process_universe_pointer = Arc::clone(&process_universe);
        for process in process_universe.processes() {
            let idx = process.index;
            if let Some(initial_value) = initial_values.get(&process.name) {
                raw_data[idx] = *initial_value;
            };
        }
        let mut filtration = Self {
            raw_data,
            process_universe,
            time_steps,
            rng_draws,
        };
        for alg_process in process_universe_pointer.algebraic_processes() {
            let mut value = 0.0;
            for term in alg_process.terms.iter() {
                value += term.coefficient.eval(&initial_time_step, &mut filtration)?;
            }
            filtration.set(&initial_time_step, alg_process, value);
        }
        Ok(filtration)
    }

    #[inline]
    pub fn get(&self, time_step: &TimeStep, process: &Process) -> f64 {
        let n_procs = self.process_universe.num_processes();
        let idx = time_step.index * n_procs + process.index;
        self.raw_data[idx]
    }

    #[inline]
    pub fn set(&mut self, time_step: &TimeStep, process: &Process, val: f64) {
        let n_procs = self.process_universe.num_processes();
        let idx = time_step.index * n_procs + process.index;
        self.raw_data[idx] = val;
    }

    #[inline]
    pub fn rng_draw(&self, time_step: &TimeStep, incremtor_index: usize) -> f64 {
        self.rng_draws.get(time_step.index, incremtor_index)
    }

    pub fn rng_seed(&self) -> u64 {
        self.rng_draws.seed
    }

    pub fn at(&mut self, time_step: &TimeStep) -> State<'_, 'a> {
        State {
            time_step: *time_step,
            filtration: self,
        }
    }
}

pub struct State<'f, 'a> {
    time_step: TimeStep,
    filtration: &'f mut Filtration<'a>,
}

impl EvalNamespace for State<'_, '_> {
    fn lookup(&mut self, name: &str, _args: Vec<f64>, _keybuf: &mut String) -> Option<f64> {
        match name {
            "t" => Some(self.time_step.value.into_inner()),
            _ => match self.filtration.process_universe.get_variable(name) {
                Some(var) => match var {
                    Variable::Current(process) => {
                        Some(self.filtration.get(&self.time_step, process))
                    }
                    Variable::Lagged { process, lag } => {
                        let lagged_time_step =
                            self.filtration.time_steps.lagged(&self.time_step, lag)?;
                        Some(self.filtration.get(lagged_time_step, process))
                    }
                },
                None => None,
            },
        }
    }
}
