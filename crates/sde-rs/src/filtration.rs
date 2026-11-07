use crate::process::{LagType, Process, ProcessUniverse};

use itertools::Itertools;
use ordered_float::OrderedFloat;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

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

    pub fn lagged(&self, time_step: &TimeStep, lag: &LagType) -> Option<&TimeStep> {
        match lag {
            LagType::Size(s) => Some(self.get_by_index(time_step.index.saturating_sub(*s))?),
            LagType::Time(t) => {
                match self.time_steps.binary_search_by(|item| item.value.cmp(t)) {
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
                                if (t - prev.value).abs() <= (next.value - t).abs() {
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

pub struct ScenarioFiltrationCache {
    pub time_step: TimeStep,
    pub values: BTreeMap<String, f64>,
}

pub struct ScenarioFiltration<'a> {
    raw_data: &'a mut [f64],
    time_steps: TimeStepContainer,
    process_universe: Arc<ProcessUniverse>,
    cache: ScenarioFiltrationCache,
}

impl<'a> ScenarioFiltration<'a> {
    pub fn new(
        raw_data: &'a mut [f64],
        process_universe: &Arc<ProcessUniverse>,
        time_steps: TimeStepContainer,
        initial_values: &HashMap<String, f64>,
    ) -> Self {
        let initial_time_step = time_steps.get_by_index(0).unwrap().clone();
        for process in process_universe.processes() {
            let idx = process.index;
            if let Some(initial_value) = initial_values.get(&process.name) {
                raw_data[idx] = *initial_value;
            };
        }
        // Use BTreeMap directly for fasteval evaluation
        let cache = ScenarioFiltrationCache {
            time_step: initial_time_step.clone(),
            values: BTreeMap::new(),
        };
        let mut scenario_filtration = Self {
            raw_data,
            time_steps,
            process_universe: Arc::clone(process_universe),
            cache,
        };
        scenario_filtration
            .refresh_cache(&initial_time_step)
            .unwrap();
        scenario_filtration
    }

    pub fn get(&self, time_step: &TimeStep, process: &Process) -> f64 {
        let n_procs = self.process_universe.num_processes();
        let idx = time_step.index * n_procs + process.index;
        self.raw_data[idx]
    }

    pub fn set(&mut self, time_step: &TimeStep, process: &Process, val: f64) {
        let n_procs = self.process_universe.num_processes();
        let idx = time_step.index * n_procs + process.index;
        self.raw_data[idx] = val;
    }

    pub fn cache_values(&mut self) -> &mut BTreeMap<String, f64> {
        &mut self.cache.values
    }

    pub fn cache_time_step(&self) -> &TimeStep {
        &self.cache.time_step
    }

    pub fn refresh_cache(
        &mut self,
        time_step: &TimeStep,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Get current (and lagged) process values and insert into BTreeMap efficiently
        for process in self.process_universe.processes() {
            let value = self.get(time_step, process);
            if let Some(val) = self.cache.values.get_mut(process.name.as_str()) {
                *val = value;
            } else {
                self.cache.values.insert(process.name.clone(), value);
            }
            for lag in process.lags.iter() {
                let value = self.get(
                    self.time_steps
                        .lagged(time_step, &lag.value)
                        .ok_or("Could not get lagged value")?,
                    process,
                );
                if let Some(val) = self.cache.values.get_mut(lag.process_name.as_str()) {
                    *val = value;
                } else {
                    self.cache.values.insert(lag.process_name.clone(), value);
                }
            }
        }
        // Update the current time step
        self.cache.time_step = time_step.clone();
        // Update time variable for fasteval
        if let Some(val) = self.cache.values.get_mut("t") {
            *val = time_step.value.into_inner();
        } else {
            self.cache
                .values
                .insert("t".to_string(), time_step.value.into_inner());
        }
        Ok(())
    }
}
