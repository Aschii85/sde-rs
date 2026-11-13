use crate::{func::Function, increment::Incrementor};

use ordered_float::OrderedFloat;
use rustc_hash::FxHashMap;
use std::{collections::HashSet, sync::Arc};

#[derive(Clone)]
pub enum LagType {
    Size(usize),
    Time(OrderedFloat<f64>),
}

#[derive(Clone)]
pub struct ProcessLag {
    pub value: LagType,
    pub process_name: String, // e.g. "X1_LAG_2" or "X3_TIME_LAG_1_5"
}

#[derive(Clone)]
pub struct ProcessTerm {
    pub coefficient: Function,
    pub incrementor: Option<Incrementor>,
}

#[derive(Clone)]
pub struct Process {
    pub index: usize,
    pub name: String,
    pub lags: Vec<ProcessLag>,
    pub is_algebraic: bool,
    pub terms: Vec<ProcessTerm>,
}

impl Process {
    pub fn new(
        index: usize,
        name: String,
        lags: Vec<ProcessLag>,
        is_algebraic: bool,
        terms: Vec<ProcessTerm>,
    ) -> Self {
        Self {
            index,
            name,
            lags,
            is_algebraic,
            terms,
        }
    }
}

#[derive(Clone)]
pub enum Variable {
    Current(Arc<Process>),
    Lagged { process: Arc<Process>, lag: LagType },
}

#[derive(Clone)]
pub struct ProcessUniverse {
    processes: Vec<Arc<Process>>,
    levy_process_indices: Vec<usize>,
    algebraic_process_indices: Vec<usize>,
    variables: FxHashMap<String, Variable>,
}

impl ProcessUniverse {
    pub fn new(processes: Vec<Process>) -> Self {
        let processes: Vec<Arc<Process>> = processes.into_iter().map(Arc::new).collect();
        let mut levy_process_indices = Vec::new();
        let mut algebraic_process_indices = Vec::new();
        for (idx, proc) in processes.iter().enumerate() {
            match proc.is_algebraic {
                false => levy_process_indices.push(idx),
                true => algebraic_process_indices.push(idx),
            }
        }
        let mut variables = FxHashMap::default();
        for process in &processes {
            variables.insert(process.name.clone(), Variable::Current(Arc::clone(process)));
            for lag in &process.lags {
                variables.insert(
                    lag.process_name.clone(),
                    Variable::Lagged {
                        process: Arc::clone(process),
                        lag: lag.value.clone(),
                    },
                );
            }
        }
        Self {
            processes,
            levy_process_indices,
            algebraic_process_indices,
            variables,
        }
    }

    pub fn processes(&self) -> impl Iterator<Item = &Arc<Process>> {
        self.processes.iter()
    }

    pub fn num_processes(&self) -> usize {
        self.processes.len()
    }

    pub fn get_variable(&self, process_name: &str) -> Option<&Variable> {
        self.variables.get(process_name)
    }

    pub fn num_random_incrementors(&self) -> usize {
        let mut incrementor_names = HashSet::new();
        for process in self.processes() {
            for inc in process.terms.iter().filter_map(|x| x.incrementor.as_ref()) {
                match inc {
                    Incrementor::Time(_) => {}
                    _ => {
                        incrementor_names.insert(inc.name());
                    }
                };
            }
        }
        incrementor_names.len()
    }

    pub fn levy_processes(&self) -> impl Iterator<Item = &Arc<Process>> {
        self.levy_process_indices
            .iter()
            .map(move |&idx| &self.processes[idx])
    }

    pub fn algebraic_processes(&self) -> impl Iterator<Item = &Arc<Process>> {
        self.algebraic_process_indices
            .iter()
            .map(move |&idx| &self.processes[idx])
    }
}
