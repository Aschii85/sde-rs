use crate::simulation::SimulationResult;

use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use zarrs::array::{Array, ArrayBuilder, data_type};
use zarrs_filesystem::FilesystemStore;

pub struct ZarrSimulationStreamer {
    array: Array<FilesystemStore>,
    num_scenarios: usize,
    num_times: usize,
    num_processes: usize,
}

impl ZarrSimulationStreamer {
    /// Dimensions: [scenarios, times, processes]
    pub fn new(
        path: PathBuf,
        num_scenarios: usize,
        num_times: usize,
        num_processes: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let store = Arc::new(FilesystemStore::new(path)?);
        let shape = vec![num_scenarios as u64, num_times as u64, num_processes as u64];
        let chunk_shape = vec![1, num_times as u64, num_processes as u64];
        let array = ArrayBuilder::new(shape, chunk_shape, data_type::float64(), 0.0)
            .build(store, "/value")?;
        array.store_metadata()?;
        Ok(Self {
            array,
            num_scenarios,
            num_times,
            num_processes,
        })
    }

    /// Create a store whose shape matches an existing result.
    pub fn for_result(
        path: PathBuf,
        res: &SimulationResult,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new(
            path,
            res.scenarios().len(),
            res.times().len(),
            res.process_names().len(),
        )
    }

    /// Write the entire result in one call (layout already matches the array).
    pub fn write_result(&self, res: &SimulationResult) -> Result<(), Box<dyn std::error::Error>> {
        self.check_shape(res)?;
        let subset = [
            0..self.num_scenarios as u64,
            0..self.num_times as u64,
            0..self.num_processes as u64,
        ];
        self.array.store_array_subset(&subset, &res.data)?;
        Ok(())
    }

    /// Write only scenarios `range` (positions, not ids) of a result.
    /// Useful to bound memory when results are produced in batches.
    pub fn write_scenario_range(
        &self,
        res: &SimulationResult,
        range: Range<usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.check_shape(res)?;
        let block = self.num_times * self.num_processes;
        let subset = [
            range.start as u64..range.end as u64,
            0..self.num_times as u64,
            0..self.num_processes as u64,
        ];
        self.array
            .store_array_subset(&subset, &res.data[range.start * block..range.end * block])?;
        Ok(())
    }

    fn check_shape(&self, res: &SimulationResult) -> Result<(), Box<dyn std::error::Error>> {
        let expected = self.num_scenarios * self.num_times * self.num_processes;
        if res.scenarios().len() != self.num_scenarios
            || res.times().len() != self.num_times
            || res.process_names().len() != self.num_processes
            || res.data.len() != expected
        {
            return Err(format!(
                "result shape ({} x {} x {}, {} values) does not match store ({} x {} x {})",
                res.scenarios().len(),
                res.times().len(),
                res.process_names().len(),
                res.data.len(),
                self.num_scenarios,
                self.num_times,
                self.num_processes
            )
            .into());
        }
        Ok(())
    }
}
