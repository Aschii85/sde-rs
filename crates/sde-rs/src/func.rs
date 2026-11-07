use crate::filtration::{ScenarioFiltration, TimeStep};

use fasteval::{Compiler, Evaler, Instruction, Slab};

pub struct Function {
    instruction: Instruction,
    slab: Slab,
    expr_str: String,
}

impl Clone for Function {
    fn clone(&self) -> Self {
        Self::new(&self.expr_str).expect("Failed to re-compile on clone")
    }
}

impl Function {
    pub fn new(expr_str: &str) -> Result<Self, String> {
        let parser = fasteval::Parser::new();
        let mut slab = Slab::new();
        let expr = parser
            .parse(expr_str, &mut slab.ps)
            .map_err(|e| format!("Parse Error: {:?}", e))?;
        let instruction = expr.from(&slab.ps).compile(&slab.ps, &mut slab.cs);
        Ok(Self {
            instruction,
            slab,
            expr_str: expr_str.to_string(),
        })
    }

    pub fn eval(
        &self,
        time_step: &TimeStep,
        filtration: &mut ScenarioFiltration,
    ) -> Result<f64, fasteval::Error> {
        if time_step != filtration.cache_time_step() {
            match filtration.refresh_cache(time_step) {
                Ok(_) => {}
                Err(_) => {}
            };
        }
        self.instruction.eval(&self.slab, filtration.cache_values())
    }
}
