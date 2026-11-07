use crate::{
    func::Function,
    increment::{Incrementor, PoissonJumpIncrementor, TimeIncrementor, WienerIncrementor},
    process::{LagType, Process, ProcessLag, ProcessTerm, ProcessUniverse},
};

use ordered_float::OrderedFloat;
use regex::Regex;
use std::{collections::HashMap, error::Error};

/// Supports two lag formats:
/// 1. Index lags (integers): X1[-2] -> X1_LAG_2 (zero lag X1[0] -> X1, positive rejected)
/// 2. Time lags (floats/time expressions): X1[t-1.5] or X1[time:-1.5] -> X1_TIME_LAG_1_5
fn convert_lags(input: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
    // 1. Validate and convert time-based lags: e.g., X1[t-1.5] or X1[time:-1.5]
    let time_lag_re = Regex::new(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\[\s*(?:t|time)\s*:\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*\]",
    )?;
    let processed = time_lag_re
        .replace_all(input, |caps: &regex::Captures| {
            let var = &caps[1];
            let val = caps[2].replace('.', "_");
            format!("{}_TIME_LAG_{}", var, val)
        })
        .to_string();

    // Check for invalid positive time lags
    let pos_time_re = Regex::new(r"([a-zA-Z_][a-zA-Z0-9_]*)\[\s*(?:t|time)\s*:\s*\+?[1-9]")?;
    if pos_time_re.is_match(&processed) {
        return Err("Future time lookups are not allowed in lags.".into());
    }

    // 2. Check for invalid positive index lags: e.g., X1[4] or X1[+2]
    let positive_re = Regex::new(r"([a-zA-Z_][a-zA-Z0-9_]*)\[\s*\+?\s*([1-9][0-9]*)\s*\]")?;
    if let Some(caps) = positive_re.captures(&processed) {
        return Err(format!(
            "Invalid positive lag term '{}[{}]': Future lookups are not allowed.",
            &caps[1], &caps[2]
        )
        .into());
    }

    // 3. Handle zero lags: X1[0] -> X1
    let zero_re = Regex::new(r"([a-zA-Z_][a-zA-Z0-9_]*)\[\s*0+\s*\]")?;
    let cleaned = zero_re.replace_all(&processed, "${1}").to_string();

    // 4. Substitute valid negative index lags: X1[-N] -> X1_LAG_N
    let negative_re = Regex::new(r"([a-zA-Z_][a-zA-Z0-9_]*)\[\s*-\s*([1-9][0-9]*)\s*\]")?;
    let converted = negative_re
        .replace_all(&cleaned, "${1}_LAG_${2}")
        .to_string();

    Ok(converted)
}

/// Extracts both index and time lags from the converted expression and stores them under the base process name.
fn extract_and_store_lags(
    expr: &str,
    lags: &mut HashMap<String, Vec<ProcessLag>>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Index lags
    let index_lag_re = Regex::new(r"([a-zA-Z_][a-zA-Z0-9_]*)_LAG_([0-9]+)")?;
    for caps in index_lag_re.captures_iter(expr) {
        let base_name = caps[1].to_string();
        if let Ok(size) = caps[2].parse::<usize>() {
            let full_name = format!("{}_LAG_{}", base_name, &caps[2]);
            lags.entry(base_name).or_default().push(ProcessLag {
                value: LagType::Size(size),
                process_name: full_name,
            });
        }
    }
    // Time lags
    let time_lag_re = Regex::new(r"([a-zA-Z_][a-zA-Z0-9_]*)_TIME_LAG_([0-9_]+)")?;
    for caps in time_lag_re.captures_iter(expr) {
        let base_name = caps[1].to_string();
        let val_str = caps[2].replace('_', ".");
        if let Ok(time_size) = val_str.parse::<f64>() {
            let full_name = format!("{}_TIME_LAG_{}", base_name, &caps[2]);
            lags.entry(base_name).or_default().push(ProcessLag {
                value: LagType::Time(OrderedFloat(time_size)),
                process_name: full_name,
            });
        }
    }
    Ok(())
}

pub fn parse_equations(
    equations: &[String],
) -> Result<ProcessUniverse, Box<dyn Error + Send + Sync>> {
    let mut incrementors = Vec::new();
    incrementors.push(Incrementor::Time(TimeIncrementor::new(0)));
    let mut processes = equations
        .iter()
        .enumerate()
        .map(|(idx, eq)| parse_single_equation(eq, idx, &mut incrementors))
        .collect::<Result<Vec<Process>, Box<dyn Error + Send + Sync>>>()?;

    let mut lags = HashMap::new();
    for eq in equations.iter() {
        let safe_eq = convert_lags(eq)?;
        extract_and_store_lags(&safe_eq, &mut lags)?;
    }

    for process in &mut processes {
        if let Some(process_lags) = lags.remove(&process.name) {
            process.lags = process_lags;
        }
    }

    Ok(ProcessUniverse::new(processes))
}

/// Splits a term into coefficient and incrementor while respecting parenthesis depth
/// so that asterisks inside jump intensity expressions (e.g., dN1(0.5 * X1)) are ignored.
fn split_term_coeff_incrementor(term: &str) -> Result<(&str, &str), String> {
    let bytes = term.as_bytes();
    let mut depth = 0;
    let mut split_idx = None;

    for i in (0..bytes.len()).rev() {
        match bytes[i] {
            b')' => depth += 1,
            b'(' => depth -= 1,
            b'*' if depth == 0 => {
                split_idx = Some(i);
                break;
            }
            _ => {}
        }
    }

    if let Some(idx) = split_idx {
        Ok((term[..idx].trim(), term[idx + 1..].trim()))
    } else {
        Err(format!(
            "Expected coefficient * incrementor in term: {}",
            term
        ))
    }
}

fn parse_single_equation(
    equation: &str,
    process_index: usize,
    incrementors: &mut Vec<Incrementor>,
) -> Result<Process, Box<dyn Error + Send + Sync>> {
    let parts: Vec<&str> = equation.split('=').collect();
    if parts.len() != 2 {
        return Err(format!("Invalid equation format: {}", equation).into());
    }

    let lhs = parts[0].trim();
    let rhs = parts[1].trim();
    let rhs_terms: Vec<&str> = rhs
        .split('+')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    let process_name = lhs.strip_prefix('d').unwrap_or(lhs).trim().to_string();
    let mut process_terms = Vec::new();

    if lhs.starts_with('d') {
        for term in rhs_terms {
            let (coeff_str, inc_str) = split_term_coeff_incrementor(term)?;
            let clean_coeff = coeff_str
                .strip_prefix('(')
                .and_then(|s| s.strip_suffix(')'))
                .unwrap_or(coeff_str)
                .trim();
            let safe_coeff = convert_lags(clean_coeff)?;
            let coeff_fn = Function::new(&safe_coeff)
                .map_err(|e| format!("Function error in '{}': {}", safe_coeff, e))?;
            let incrementor = build_incrementor(inc_str, incrementors)?;
            process_terms.push(ProcessTerm {
                coefficient: coeff_fn,
                incrementor: Some(incrementor),
            });
        }
        Ok(Process::new(
            process_index,
            process_name,
            Vec::new(),
            false,
            process_terms,
        ))
    } else {
        let safe_rhs = convert_lags(rhs)?;
        let coeff_fn = Function::new(&safe_rhs)?;
        process_terms.push(ProcessTerm {
            coefficient: coeff_fn,
            incrementor: None,
        });
        Ok(Process::new(
            process_index,
            process_name,
            Vec::new(),
            true,
            process_terms,
        ))
    }
}

fn build_incrementor(
    inc_str: &str,
    incrementors: &mut Vec<Incrementor>,
) -> Result<Incrementor, Box<dyn Error + Send + Sync>> {
    if inc_str == "dt" {
        for incrementor in incrementors.iter() {
            if incrementor.name() == inc_str {
                return Ok(incrementor.clone());
            }
        }
        let index = incrementors.len();
        let incrementor = Incrementor::Time(TimeIncrementor::new(index));
        incrementors.push(incrementor.clone());
        Ok(incrementor)
    } else if inc_str.starts_with("dW") {
        for incrementor in incrementors.iter() {
            if incrementor.name() == inc_str {
                return Ok(incrementor.clone());
            }
        }
        let index = incrementors.len();
        let incrementor = Incrementor::Wiener(WienerIncrementor::new(index, inc_str.to_string()));
        incrementors.push(incrementor.clone());
        Ok(incrementor)
    } else if inc_str.starts_with("dN") {
        let start = inc_str.find('(').ok_or("dN missing opening parenthesis")?;
        let end = inc_str.rfind(')').ok_or("dN missing closing parenthesis")?;
        let lambda_expr = &inc_str[start + 1..end];
        let safe_lambda = convert_lags(lambda_expr.trim())?;
        let lambda_fn = Function::new(&safe_lambda)
            .map_err(|e| format!("Function error in jump intensity: {}", e))?;
        let name = &inc_str[0..start];
        for incrementor in incrementors.iter() {
            if incrementor.name() == name {
                return Ok(incrementor.clone());
            }
        }
        let index = incrementors.len();
        let incrementor = Incrementor::PoissonJump(PoissonJumpIncrementor::new(
            index,
            name.to_string(),
            lambda_fn,
        ));
        incrementors.push(incrementor.clone());
        Ok(incrementor)
    } else {
        Err(format!("Unknown incrementor type: {}", inc_str).into())
    }
}
