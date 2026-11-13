//! Equation parser.
//!
//! # Grammar
//!
//! ```text
//! equation      := differential | algebraic
//! differential  := "d" NAME "=" term (("+" | "-") term)*
//! term          := coefficient "*" incrementor | incrementor      (incrementor LAST)
//! incrementor   := "dt" | "dW" IDENT* | "dN" IDENT* "(" intensity ")"
//! algebraic     := NAME "=" expression                            (no incrementors)
//! ```
//!
//! * `+`/`-` only split terms at parenthesis depth 0, so `(mu + X) * dt` and
//!   `dN1(0.5 + X1)` work. Signs are supported (`a*dt - b*dW1`) and so is
//!   scientific notation (`1e-5 * dt`).
//! * An equation is differential when its left-hand side is `d<name>` **and**
//!   the right-hand side contains incrementors. Otherwise it is algebraic, so
//!   `discount = 0.9 * X` is a normal algebraic process called `discount`.
//!   Consequently `dt`, `dW*` and `dN*` are reserved as trailing factors.
//! * Lags: `X[-2]` (index lag), `X[0]` (current), `X[t-1.5]`, `X[t:-1.5]` or
//!   `X[time:-1.5]` (time lag, relative to the current time). Future lookups
//!   are rejected.

use crate::{
    func::Function,
    increment::{Incrementor, PoissonJumpIncrementor, TimeIncrementor, WienerIncrementor},
    process::{LagType, Process, ProcessLag, ProcessTerm, ProcessUniverse},
};

use ordered_float::OrderedFloat;
use regex::Regex;
use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
    sync::OnceLock,
};

// ───────────────────────────── public entry point ─────────────────────────────

pub fn parse_equations(
    equations: &[String],
) -> Result<ProcessUniverse, Box<dyn Error + Send + Sync>> {
    let mut parser = Parser::default();
    let mut processes: Vec<Process> = Vec::with_capacity(equations.len());

    for (i, equation) in equations.iter().enumerate() {
        let process = parser
            .parse_equation(equation, i)
            .map_err(|e| format!("equation {} ('{}'): {}", i + 1, equation.trim(), e))?;
        if processes.iter().any(|p| p.name == process.name) {
            return Err(format!(
                "equation {} ('{}'): process '{}' is defined more than once",
                i + 1,
                equation.trim(),
                process.name
            )
            .into());
        }
        processes.push(process);
    }

    // Attach the collected lags to the process they refer to.
    for (base, lags) in parser.lags.by_base {
        let process = processes
            .iter_mut()
            .find(|p| p.name == base)
            .ok_or_else(|| format!("a lag refers to the unknown process '{base}'"))?;
        process.lags = lags;
    }

    Ok(ProcessUniverse::new(processes))
}

// ───────────────────────────────── parser state ───────────────────────────────

#[derive(Default)]
struct Parser {
    registry: IncrementorRegistry,
    lags: LagTable,
}

impl Parser {
    fn parse_equation(&mut self, equation: &str, index: usize) -> Result<Process, String> {
        let (lhs, rhs) = equation
            .split_once('=')
            .ok_or("missing '=' (expected 'dX = ...' or 'Y = ...')")?;
        let (lhs, rhs) = (lhs.trim(), rhs.trim());

        if !is_ident(lhs) {
            return Err(format!("invalid left-hand side '{lhs}'"));
        }
        if rhs.is_empty() || rhs.starts_with('=') {
            return Err("empty or malformed right-hand side".into());
        }

        // Split the RHS into signed terms and find the incrementor of each one.
        let mut terms = Vec::new();
        for (negative, text) in split_signed_terms(rhs)? {
            let (coefficient, incrementor) = split_term(text)?;
            terms.push(Term {
                negative,
                text,
                coefficient,
                incrementor,
            });
        }
        let with_incrementor = terms.iter().filter(|t| t.incrementor.is_some()).count();

        if with_incrementor == 0 {
            return self.build_algebraic(index, lhs, rhs);
        }

        // ── differential equation ──
        let name = lhs
            .strip_prefix('d')
            .filter(|n| is_ident(n))
            .ok_or_else(|| {
                format!(
                    "the right-hand side has incrementors, so the left-hand side must be \
                     'd<name>' (e.g. 'd{lhs}'), got '{lhs}'"
                )
            })?;
        check_process_name(name)?;

        if let Some(bad) = terms.iter().find(|t| t.incrementor.is_none()) {
            return Err(format!(
                "term '{}' has no incrementor; every term of a differential equation must \
                 end with dt, dW<id> or dN<id>(intensity)",
                bad.text
            ));
        }

        let mut process_terms = Vec::with_capacity(terms.len());
        for term in terms {
            let kind = term
                .incrementor
                .ok_or("internal error: missing incrementor")?;
            let mut safe = rewrite_lags(&term.coefficient, &mut self.lags)?;
            if term.negative {
                safe = format!("-({safe})");
            }
            let coefficient =
                Function::new(&safe).map_err(|e| format!("in term '{}': {}", term.text, e))?;
            let incrementor = self.registry.resolve(kind, &mut self.lags)?;
            process_terms.push(ProcessTerm {
                coefficient,
                incrementor: Some(incrementor),
            });
        }

        Ok(Process::new(
            index,
            name.to_string(),
            Vec::new(),
            false,
            process_terms,
        ))
    }

    fn build_algebraic(&mut self, index: usize, name: &str, rhs: &str) -> Result<Process, String> {
        check_process_name(name)?;
        // The whole right-hand side is ONE expression (no term splitting here).
        let safe = rewrite_lags(rhs, &mut self.lags)?;
        let coefficient =
            Function::new(&safe).map_err(|e| format!("in expression '{rhs}': {e}"))?;
        Ok(Process::new(
            index,
            name.to_string(),
            Vec::new(),
            true,
            vec![ProcessTerm {
                coefficient,
                incrementor: None,
            }],
        ))
    }
}

struct Term<'a> {
    negative: bool,
    text: &'a str,
    coefficient: String,
    incrementor: Option<IncKind<'a>>,
}

fn check_process_name(name: &str) -> Result<(), String> {
    if name == "t" || name == "dt" {
        Err(format!("'{name}' is a reserved name"))
    } else {
        Ok(())
    }
}

// ─────────────────────────────── incrementors ────────────────────────────────

#[derive(Clone, Copy)]
enum IncKind<'a> {
    Time,
    Wiener(&'a str),
    Jump { name: &'a str, intensity: &'a str },
}

/// Creates each incrementor once and hands out clones afterwards.
/// Random incrementors (dW*, dN*) get consecutive indices in order of first use;
/// `dt` has no index. This matches `ProcessUniverse::num_random_incrementors`.
#[derive(Default)]
struct IncrementorRegistry {
    by_name: HashMap<String, Incrementor>,
    intensities: HashMap<String, String>,
    next_index: usize,
}

impl IncrementorRegistry {
    fn resolve(&mut self, kind: IncKind<'_>, lags: &mut LagTable) -> Result<Incrementor, String> {
        match kind {
            IncKind::Time => Ok(self
                .by_name
                .entry("dt".to_string())
                .or_insert_with(|| Incrementor::Time(TimeIncrementor::new()))
                .clone()),

            IncKind::Wiener(name) => {
                if let Some(existing) = self.by_name.get(name) {
                    return Ok(existing.clone());
                }
                let inc =
                    Incrementor::Wiener(WienerIncrementor::new(self.next_index, name.to_string()));
                self.next_index += 1;
                self.by_name.insert(name.to_string(), inc.clone());
                Ok(inc)
            }

            IncKind::Jump { name, intensity } => {
                let normalized: String = intensity.split_whitespace().collect();
                if let Some(existing) = self.by_name.get(name) {
                    if self.intensities.get(name) != Some(&normalized) {
                        return Err(format!(
                            "jump incrementor '{name}' is used with two different intensities; \
                             give them different names (e.g. dN1, dN2)"
                        ));
                    }
                    return Ok(existing.clone());
                }
                let safe = rewrite_lags(intensity, lags)?;
                let lambda = Function::new(&safe)
                    .map_err(|e| format!("invalid jump intensity '{intensity}': {e}"))?;
                let inc = Incrementor::PoissonJump(PoissonJumpIncrementor::new(
                    self.next_index,
                    name.to_string(),
                    lambda,
                ));
                self.next_index += 1;
                self.by_name.insert(name.to_string(), inc.clone());
                self.intensities.insert(name.to_string(), normalized);
                Ok(inc)
            }
        }
    }
}

/// Recognises a trailing factor as an incrementor.
/// `Ok(None)` means "an ordinary factor", `Err` means "looks like a jump but is malformed".
fn classify_incrementor(factor: &str) -> Result<Option<IncKind<'_>>, String> {
    if factor == "dt" {
        return Ok(Some(IncKind::Time));
    }
    if let Some(rest) = factor.strip_prefix("dW") {
        return Ok(rest
            .chars()
            .all(is_ident_char)
            .then_some(IncKind::Wiener(factor)));
    }
    if let Some(rest) = factor.strip_prefix("dN") {
        return match rest.find('(') {
            None => {
                if rest.chars().all(is_ident_char) {
                    Err(format!(
                        "jump incrementor '{factor}' needs an intensity, e.g. {factor}(0.5)"
                    ))
                } else {
                    Ok(None)
                }
            }
            Some(p) => {
                if !rest[..p].chars().all(is_ident_char) {
                    return Ok(None);
                }
                let open = 2 + p; // byte offset of '(' inside `factor`
                if !factor.ends_with(')') || !closes_at_end(&factor[open..]) {
                    return Err(format!("malformed jump incrementor '{factor}'"));
                }
                let intensity = factor[open + 1..factor.len() - 1].trim();
                if intensity.is_empty() {
                    return Err(format!(
                        "jump incrementor '{factor}' has an empty intensity"
                    ));
                }
                Ok(Some(IncKind::Jump {
                    name: &factor[..open],
                    intensity,
                }))
            }
        };
    }
    Ok(None)
}

/// `s` starts with '(' — true if that parenthesis is closed by the very last character.
fn closes_at_end(s: &str) -> bool {
    let mut depth = 0i32;
    for (i, b) in s.bytes().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return i == s.len() - 1;
                }
            }
            _ => {}
        }
    }
    false
}

// ───────────────────────────── term splitting ────────────────────────────────

/// Splits `s` at top-level binary `+` / `-` (outside any `()` or `[]`).
/// Returns `(is_negative, term_text)`. Unary signs, exponent signs (`1e-5`) and
/// operators after other operators stay inside their term.
fn split_signed_terms(s: &str) -> Result<Vec<(bool, &str)>, String> {
    let mut terms = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut negative = false;

    for (i, b) in s.bytes().enumerate() {
        match b {
            b'(' | b'[' => depth += 1,
            b')' | b']' => {
                depth -= 1;
                if depth < 0 {
                    return Err("unbalanced closing bracket".into());
                }
            }
            b'+' | b'-' if depth == 0 && is_binary_sign(s, i) => {
                terms.push((negative, s[start..i].trim()));
                negative = b == b'-';
                start = i + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err("unbalanced opening bracket".into());
    }
    terms.push((negative, s[start..].trim()));

    if terms.iter().any(|(_, t)| t.is_empty()) {
        return Err("empty term (dangling '+' or '-')".into());
    }
    Ok(terms)
}

fn is_binary_sign(s: &str, i: usize) -> bool {
    let before = s[..i].trim_end();
    let Some(prev) = before.bytes().last() else {
        return false; // leading sign
    };
    if matches!(
        prev,
        b'+' | b'-'
            | b'*'
            | b'/'
            | b'%'
            | b'^'
            | b'('
            | b'['
            | b','
            | b'<'
            | b'>'
            | b'='
            | b'!'
            | b'&'
            | b'|'
            | b'?'
            | b':'
    ) {
        return false; // unary sign after an operator
    }
    // Exponent sign of a numeric literal such as `1e-5` or `2.5E+3`.
    if matches!(prev, b'e' | b'E') && before.len() == i {
        let body = &before[..before.len() - 1];
        let tok_start = body
            .rfind(|c: char| !(c.is_ascii_digit() || c == '.'))
            .map_or(0, |p| p + 1);
        let mantissa = &body[tok_start..];
        let after_ident = body[..tok_start]
            .chars()
            .last()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
        if !after_ident && mantissa.bytes().any(|c| c.is_ascii_digit()) {
            return false;
        }
    }
    true
}

/// Splits one term into (coefficient, incrementor). Terms without an incrementor
/// are returned whole with `None`.
fn split_term(text: &str) -> Result<(String, Option<IncKind<'_>>), String> {
    if let Some(idx) = last_top_level_star(text) {
        let factor = text[idx + 1..].trim();
        if let Some(kind) = classify_incrementor(factor)? {
            let coefficient = text[..idx].trim();
            if coefficient.is_empty() {
                return Err(format!("missing coefficient before '{factor}'"));
            }
            return Ok((coefficient.to_string(), Some(kind)));
        }
        return Ok((text.to_string(), None));
    }

    // No '*': a bare incrementor (optionally signed), e.g. `dt` or `-dW1`.
    let (negative, body) = match text.strip_prefix('-') {
        Some(rest) => (true, rest.trim()),
        None => (false, text.strip_prefix('+').map_or(text, str::trim)),
    };
    if let Some(kind) = classify_incrementor(body)? {
        return Ok((String::from(if negative { "-1" } else { "1" }), Some(kind)));
    }
    Ok((text.to_string(), None))
}

fn last_top_level_star(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    for (i, b) in s.bytes().enumerate().rev() {
        match b {
            b')' | b']' => depth += 1,
            b'(' | b'[' => depth -= 1,
            b'*' if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(is_ident_char)
}

// ───────────────────────────────── lags ──────────────────────────────────────

/// Collects every lagged variable used anywhere, keyed by the base process name.
#[derive(Default)]
struct LagTable {
    by_base: BTreeMap<String, Vec<ProcessLag>>,
}

impl LagTable {
    /// Registers a lag and returns the variable name the expression should use.
    fn register(&mut self, base: &str, lag: LagType) -> String {
        let name = match &lag {
            LagType::Size(n) => format!("{base}_LAG_{n}"),
            LagType::Time(t) => format!(
                "{base}_TIME_LAG_{}",
                t.into_inner().to_string().replace('.', "_")
            ),
        };
        let entry = self.by_base.entry(base.to_string()).or_default();
        if !entry.iter().any(|l| l.process_name == name) {
            entry.push(ProcessLag {
                value: lag,
                process_name: name.clone(),
            });
        }
        name
    }
}

fn bracket_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"([A-Za-z_][A-Za-z0-9_]*)\s*\[([^\]]*)\]").expect("valid bracket regex")
    })
}

fn time_lag_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^(?:time|t)\s*:?\s*(?:([+-])\s*([0-9]*\.?[0-9]+))?$")
            .expect("valid time-lag regex")
    })
}

fn index_lag_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^([+-]?)\s*([0-9]+)$").expect("valid index-lag regex"))
}

/// `Ok(None)` = zero lag (use the current value).
fn parse_lag(inner: &str) -> Result<Option<LagType>, String> {
    let inner = inner.trim();
    let future = || format!("'[{inner}]' looks into the future; lags must be negative");

    if let Some(c) = time_lag_regex().captures(inner) {
        let Some(sign) = c.get(1) else {
            return Ok(None); // plain `X[t]`
        };
        let value: f64 = c[2]
            .parse()
            .map_err(|_| format!("invalid time lag '[{inner}]'"))?;
        if value == 0.0 {
            return Ok(None);
        }
        if sign.as_str() == "+" {
            return Err(future());
        }
        return Ok(Some(LagType::Time(OrderedFloat(value))));
    }

    if let Some(c) = index_lag_regex().captures(inner) {
        let n: usize = c[2]
            .parse()
            .map_err(|_| format!("lag '[{inner}]' is too large"))?;
        if n == 0 {
            return Ok(None);
        }
        if &c[1] == "-" {
            return Ok(Some(LagType::Size(n)));
        }
        return Err(future());
    }

    Err(format!(
        "unsupported lag '[{inner}]' (use [-2], [0], [t-1.5] or [time:-1.5])"
    ))
}

/// Replaces every `X[...]` by a plain variable name (registering the lag) and
/// rejects any bracket that is left over.
fn rewrite_lags(expr: &str, lags: &mut LagTable) -> Result<String, String> {
    let mut out = String::with_capacity(expr.len());
    let mut last = 0usize;

    for caps in bracket_regex().captures_iter(expr) {
        let whole = caps.get(0).ok_or("internal error: empty match")?;
        out.push_str(&expr[last..whole.start()]);
        let base = &caps[1];
        match parse_lag(&caps[2])? {
            None => out.push_str(base),
            Some(lag) => out.push_str(&lags.register(base, lag)),
        }
        last = whole.end();
    }
    out.push_str(&expr[last..]);

    if out.contains('[') || out.contains(']') {
        return Err(format!(
            "unsupported bracket syntax in '{expr}' (brackets are only valid as 'name[lag]')"
        ));
    }
    Ok(out)
}

// ───────────────────────────────── tests ─────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn eqs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn splits_only_top_level_binary_signs() {
        let t = split_signed_terms("(mu + X) * dt - 0.2*X*dW1 + 1e-5 * dt").unwrap();
        assert_eq!(
            t,
            vec![
                (false, "(mu + X) * dt"),
                (true, "0.2*X*dW1"),
                (false, "1e-5 * dt")
            ]
        );
    }

    #[test]
    fn unary_signs_and_brackets_are_not_split() {
        let t = split_signed_terms("-X * dt + Y[t - 1.5] * dW1").unwrap();
        assert_eq!(t, vec![(false, "-X * dt"), (false, "Y[t - 1.5] * dW1")]);
    }

    #[test]
    fn rewrites_all_lag_forms() {
        let mut lags = LagTable::default();
        let out = rewrite_lags("X1[-2] + X1[t-1.5] + X1[0] + Y[time:-0.5]", &mut lags).unwrap();
        assert_eq!(out, "X1_LAG_2 + X1_TIME_LAG_1_5 + X1 + Y_TIME_LAG_0_5");
        assert_eq!(lags.by_base["X1"].len(), 2);
    }

    #[test]
    fn rejects_future_and_unknown_lags() {
        for bad in ["X[1]", "X[+2]", "X[t+1]", "X[t:0.5]", "X[foo]", "(X)[-1]"] {
            assert!(
                rewrite_lags(bad, &mut LagTable::default()).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn parses_the_example_system() {
        let u = parse_equations(&eqs(&[
            "dX1 = ( sin(t) ) * dt + (0.01 * X1) * dW1 + (0.001 * X1) * dN1(0.5 * cos(t))",
            "X2 = max(X1 - 100.0, 0.0)",
        ]))
        .unwrap();
        assert_eq!(u.num_processes(), 2);
        assert_eq!(u.num_random_incrementors(), 2);
        assert_eq!(u.levy_processes().count(), 1);
        let x1 = u.processes().find(|p| p.name == "X1").unwrap();
        assert_eq!(x1.terms.len(), 3);
        let x2 = u.processes().find(|p| p.name == "X2").unwrap();
        assert!(x2.is_algebraic);
    }

    #[test]
    fn handles_plus_inside_parentheses_and_minus_terms() {
        let u = parse_equations(&eqs(&["dX = (0.1 + X) * dt - 0.2 * X * dW1"])).unwrap();
        assert_eq!(u.processes().next().unwrap().terms.len(), 2);
    }

    #[test]
    fn algebraic_names_may_start_with_d() {
        let u = parse_equations(&eqs(&["dX = 0.1 * X * dt", "discount = 0.9 * X"])).unwrap();
        let d = u.processes().find(|p| p.name == "discount").unwrap();
        assert!(d.is_algebraic);
    }

    #[test]
    fn registers_lags_on_their_process() {
        let u = parse_equations(&eqs(&["dX = X[-1] * dt + X[t-0.5] * dW1"])).unwrap();
        assert!(u.get_variable("X_LAG_1").is_some());
        assert!(u.get_variable("X_TIME_LAG_0_5").is_some());
    }

    #[test]
    fn reports_errors() {
        // algebraic equation containing an incrementor
        assert!(parse_equations(&eqs(&["X2 = X1 * dt"])).is_err());
        // differential term without incrementor
        assert!(parse_equations(&eqs(&["dX = 0.1 * X * dt + 5"])).is_err());
        // jump without intensity / with conflicting intensities
        assert!(parse_equations(&eqs(&["dX = X * dN1"])).is_err());
        assert!(parse_equations(&eqs(&["dX = X * dN1(0.5) + X * dN1(0.7)"])).is_err());
        // duplicate process
        assert!(parse_equations(&eqs(&["dX = X * dt", "X = 1"])).is_err());
        // lag of an unknown process
        assert!(parse_equations(&eqs(&["dX = Y[-1] * dt"])).is_err());
    }
}
