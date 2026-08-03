/*
 * GH Zero performance comparator CLI
 * Accepts only the two approved command shapes, coordinates strict evidence
 * validation and pure decision evaluation, and emits one bounded JSON summary.
 * It owns no JSONL field rules, performance formula, or model execution.
 */

mod decision;
mod evidence;

use serde::Serialize;

enum Command {
    Validate(String),
    Compare {
        baseline: String,
        candidate: String,
        target: String,
        attempt: u8,
    },
}

#[derive(Serialize)]
struct Validation<'a> {
    schema_version: u8,
    validation: &'static str,
    revision: &'a str,
    total: u8,
    pass: u64,
    fail: u64,
    external_verification: u64,
}

fn main() {
    let args = match std::env::args_os()
        .skip(1)
        .map(|value| value.into_string().map_err(|_| ()))
        .collect::<Result<Vec<_>, _>>()
        .and_then(parse)
    {
        Ok(args) => args,
        Err(()) => invalid("compare: invalid arguments"),
    };
    match run(args) {
        Ok((json, code)) => {
            println!("{json}");
            std::process::exit(code);
        }
        Err(()) => invalid("compare: invalid evidence"),
    }
}

fn run(command: Command) -> Result<(String, i32), ()> {
    match command {
        Command::Validate(path) => {
            let matrix = evidence::load(&path)?;
            let summary = Validation {
                schema_version: 1,
                validation: "valid",
                revision: &matrix.revision,
                total: 30,
                pass: matrix.counts.pass,
                fail: matrix.counts.fail,
                external_verification: matrix.counts.external,
            };
            Ok((serde_json::to_string(&summary).map_err(|_| ())?, 0))
        }
        Command::Compare {
            baseline,
            candidate,
            target,
            attempt,
        } => {
            let baseline = evidence::load(&baseline)?;
            let candidate = evidence::load(&candidate)?;
            evidence::comparable_tuples(&baseline, &candidate)?;
            let result = decision::evaluate(&baseline, &candidate, &target, attempt);
            let code = result.exit_code();
            Ok((serde_json::to_string(&result).map_err(|_| ())?, code))
        }
    }
}

fn parse(args: Vec<String>) -> Result<Command, ()> {
    match args.as_slice() {
        [flag, path] if flag == "--validate" && !path.is_empty() => {
            Ok(Command::Validate(path.clone()))
        }
        [
            baseline_flag,
            baseline,
            candidate_flag,
            candidate,
            target_flag,
            target,
            attempt_flag,
            attempt,
        ] if baseline_flag == "--baseline"
            && candidate_flag == "--candidate"
            && target_flag == "--target"
            && attempt_flag == "--attempt"
            && !baseline.is_empty()
            && !candidate.is_empty()
            && matches!(target.as_str(), "prefill" | "decode" | "both")
            && matches!(attempt.as_str(), "1" | "2") =>
        {
            Ok(Command::Compare {
                baseline: baseline.clone(),
                candidate: candidate.clone(),
                target: target.clone(),
                attempt: attempt.parse().map_err(|_| ())?,
            })
        }
        _ => Err(()),
    }
}

fn invalid(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_accepts_only_the_two_exact_modes() {
        assert!(matches!(
            parse(vec!["--validate".into(), "matrix".into()]),
            Ok(Command::Validate(_))
        ));
        let compare = "--baseline a --candidate b --target both --attempt 2"
            .split_whitespace()
            .map(str::to_owned)
            .collect();
        assert!(matches!(
            parse(compare),
            Ok(Command::Compare { attempt: 2, .. })
        ));
        for invalid in [
            "",
            "--validate",
            "--validate a trailing",
            "--baseline a --candidate b --target both",
            "--baseline a --candidate b --target other --attempt 1",
            "--candidate b --baseline a --target both --attempt 1",
        ] {
            assert!(parse(invalid.split_whitespace().map(str::to_owned).collect()).is_err());
        }
    }
}
