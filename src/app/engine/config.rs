/*
 * Graph Orizon app engine config
 * Single responsibility: convert the chat-only parsed runtime flags into
 * `EngineConfig`. It validates numeric/backend knobs, depends only on app args
 * and graph_orizon_engine types, and does not expose tools or reasoning controls.
 */

use graph_orizon_engine::{EngineConfig, KvQuant};

use crate::app::args;

// None means AUTO. Explicit values must be in 0..=100; 0 is the hybrid CPU-only
// override and values above 100 are startup errors, never silently clamped.
fn parse_percent(raw: Option<&str>) -> Result<Option<u8>, String> {
    match raw {
        None => Ok(None),
        Some(s) => match s.parse::<u8>() {
            Ok(v) if v <= 100 => Ok(Some(v)),
            _ => Err(format!(
                "--vram-weights-percent: valore non valido '{s}' (atteso un intero tra 0 e 100)"
            )),
        },
    }
}

// None means system parallelism. A value must be an integer >= 1; `0` and
// non-numeric are usage errors.
fn parse_threads(raw: Option<&str>) -> Result<Option<usize>, String> {
    match raw {
        None => Ok(None),
        Some(s) => match s.parse::<usize>() {
            Ok(n) if n >= 1 => Ok(Some(n)),
            _ => Err(format!(
                "--cpu-threads: valore non valido '{s}' (atteso un intero ≥ 1)"
            )),
        },
    }
}

// Context override shared by every local surface. Explicit values are strict:
// absence delegates to the engine's versioned policy, while invalid or zero
// values fail before load.
fn parse_context_tokens(raw: Option<&str>) -> Result<Option<usize>, String> {
    match raw {
        None => Ok(None),
        Some(s) => match s.parse::<usize>() {
            Ok(n) if n >= 1 => Ok(Some(n)),
            _ => Err(format!(
                "--context-tokens: valore non valido '{s}' (atteso un intero ≥ 1)"
            )),
        },
    }
}

pub(crate) fn context_tokens_from_args() -> Option<usize> {
    or_exit(parse_context_tokens(
        args::value("--context-tokens").as_deref(),
    ))
}

// None means the engine's hardware-agnostic default. A value must be an integer
// >= 0 (MiB); negative/non-numeric are usage errors.
fn parse_reserve_mib(raw: Option<&str>) -> Result<Option<u64>, String> {
    match raw {
        None => Ok(None),
        Some(s) => match s.parse::<u64>() {
            Ok(n) => {
                // The engine consumes MiB; reject only arithmetic overflow before
                // any later conversion to bytes can wrap.
                let _ = n
                    .checked_mul(1024 * 1024)
                    .ok_or_else(|| format!("--vram-reserve-mib: valore troppo grande '{s}'"))?;
                Ok(Some(n))
            }
            Err(_) => Err(format!(
                "--vram-reserve-mib: valore non valido '{s}' (atteso un intero ≥ 0)"
            )),
        },
    }
}

// None means `f16`; an explicit `f16` is identical to the flag being absent. Any
// other value must parse via `KvQuant::parse` (lowercase, case-sensitive); an
// invalid one is a usage error listing the valid values.
fn parse_kv_quant(raw: Option<&str>) -> Result<KvQuant, String> {
    match raw {
        None => Ok(KvQuant::default()),
        Some(s) => KvQuant::parse(s).ok_or_else(|| {
            let valid: Vec<&str> = KvQuant::ALL.iter().map(|q| q.name()).collect();
            format!(
                "--kv-quant: valore non valido '{s}' (valori validi: {})",
                valid.join(", ")
            )
        }),
    }
}

fn parse_max_tokens(raw: Option<&str>) -> Result<usize, String> {
    match raw {
        None => Ok(2048),
        Some(s) => match s.parse::<usize>() {
            Ok(n) => Ok(n),
            _ => Err(format!(
                "--max-tokens: valore non valido '{s}' (atteso un intero ≥ 0)"
            )),
        },
    }
}

pub(crate) fn max_tokens_from_args() -> usize {
    or_exit(parse_max_tokens(args::value("--max-tokens").as_deref()))
}

// Prints the usage error to stderr + the help, then exits non-zero (no load started).
fn or_exit<T>(result: Result<T, String>) -> T {
    match result {
        Ok(value) => value,
        Err(message) => {
            eprintln!("{message}");
            eprintln!("{}", args::usage());
            std::process::exit(1);
        }
    }
}

// Builds the validated `EngineConfig` from parsed flags. A caller-resolved
// context wins; otherwise the optional CLI value is parsed here so the common
// startup validation rejects bad input before any surface or model starts.
pub(crate) fn engine_config(context_tokens: Option<usize>) -> EngineConfig {
    EngineConfig {
        context_tokens: context_tokens.or_else(context_tokens_from_args),
        vram_weights_percent: or_exit(parse_percent(
            args::value("--vram-weights-percent").as_deref(),
        )),
        vram_reserve_mib: or_exit(parse_reserve_mib(
            args::value("--vram-reserve-mib").as_deref(),
        )),
        cpu_threads: or_exit(parse_threads(args::value("--cpu-threads").as_deref())),
        no_attn_simd: args::is_present("--no-attn-simd"),
        kv_quant: or_exit(parse_kv_quant(args::value("--kv-quant").as_deref())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percent_validation() {
        assert_eq!(parse_percent(None), Ok(None)); // absent ⇒ AUTO
        assert_eq!(parse_percent(Some("50")), Ok(Some(50)));
        assert_eq!(parse_percent(Some("0")), Ok(Some(0)));
        assert_eq!(parse_percent(Some("100")), Ok(Some(100)));
        assert!(parse_percent(Some("101")).is_err());
        assert!(parse_percent(Some("abc")).is_err());
        assert!(parse_percent(Some("-1")).is_err());
        assert!(parse_percent(Some("50%")).is_err());
    }

    #[test]
    fn context_tokens_validation_is_strict() {
        assert_eq!(parse_context_tokens(None), Ok(None));
        assert_eq!(parse_context_tokens(Some("2000")), Ok(Some(2000)));
        assert!(parse_context_tokens(Some("0")).is_err());
        assert!(parse_context_tokens(Some("abc")).is_err());
        assert!(parse_context_tokens(Some("-1")).is_err());
    }

    #[test]
    fn threads_validation() {
        assert_eq!(parse_threads(None), Ok(None));
        assert_eq!(parse_threads(Some("8")), Ok(Some(8)));
        assert!(parse_threads(Some("0")).is_err());
        assert!(parse_threads(Some("x")).is_err());
        assert!(parse_threads(Some("-2")).is_err());
    }

    #[test]
    fn kv_quant_validation() {
        // Absent and explicit `f16` are identical.
        assert_eq!(parse_kv_quant(None), Ok(KvQuant::F16));
        assert_eq!(parse_kv_quant(Some("f16")), Ok(KvQuant::F16));
        assert_eq!(parse_kv_quant(Some("int8")), Ok(KvQuant::Int8));
        // Case-sensitive (D8) and unknown values list the valid set.
        let err = parse_kv_quant(Some("INT8")).unwrap_err();
        assert!(err.contains("valori validi: f16, int8"), "{err}");
        assert!(parse_kv_quant(Some("bogus")).is_err());
    }

    #[test]
    fn reserve_validation() {
        assert_eq!(parse_reserve_mib(None), Ok(None));
        assert_eq!(parse_reserve_mib(Some("0")), Ok(Some(0)));
        assert_eq!(parse_reserve_mib(Some("512")), Ok(Some(512)));
        assert!(parse_reserve_mib(Some("x")).is_err());
        assert!(parse_reserve_mib(Some("-5")).is_err());
    }

    #[test]
    fn max_tokens_validation() {
        assert_eq!(parse_max_tokens(None), Ok(2048));
        assert_eq!(parse_max_tokens(Some("0")), Ok(0));
        assert_eq!(parse_max_tokens(Some("128")), Ok(128));
        assert!(parse_max_tokens(Some("x")).is_err());
        assert!(parse_max_tokens(Some("-1")).is_err());
    }
}
