/*
 * Graph Horizon app arguments
 * Single responsibility: own the chat-only runtime flag table, parser, and
 * usage text. It accepts only model/server/generation/backend configuration,
 * depends only on `std`, and does not implement tools or reasoning controls.
 */

use std::sync::OnceLock;

// Whether a flag carries a value (`--flag value`) or is a boolean toggle whose mere
// presence enables it (`--flag`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FlagKind {
    Value,
    Bool,
}

// The runtime flags, in the order shown by `usage`. This table is the single source of
// truth: the parser only accepts a `--flag` listed here, and `usage` is generated from
// it. Removed runtime/tool flags are deliberately absent so they fail as unknown.
const FLAGS: &[(&str, FlagKind)] = &[
    ("--model", FlagKind::Value),
    ("--mode", FlagKind::Value),
    ("--provider", FlagKind::Value),
    ("--host", FlagKind::Value),
    ("--port", FlagKind::Value),
    ("--context-tokens", FlagKind::Value),
    ("--system-prompt", FlagKind::Value),
    ("--base-url", FlagKind::Value),
    ("--max-tokens", FlagKind::Value),
    ("--vram-weights-percent", FlagKind::Value),
    ("--vram-reserve-mib", FlagKind::Value),
    ("--cpu-threads", FlagKind::Value),
    ("--kv-quant", FlagKind::Value),
    ("--no-attn-simd", FlagKind::Bool),
];

// The parsed arguments, written exactly once by `init`: value flags as `(flag, value)`
// pairs and boolean flags as the set of flags seen. The default an empty parse installs
// means "no flags were passed", so every parameter falls back to its default.
struct Parsed {
    values: Vec<(&'static str, String)>,
    bools: Vec<&'static str>,
}

static PARSED: OnceLock<Parsed> = OnceLock::new();

// Parses `std::env::args()` once and records the recognized flags. Call it as the very
// first thing in `main`, before the terminal is initialized, so a bad argument fails
// cleanly. `--help`/`-h` prints the usage and exits 0; an unknown flag or a value flag
// missing its value prints the usage and exits non-zero. A boolean flag never consumes
// the following token. A second call is a no-op (the OnceLock is already set).
pub(crate) fn init() {
    if PARSED.get().is_some() {
        return;
    }

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut values: Vec<(&'static str, String)> = Vec::new();
    let mut bools: Vec<&'static str> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--help" || arg == "-h" {
            println!("{}", usage());
            std::process::exit(0);
        }
        // The static flag name (not the user's slice) so the parsed entries carry a
        // `'static` lifetime, matching `value`/`is_present` lookups.
        let entry = FLAGS.iter().find(|(flag, _)| *flag == arg);
        match entry {
            Some((flag, FlagKind::Value)) => {
                // The value is the next argument; its absence (flag passed last) is an
                // error (E-FLAG-VALUE-MISSING), never a silent empty value.
                let Some(value) = args.get(i + 1) else {
                    eprintln!("valore mancante per {arg}");
                    eprintln!("{}", usage());
                    std::process::exit(1);
                };
                values.push((flag, value.clone()));
                i += 2;
            }
            Some((flag, FlagKind::Bool)) => {
                // Presence-only: do NOT consume the following token (it is the next flag
                // to process, or an unknown-flag error).
                bools.push(flag);
                i += 1;
            }
            None => {
                eprintln!("argomento non riconosciuto: {arg}");
                eprintln!("{}", usage());
                std::process::exit(1);
            }
        }
    }

    // Ignore a lost race: whoever set it first wins, the values are identical.
    let _ = PARSED.set(Parsed { values, bools });
}

// Resolves one value flag: the parsed value when present, otherwise `None` (the call
// site applies the default). NO environment fallback — the CLI is the single authority.
// Safe to call before `init`: an unset OnceLock reads as "no flags".
pub(crate) fn value(flag: &str) -> Option<String> {
    PARSED.get().and_then(|parsed| {
        parsed
            .values
            .iter()
            .find(|(f, _)| *f == flag)
            .map(|(_, v)| v.clone())
    })
}

// True when a boolean flag was passed. Safe to call before `init` (reads as absent).
pub(crate) fn is_present(flag: &str) -> bool {
    PARSED
        .get()
        .map(|parsed| parsed.bools.contains(&flag))
        .unwrap_or(false)
}

// Help text, generated from the FLAGS table so the listing can never drift. Value flags
// show `--flag <valore>`, boolean flags just `--flag`. No environment variables are
// mentioned: the CLI is the single source of configuration.
pub(crate) fn usage() -> String {
    let mut out = String::from(
        "Uso: graph-horizon [opzioni]\n\n\
         La configurazione avviene solo via flag (precedenza: flag > default).\n\n\
         Opzioni:\n",
    );
    for (flag, kind) in FLAGS {
        let shown = match kind {
            FlagKind::Value if *flag == "--kv-quant" => format!("{flag} <f16|int8>"),
            FlagKind::Value => format!("{flag} <valore>"),
            FlagKind::Bool => flag.to_string(),
        };
        out.push_str(&format!("  {shown}\n"));
    }
    out.push_str("  --help, -h\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_matrix_e12_flag_table_rejects_removed_flags() {
        let kind = |flag: &str| FLAGS.iter().find(|(f, _)| *f == flag).map(|(_, k)| *k);
        assert_eq!(kind("--model"), Some(FlagKind::Value));
        assert_eq!(kind("--max-tokens"), Some(FlagKind::Value));
        assert_eq!(kind("--cpu-threads"), Some(FlagKind::Value));
        assert_eq!(kind("--kv-quant"), Some(FlagKind::Value));
        assert_eq!(kind("--no-attn-simd"), Some(FlagKind::Bool));
        for removed in [
            "--think",
            "--profile",
            "--workspace",
            "--no-calibration",
            "--kv-attention-on-cpu",
        ] {
            assert_eq!(kind(removed), None, "{removed} must be rejected");
        }
        assert_eq!(kind("--unknown"), None);
    }

    #[test]
    fn usage_lists_new_flags_without_env() {
        let u = usage();
        // The engine knobs appear.
        for f in [
            "--vram-weights-percent",
            "--vram-reserve-mib",
            "--cpu-threads",
            "--kv-quant",
            "--no-attn-simd",
        ] {
            assert!(u.contains(f), "usage missing {f}");
        }
        for f in [
            "--think",
            "--profile",
            "--workspace",
            "--kv-attention-on-cpu",
            "--no-calibration",
        ] {
            assert!(!u.contains(f), "usage still lists removed flag {f}");
        }
        // Value flags show their placeholder; booleans do not.
        assert!(u.contains("--cpu-threads <valore>"));
        assert!(u.contains("--kv-quant <f16|int8>"));
        assert!(u.contains("--no-attn-simd\n"));
        // No environment variable is mentioned anywhere.
        assert!(!u.to_lowercase().contains("env"));
        assert!(!u.contains("GRAPH_HORIZON_"));
    }
}
