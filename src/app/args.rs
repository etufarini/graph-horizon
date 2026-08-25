/*
 * Graph Horizon app arguments
 * Single responsibility: own the CLI and Web runtime flag table, parser, and
 * usage text. It accepts only model/generation/backend and Web bind settings,
 * depends only on `std`, and does not implement tools or reasoning controls.
 */

use std::sync::OnceLock;

// The runtime flags, in the order shown by `usage`. This table is the single source of
// truth: the parser only accepts a `--flag` listed here, and `usage` is generated from
// it. Removed runtime/tool flags are deliberately absent so they fail as unknown.
const FLAGS: &[&str] = &[
    "--model",
    "--mode",
    "--host",
    "--port",
    "--search-url",
    "--search-key-file",
    "--context-tokens",
    "--system-prompt",
    "--max-tokens",
    "--vram-weights-percent",
    "--vram-reserve-mib",
    "--cpu-threads",
    "--kv-quant",
];

// The parsed arguments are written exactly once by `init`. The default empty parse
// means "no flags were passed", so every parameter falls back to its default.
struct Parsed {
    values: Vec<(&'static str, String)>,
}

static PARSED: OnceLock<Parsed> = OnceLock::new();

// Parses `std::env::args()` once and records the recognized flags. Call it as the very
// first thing in `main`, before the terminal is initialized, so a bad argument fails
// cleanly. Help and version requests print and exit 0; an unknown flag or a value flag
// missing its value prints the usage and exits non-zero. A second call is a no-op
// because the parse is already installed.
pub(crate) fn init() {
    if PARSED.get().is_some() {
        return;
    }

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut values: Vec<(&'static str, String)> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--help" || arg == "-h" {
            println!("{}", usage());
            std::process::exit(0);
        }
        if arg == "--version" || arg == "-V" {
            println!("graph-horizon {}", env!("CARGO_PKG_VERSION"));
            std::process::exit(0);
        }
        // The static flag name (not the user's slice) so the parsed entries carry a
        // `'static` lifetime, matching `value` lookups.
        let entry = FLAGS.iter().find(|flag| **flag == arg);
        match entry {
            Some(flag) => {
                // The value is the next argument; its absence (flag passed last) is an
                // error (E-FLAG-VALUE-MISSING), never a silent empty value.
                let Some(value) = args.get(i + 1) else {
                    eprintln!("missing value for {arg}");
                    eprintln!("{}", usage());
                    std::process::exit(1);
                };
                values.push((flag, value.clone()));
                i += 2;
            }
            None => {
                eprintln!("unrecognized argument: {arg}");
                eprintln!("{}", usage());
                std::process::exit(1);
            }
        }
    }

    // Ignore a lost race: whoever set it first wins, the values are identical.
    let _ = PARSED.set(Parsed { values });
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

// Help text, generated from the FLAGS table so the listing can never drift. No
// environment variables are mentioned: the CLI is the single source of configuration.
pub(crate) fn usage() -> String {
    let mut out = String::from(
        "Usage: graph-horizon [options]\n\n\
         Configuration uses flags only (precedence: flag > default).\n\n\
         Options:\n",
    );
    for flag in FLAGS {
        let shown = match *flag {
            "--mode" => format!("{flag} <cli|web>"),
            "--kv-quant" => format!("{flag} <f16|int8>"),
            _ => format!("{flag} <value>"),
        };
        out.push_str(&format!("  {shown}\n"));
    }
    out.push_str("  --version, -V\n  --help, -h\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_matrix_e12_flag_table_rejects_removed_flags() {
        assert!(FLAGS.contains(&"--model"));
        assert!(FLAGS.contains(&"--max-tokens"));
        assert!(FLAGS.contains(&"--cpu-threads"));
        assert!(FLAGS.contains(&"--kv-quant"));
        assert!(FLAGS.contains(&"--search-url"));
        assert!(FLAGS.contains(&"--search-key-file"));
        for removed in [
            "--provider",
            "--base-url",
            "--think",
            "--profile",
            "--workspace",
            "--no-calibration",
            "--kv-attention-on-cpu",
            "--no-attn-simd",
        ] {
            assert!(!FLAGS.contains(&removed), "{removed} must be rejected");
        }
        assert!(!FLAGS.contains(&"--unknown"));
    }

    #[test]
    fn usage_lists_new_flags_without_env() {
        let u = usage();
        for f in [
            "--vram-weights-percent",
            "--vram-reserve-mib",
            "--cpu-threads",
            "--kv-quant",
            "--search-url",
            "--search-key-file",
        ] {
            assert!(u.contains(f), "usage missing {f}");
        }
        for f in [
            "--provider",
            "--base-url",
            "--think",
            "--profile",
            "--workspace",
            "--kv-attention-on-cpu",
            "--no-calibration",
            "--no-attn-simd",
        ] {
            assert!(!u.contains(f), "usage still lists removed flag {f}");
        }
        assert!(u.contains("--mode <cli|web>"));
        assert!(u.contains("--cpu-threads <value>"));
        assert!(u.contains("--kv-quant <f16|int8>"));
        assert!(u.contains("--version, -V"));
        assert!(!u.to_lowercase().contains("env"));
        assert!(!u.contains("GRAPH_HORIZON_"));
    }
}
