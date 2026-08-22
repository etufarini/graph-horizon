/*
 * Graph Horizon app mode
 * Closed vocabulary of process modes. It converts the raw `--mode` flag into a
 * small enum and applies the default CLI mode when the flag is absent.
 */

use color_eyre::eyre::{Result, eyre};

use crate::app::args;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    Cli,
    Web,
}

pub(super) fn selected() -> Result<Mode> {
    from_value(args::value("--mode").as_deref())
}

fn from_value(value: Option<&str>) -> Result<Mode> {
    match value.map(str::trim) {
        None | Some("") | Some("cli") => Ok(Mode::Cli),
        Some("web") => Ok(Mode::Web),
        Some(_) => Err(eyre!("invalid mode")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_names_are_stable() {
        assert_eq!(Mode::Cli, Mode::Cli);
        assert_ne!(Mode::Cli, Mode::Web);
    }

    #[test]
    fn legacy_server_mode_is_rejected() {
        assert!(from_value(Some("server")).is_err());
    }
}
