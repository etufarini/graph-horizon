/*
 * Graph Orizon app mode
 * Closed vocabulary of process modes. It converts the raw `--mode` flag into a
 * small enum and applies the default CLI mode when the flag is absent.
 */

use color_eyre::eyre::{Result, eyre};

use crate::app::args;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    Cli,
    Server,
    Web,
}

pub(super) fn selected() -> Result<Mode> {
    match args::value("--mode").as_deref().map(str::trim) {
        None | Some("") | Some("cli") => Ok(Mode::Cli),
        Some("server") => Ok(Mode::Server),
        Some("web") => Ok(Mode::Web),
        Some(_) => Err(eyre!("mode non valida")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_names_are_stable() {
        assert_eq!(Mode::Cli, Mode::Cli);
        assert_ne!(Mode::Cli, Mode::Server);
        assert_ne!(Mode::Server, Mode::Web);
    }
}
