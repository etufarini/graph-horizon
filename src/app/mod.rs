/*
 * Graph Horizon app domain
 * Single responsibility: validate chat runtime configuration and dispatch one
 * selected surface. Surface-specific resource ownership stays at each startup
 * boundary; this module does not initialize tools or mutable workspaces.
 */

use color_eyre::eyre::Result;

use mode::Mode;

pub(crate) mod args;
pub(crate) mod engine;
mod mode;

pub(crate) async fn run() -> Result<()> {
    // Validate the shared EngineConfig flags before cwd changes or model loads so every mode
    // fails fast on malformed numeric input.
    let _ = engine::config::engine_config(None);

    let mode = mode::selected()?;
    dispatch(mode, args::value("--model")).await
}

async fn dispatch(mode: Mode, model_path: Option<String>) -> Result<()> {
    match mode {
        Mode::Cli => crate::graph_horizon_cli::startup::run(model_path).await,
        Mode::Web => crate::graph_horizon_web::startup::run(model_path).await,
    }
}
