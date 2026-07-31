/*
 * GH Zero app domain
 * Single responsibility: validate chat runtime configuration and dispatch one
 * selected surface. It captures the immutable startup file authority before
 * dispatch and does not initialize tools, mutable workspaces, or reasoning.
 */

use color_eyre::eyre::Result;

use mode::Mode;

pub(crate) mod args;
pub(crate) mod engine;
mod mode;
pub(crate) mod sse;

pub(crate) async fn run() -> Result<()> {
    // Validate engine flags before cwd changes or model loads so every mode
    // fails fast on malformed numeric input.
    let _ = engine::config::engine_config(None);

    let files = crate::gh_zero_cli::plugins::attachments::FileAuthority::capture()
        .map_err(|_| color_eyre::eyre::eyre!("startup directory is unavailable"))?;
    let mode = mode::selected()?;
    dispatch(mode, args::value("--model"), files).await
}

async fn dispatch(
    mode: Mode,
    model_path: Option<String>,
    files: crate::gh_zero_cli::plugins::attachments::FileAuthority,
) -> Result<()> {
    match mode {
        Mode::Cli => crate::gh_zero_cli::startup::run(model_path, files).await,
        Mode::Server => crate::gh_zero_server::startup::run(model_path).await,
        Mode::Web => crate::gh_zero_web::startup::run(model_path).await,
    }
}
