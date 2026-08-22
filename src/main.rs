/*
 * Graph Horizon binary entrypoint
 * Neutral shell of the binary: installs color_eyre, initializes runtime
 * arguments through `app::args`, and delegates to `app::run`. It does not know
 * how CLI or web mode are started.
 */

mod app;
mod graph_horizon_cli;
mod graph_horizon_web;
#[cfg(test)]
mod support_scripts;

#[tokio::main]
async fn main() {
    if color_eyre::config::HookBuilder::default()
        .display_location_section(false)
        .display_env_section(false)
        .install()
        .is_err()
    {
        eprintln!("startup failed");
        std::process::exit(1);
    }
    app::args::init();
    if let Err(error) = app::run().await {
        // Startup errors are already normalized at their boundary; Display emits
        // only the public top-level message, never a report chain or backtrace.
        eprintln!("{error}");
        std::process::exit(1);
    }
}
