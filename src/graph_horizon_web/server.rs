/*
 * Graph Horizon Web UI listener
 * Single responsibility: own the local TCP listener for web mode and share
 * static assets plus private chat state with each request. It does not carry
 * tools, confirmations, workspace state, or reasoning state.
 */

use std::sync::Arc;

use color_eyre::eyre::{Result, eyre};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

use super::assets::Assets;
use super::chat::State;
use super::config::WebConfig;
use super::router;

pub(super) async fn serve(config: WebConfig, assets: Assets, chat: State) -> Result<()> {
    let addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&addr)
        .await
        .map_err(|_| eyre!("failed to bind the Web UI listener"))?;
    let assets = Arc::new(assets);

    loop {
        let (stream, _peer) = match listener.accept().await {
            Ok(pair) => pair,
            Err(_) => {
                eprintln!("web: dropped a connection on accept error");
                continue;
            }
        };

        let io = TokioIo::new(stream);
        let assets = assets.clone();
        let chat = chat.clone();
        tokio::spawn(async move {
            let service = service_fn(move |req| router::handle(req, assets.clone(), chat.clone()));
            if let Err(err) = http1::Builder::new().serve_connection(io, service).await
                && !is_client_disconnect(&err)
            {
                eprintln!("web: a connection ended with an error: {err}");
            }
        });
    }
}

/// Routine browser disconnects do not indicate a failed Web UI listener.
fn is_client_disconnect(err: &hyper::Error) -> bool {
    if err.is_incomplete_message() {
        return true;
    }
    let mut source = std::error::Error::source(err);
    while let Some(cause) = source {
        if let Some(io) = cause.downcast_ref::<std::io::Error>() {
            return matches!(
                io.kind(),
                std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::BrokenPipe
                    | std::io::ErrorKind::NotConnected
            );
        }
        source = cause.source();
    }
    false
}
