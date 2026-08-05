/*
 * Graph Orizon web server loop
 * Single responsibility: own the local TCP listener for web mode and share
 * static assets plus headless chat state with each request. It does not carry
 * tools, confirmations, workspace state, or reasoning state.
 */

use std::sync::Arc;

use color_eyre::eyre::{Result, eyre};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

use crate::graph_orizon_server::ServerState;
use crate::graph_orizon_server::server::is_client_disconnect;

use super::assets::Assets;
use super::config::WebConfig;
use super::router;

pub(super) async fn serve(config: WebConfig, assets: Assets, chat: ServerState) -> Result<()> {
    let addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&addr)
        .await
        .map_err(|_| eyre!("failed to bind the web server"))?;
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
