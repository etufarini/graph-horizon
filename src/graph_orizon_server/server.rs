/*
 * Graph Orizon headless server loop
 * Owns TCP bind and per-connection HTTP serving for `--mode server`. It shares a
 * prepared `ServerState` with handlers and contains no request validation logic.
 */

use color_eyre::eyre::{Result, WrapErr};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

use super::handler;
use super::state::ServerState;

pub(crate) async fn serve(state: ServerState) -> Result<()> {
    let addr = format!("{}:{}", state.config.host, state.config.port);
    let listener = TcpListener::bind(&addr)
        .await
        .wrap_err_with(|| format!("failed to bind the server on {addr}"))?;

    loop {
        let (stream, _peer) = match listener.accept().await {
            Ok(pair) => pair,
            Err(_) => {
                eprintln!("server: dropped a connection on accept error");
                continue;
            }
        };

        let io = TokioIo::new(stream);
        let state = state.clone();
        tokio::spawn(async move {
            let service = service_fn(move |req| handler::handle(req, state.clone()));
            if let Err(err) = http1::Builder::new().serve_connection(io, service).await
                && !is_client_disconnect(&err)
            {
                eprintln!("server: a connection ended with an error: {err}");
            }
        });
    }
}

/// True when the error only means the client hung up: browsers open
/// speculative sockets and close them without a request (IncompleteMessage),
/// reset idle keep-alive connections, or abort a response mid-stream. These
/// are routine and leave the server healthy, so they are not worth logging.
pub(crate) fn is_client_disconnect(err: &hyper::Error) -> bool {
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
