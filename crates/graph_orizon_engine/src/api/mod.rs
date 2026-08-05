/*
 * graph_orizon_engine — final public chat boundary
 * Exposes only the persistent engine, plain text messages, generation requests,
 * and the three-event stream. Concrete families and backends remain private.
 */

pub(crate) mod engine;
pub(crate) mod event;
pub(crate) mod message;
pub(crate) mod request;
