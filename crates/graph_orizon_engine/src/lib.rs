/*
 * graph_orizon_engine — final library facade
 * Exposes the text-only engine contract, the selectable KV scheme, and the
 * read-only GGUF/tokenizer inspection types consumed by supported examples.
 * The engine has no public weight-profile selector; sampling, cache layout,
 * graph wiring, and backend ownership remain internal implementation details.
 */

mod api;
mod backend;
mod family;
mod gguf;
pub mod harness;
mod kv_cache;
mod rng;
mod runtime;
mod sampling;

pub use api::engine::{BackendMemory, Engine, EngineConfig, PlacementReport};
pub use api::event::{Event, GenerationStats};
pub use api::message::{Message, Role};
pub use api::request::{EventSink, Request, SamplingParams};
pub use family::mistral::template::render as render_chat_prompt;
pub use family::mistral::{MistralConfig, TekkenTokenizer};
pub use gguf::loader::{GgufFile, GgufValue};
pub use gguf::tensor_index::{GgmlType, TensorInfo};
pub use kv_cache::scheme::KvQuant;
