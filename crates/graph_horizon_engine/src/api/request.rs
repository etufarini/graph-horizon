/*
 * graph_horizon_engine — public chat request contract
 * Carries one immutable text conversation, its sampling policy, and the sole
 * generation cap. Cancellation is expressed only by the event sink returning
 * false; the KV scheme belongs to EngineConfig because placement happens once
 * before requests can be submitted.
 */

use super::event::Event;
use super::message::Message;

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub seed: u64,
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            seed: 0,
        }
    }
}

pub struct Request {
    pub messages: Vec<Message>,
    pub sampling: SamplingParams,
    pub max_tokens: usize,
}

pub trait EventSink {
    // Checked before each prefill batch and decode step so cancellation does not
    // require a new output token to become observable.
    fn cancelled(&self) -> bool;
    fn emit(&mut self, event: Event) -> bool;
}

impl<F: FnMut(Event) -> bool> EventSink for F {
    fn cancelled(&self) -> bool {
        false
    }

    fn emit(&mut self, event: Event) -> bool {
        self(event)
    }
}
