/*
 * graph_horizon_engine — public generation event contract
 * Defines the text-only stream vocabulary and owns its terminal-emission
 * invariant. Operational details never cross this boundary: every generation
 * failure is represented by the same sanitized string.
 */

use super::request::EventSink;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub prefill_ms: u64,
    pub decode_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    TextDelta(String),
    Finished(GenerationStats),
    Error(String),
}

pub(crate) struct Terminal<'a> {
    sink: &'a mut dyn EventSink,
    closed: bool,
}

impl<'a> Terminal<'a> {
    pub(crate) fn new(sink: &'a mut dyn EventSink) -> Self {
        Self {
            sink,
            closed: false,
        }
    }

    pub(crate) fn cancelled(&self) -> bool {
        self.closed || self.sink.cancelled()
    }

    pub(crate) fn delta(&mut self, text: String) -> bool {
        if self.cancelled() || !self.sink.emit(Event::TextDelta(text)) {
            self.closed = true;
        }
        !self.closed
    }

    pub(crate) fn finish(&mut self, stats: GenerationStats) {
        if !self.cancelled() {
            self.sink.emit(Event::Finished(stats));
        }
        self.closed = true;
    }

    pub(crate) fn fail(&mut self) {
        if !self.cancelled() {
            self.sink.emit(Event::Error("generation failed".into()));
        }
        self.closed = true;
    }
}
