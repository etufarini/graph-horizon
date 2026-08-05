/*
 * graph_horizon_engine — final Ministral request lifecycle
 * Renders one text chat, drives a neutral request session, streams UTF-8 deltas,
 * and normalizes cancellation or failure into at most one terminal event.
 */

mod session;
#[cfg(test)]
pub(crate) mod tests;

use color_eyre::eyre::Result;

use super::RuntimeModel;
use super::decode::TextDecoder;
use super::template;
use crate::api::event::{GenerationStats, Terminal};
use crate::api::request::{EventSink, Request};
use crate::runtime::RuntimeSession;
use crate::sampling::{self, Rng};

pub(crate) fn generate(model: &RuntimeModel, request: Request, sink: &mut dyn EventSink) {
    let mut terminal = Terminal::new(sink);
    match execute(model, &request, &mut terminal) {
        Ok(Some(stats)) => terminal.finish(stats),
        Ok(None) => {}
        Err(_) => terminal.fail(),
    }
}

fn execute(
    model: &RuntimeModel,
    request: &Request,
    terminal: &mut Terminal<'_>,
) -> Result<Option<GenerationStats>> {
    let prompt = template::render(&request.messages, &model.tokenizer, model.context)?;
    let session = session::new(model)?;
    drive(model, request, &prompt, &session, terminal)
}

fn drive<S: RuntimeSession>(
    model: &RuntimeModel,
    request: &Request,
    prompt: &[u32],
    session: &S,
    terminal: &mut Terminal<'_>,
) -> Result<Option<GenerationStats>> {
    let prefill_start = std::time::Instant::now();
    session.prefill(prompt, &mut || {
        (!terminal.cancelled())
            .then_some(())
            .ok_or_else(|| color_eyre::eyre::eyre!("generation cancelled"))
    })?;
    let decode_start = std::time::Instant::now();
    let prefill_ms = decode_start.duration_since(prefill_start).as_millis() as u64;
    let mut decoder = TextDecoder::default();
    let mut recent = prompt.to_vec();
    let mut rng = Rng::new(request.sampling.seed);
    let mut produced = 0;
    let sample_path = sampling::plan(&request.sampling);

    while produced < request.max_tokens && prompt.len() + produced < model.context {
        if terminal.cancelled() {
            return Ok(None);
        }
        let token = match sample_path {
            sampling::SamplePath::Greedy => session.argmax(model.config.vocab_size)?,
            sampling::SamplePath::TopK(k) => {
                let mut candidates = session.topk(model.config.vocab_size, k)?;
                let inv_temperature = 1.0 / request.sampling.temperature;
                for (_, logit) in &mut candidates {
                    *logit *= inv_temperature;
                }
                sampling::sample_from_candidates(candidates, &request.sampling, &mut rng)
            }
            sampling::SamplePath::Fallback => {
                let mut logits = session.logits(model.config.vocab_size)?;
                sampling::sample(&mut logits, &request.sampling, &recent, &mut rng)
            }
        };
        if token == model.tokenizer.eos_id() {
            break;
        }
        if let Some(text) = decoder.push(&model.tokenizer.decode_bytes(&[token]))
            && !terminal.delta(text)
        {
            return Ok(None);
        }
        recent.push(token);
        produced += 1;
        if produced < request.max_tokens && prompt.len() + produced < model.context {
            session.token(token, prompt.len() + produced - 1)?;
        }
    }
    decoder.finish();
    Ok(Some(GenerationStats {
        prompt_tokens: prompt.len(),
        completion_tokens: produced,
        prefill_ms,
        decode_ms: decode_start.elapsed().as_millis() as u64,
    }))
}
