/*
 * gh_zero_engine — final Ministral request lifecycle
 * Renders one text chat, selects one immutable sampling path, drives its neutral
 * request-session loop, streams UTF-8 deltas, and normalizes cancellation or
 * failure into at most one terminal event. Backend graph details stay outside.
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
    let produced = match sampling::plan(&request.sampling) {
        sampling::SamplePath::Greedy => greedy(
            model,
            request,
            prompt.len(),
            session,
            terminal,
            &mut decoder,
        )?,
        sampling::SamplePath::TopK(k) => topk(
            model,
            request,
            prompt.len(),
            session,
            terminal,
            &mut decoder,
            k,
        )?,
        sampling::SamplePath::Fallback => {
            fallback(model, request, prompt, session, terminal, &mut decoder)?
        }
    };
    let Some(produced) = produced else {
        return Ok(None);
    };
    decoder.finish();
    Ok(Some(GenerationStats {
        prompt_tokens: prompt.len(),
        completion_tokens: produced,
        prefill_ms,
        decode_ms: decode_start.elapsed().as_millis() as u64,
    }))
}

fn greedy<S: RuntimeSession>(
    model: &RuntimeModel,
    request: &Request,
    prompt_tokens: usize,
    session: &S,
    terminal: &mut Terminal<'_>,
    decoder: &mut TextDecoder,
) -> Result<Option<usize>> {
    if request.max_tokens == 0 || prompt_tokens >= model.context {
        return Ok(Some(0));
    }
    if terminal.cancelled() {
        return Ok(None);
    }
    let mut produced = 0;
    let mut token = session.argmax(model.config.vocab_size)?;
    loop {
        if token == model.tokenizer.eos_id() {
            break;
        }
        if !output(model, terminal, decoder, token) {
            return Ok(None);
        }
        produced += 1;
        if produced == request.max_tokens || prompt_tokens + produced == model.context {
            break;
        }
        if terminal.cancelled() {
            return Ok(None);
        }
        token =
            session.token_argmax(token, prompt_tokens + produced - 1, model.config.vocab_size)?;
        // Fusion removes the old cancellation seam between graph submit and
        // sampling; checking here preserves cancellation before token emission.
        if terminal.cancelled() {
            return Ok(None);
        }
    }
    Ok(Some(produced))
}

#[allow(clippy::too_many_arguments)]
fn topk<S: RuntimeSession>(
    model: &RuntimeModel,
    request: &Request,
    prompt_tokens: usize,
    session: &S,
    terminal: &mut Terminal<'_>,
    decoder: &mut TextDecoder,
    k: usize,
) -> Result<Option<usize>> {
    let mut rng = Rng::new(request.sampling.seed);
    let inv_temperature = 1.0 / request.sampling.temperature;
    let mut produced = 0;
    while produced < request.max_tokens && prompt_tokens + produced < model.context {
        if terminal.cancelled() {
            return Ok(None);
        }
        let mut candidates = session.topk(model.config.vocab_size, k)?;
        for (_, logit) in &mut candidates {
            *logit *= inv_temperature;
        }
        let token = sampling::sample_from_candidates(candidates, &request.sampling, &mut rng);
        if token == model.tokenizer.eos_id() {
            break;
        }
        if !output(model, terminal, decoder, token) {
            return Ok(None);
        }
        produced += 1;
        if produced < request.max_tokens && prompt_tokens + produced < model.context {
            session.token(token, prompt_tokens + produced - 1)?;
        }
    }
    Ok(Some(produced))
}

fn fallback<S: RuntimeSession>(
    model: &RuntimeModel,
    request: &Request,
    prompt: &[u32],
    session: &S,
    terminal: &mut Terminal<'_>,
    decoder: &mut TextDecoder,
) -> Result<Option<usize>> {
    let mut rng = Rng::new(request.sampling.seed);
    let mut recent = prompt.to_vec();
    let mut produced = 0;
    while produced < request.max_tokens && prompt.len() + produced < model.context {
        if terminal.cancelled() {
            return Ok(None);
        }
        let mut logits = session.logits(model.config.vocab_size)?;
        let token = sampling::sample(&mut logits, &request.sampling, &recent, &mut rng);
        if token == model.tokenizer.eos_id() {
            break;
        }
        if !output(model, terminal, decoder, token) {
            return Ok(None);
        }
        recent.push(token);
        produced += 1;
        if produced < request.max_tokens && prompt.len() + produced < model.context {
            session.token(token, prompt.len() + produced - 1)?;
        }
    }
    Ok(Some(produced))
}

fn output(
    model: &RuntimeModel,
    terminal: &mut Terminal<'_>,
    decoder: &mut TextDecoder,
    token: u32,
) -> bool {
    decoder
        .push(&model.tokenizer.decode_bytes(&[token]))
        .is_none_or(|text| terminal.delta(text))
}
