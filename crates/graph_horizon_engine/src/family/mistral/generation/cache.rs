/*
 * graph_horizon_engine — single-slot homogeneous GPU request-session cache
 * Retains one KV allocation between serialized requests and optionally reuses a
 * caller-keyed token prefix. It owns cache replacement and failure invalidation;
 * the shared generation driver continues to own prefill, decode, and terminal data.
 */

use color_eyre::eyre::Result;

use super::super::RuntimeModel;
use super::super::graph::MistralGraph;
use super::{drive, template};
use crate::api::event::{GenerationStats, Terminal};
use crate::api::request::Request;
use crate::backend::selection;

pub(in crate::family::mistral) struct SessionCache {
    prefix: Option<CachedPrefix>,
    state: selection::CachedState,
}

struct CachedPrefix {
    key: [u8; 16],
    tokens: Vec<u32>,
}

pub(super) fn execute(
    model: &RuntimeModel,
    request: &Request,
    key: Option<[u8; 16]>,
    terminal: &mut Terminal<'_>,
) -> Result<Option<GenerationStats>> {
    let mut slot = model
        .session_cache
        .lock()
        .map_err(|_| color_eyre::eyre::eyre!("session cache unavailable"))?;
    let previous = slot.take();
    let prompt = match template::render(&request.messages, &model.tokenizer, model.context) {
        Ok(prompt) => prompt,
        Err(error) => {
            if let Some(cache) = previous {
                free_cache(&model.backend, cache);
            }
            return Err(error);
        }
    };
    let prefix = key.map_or(0, |key| {
        reusable_prefix(
            previous.as_ref().and_then(|cache| {
                cache
                    .prefix
                    .as_ref()
                    .map(|prefix| (&prefix.key, prefix.tokens.as_slice()))
            }),
            &key,
            &prompt,
        )
    });
    let state = previous.map(|cache| cache.state);
    let session = selection::cached_session::<MistralGraph>(
        &model.backend,
        &model.config,
        model.shape(),
        model.context,
        model.scheme,
        state,
    )?;

    // The retained prefix is read-only: causal attention can observe only
    // positions below `prefix`, while every suffix position is overwritten.
    let outcome = drive(model, request, &prompt, prefix, &session, terminal);
    if matches!(&outcome, Ok(Some(_))) {
        // A non-keyed request overwrites from position zero. Stale bytes beyond
        // its current position are unreachable under causal attention.
        *slot = Some(SessionCache {
            prefix: key.map(|key| CachedPrefix {
                key,
                tokens: prompt,
            }),
            state: session.into_state(),
        });
    }
    outcome
}

pub(in crate::family::mistral) fn free_cache(
    backend: &selection::SelectedBackend,
    cache: SessionCache,
) {
    selection::free_cached_state(backend, cache.state);
}

fn reusable_prefix(previous: Option<(&[u8; 16], &[u32])>, key: &[u8; 16], prompt: &[u32]) -> usize {
    let Some((_, tokens)) = previous.filter(|(previous_key, _)| *previous_key == key) else {
        return 0;
    };
    let common = tokens
        .iter()
        .zip(prompt)
        .take_while(|(old, new)| old == new)
        .count();

    // Prefill at least one token so logits always describe the new prompt tail.
    if common == prompt.len() {
        common.saturating_sub(1)
    } else {
        common
    }
}

#[cfg(test)]
mod tests {
    use super::reusable_prefix;

    #[test]
    fn reuse_requires_the_same_key_and_exact_tokens() {
        let key = [7; 16];
        let other = [8; 16];
        let tokens = [1, 2, 3, 4];

        assert_eq!(reusable_prefix(Some((&key, &tokens)), &other, &tokens), 0);
        assert_eq!(reusable_prefix(Some((&key, &tokens)), &key, &[1, 2, 9]), 2);
    }

    #[test]
    fn identical_prompt_refills_its_tail_for_fresh_logits() {
        let key = [7; 16];
        assert_eq!(
            reusable_prefix(Some((&key, &[1, 2, 3])), &key, &[1, 2, 3]),
            2
        );
    }
}
