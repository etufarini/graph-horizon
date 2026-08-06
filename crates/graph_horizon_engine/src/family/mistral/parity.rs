/*
 * graph_horizon_engine — selected-runtime parity protocol
 * Parses fixed oracle vectors, renders the neutral Ministral prompt, and checks
 * sixteen teacher-forced full-logit steps through the statically selected
 * runtime. It owns request-local KV and test crossing evidence, but no backend
 * choice, artifact lookup, oracle process, retry, or fallback.
 */

use color_eyre::eyre::{Result, bail, eyre};

use super::graph::MistralGraph;
use super::{RuntimeModel, template};
use crate::api::message::{Message, Role};
#[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
use crate::backend::hybrid::HybridMode;
use crate::backend::selection;
use crate::runtime::contract::RuntimeSession;

pub const USER_CONTENT: &str = "Quanto fa 17 × 19?";
pub const CONTEXT: usize = 4096;
pub const TOKEN_COUNT: usize = 16;

pub struct ParityReport {
    pub prompt_ids: Vec<u32>,
    pub local_ids: Vec<u32>,
    pub top_two: Vec<[u32; 2]>,
    pub crossings: usize,
}

pub(crate) fn validate(
    model: &RuntimeModel,
    prompt_ids: &str,
    completion_ids: &str,
) -> Result<ParityReport> {
    if model.context != CONTEXT {
        bail!("parity context must be 4096");
    }
    let expected_prompt = parse("GRAPH_HORIZON_REFERENCE_PROMPT_IDS", prompt_ids, None)?;
    let completion = parse(
        "GRAPH_HORIZON_REFERENCE_COMPLETION_IDS",
        completion_ids,
        Some(TOKEN_COUNT),
    )?;
    let prompt = template::render(&conversation(), &model.tokenizer, model.context)?;
    if prompt != expected_prompt {
        bail!("oracle prompt IDs do not match the local prompt");
    }
    validate_vocab(&prompt, model.config.vocab_size, "prompt")?;
    validate_vocab(&completion, model.config.vocab_size, "completion")?;

    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    crate::backend::hybrid::crossing::reset_count();
    let session = selection::session::<MistralGraph>(
        &model.backend,
        &model.config,
        model.shape(),
        model.context,
        model.scheme,
    )?;
    session.prefill(&prompt, &mut || Ok(()))?;
    let mut local_ids = Vec::with_capacity(TOKEN_COUNT);
    let mut top_two = Vec::with_capacity(TOKEN_COUNT);
    for (step, &oracle) in completion.iter().enumerate() {
        let ranked = ranked_logits(session.logits(model.config.vocab_size)?)?;
        local_ids.push(ranked[0]);
        top_two.push([ranked[0], ranked[1]]);
        if !top_two[step].contains(&oracle) {
            bail!("oracle completion ID is absent from local top two at step {step}");
        }
        if step + 1 < TOKEN_COUNT {
            session.token(oracle, prompt.len() + step)?;
        }
    }

    let crossings = crossing_count();
    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    let expected_crossings = match selection::placement(&model.backend).map(|plan| plan.mode) {
        Some(HybridMode::Mixed) => {
            prompt.len().div_ceil(model.shape().mixed_prefill_rows) + TOKEN_COUNT - 1
        }
        _ => 0,
    };
    #[cfg(not(any(feature = "vulkan-hybrid", feature = "metal-hybrid")))]
    let expected_crossings = 0;
    if crossings != expected_crossings {
        bail!("parity crossing count mismatch");
    }
    Ok(ParityReport {
        prompt_ids: prompt,
        local_ids,
        top_two,
        crossings,
    })
}

fn conversation() -> [Message; 2] {
    [
        Message {
            role: Role::System,
            content: String::new(),
        },
        Message {
            role: Role::User,
            content: USER_CONTENT.into(),
        },
    ]
}

fn ranked_logits(logits: Vec<f32>) -> Result<Vec<u32>> {
    if logits.len() < 2 || logits.iter().any(|value| !value.is_finite()) {
        bail!("parity logits must contain at least two finite values");
    }
    let mut indices = (0..logits.len()).collect::<Vec<_>>();
    // The stable protocol resolves a score tie by the lower token ID.
    indices.sort_unstable_by(|&left, &right| {
        logits[right]
            .total_cmp(&logits[left])
            .then_with(|| left.cmp(&right))
    });
    indices
        .into_iter()
        .map(|index| u32::try_from(index).map_err(|_| eyre!("parity token ID overflow")))
        .collect()
}

fn parse(name: &str, value: &str, expected_len: Option<usize>) -> Result<Vec<u32>> {
    if value.is_empty() {
        bail!("{name} must not be empty");
    }
    let ids = value
        .split(',')
        .map(|part| {
            if part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()) {
                bail!("{name} contains an invalid ID");
            }
            part.parse::<u32>()
                .map_err(|_| eyre!("{name} contains an invalid ID"))
        })
        .collect::<Result<Vec<_>>>()?;
    if let Some(expected) = expected_len
        && ids.len() != expected
    {
        bail!("{name} must contain exactly {expected} IDs");
    }
    Ok(ids)
}

fn validate_vocab(ids: &[u32], vocab: usize, label: &str) -> Result<()> {
    if ids.iter().any(|&id| id as usize >= vocab) {
        bail!("oracle {label} ID is outside the model vocabulary");
    }
    Ok(())
}

#[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
fn crossing_count() -> usize {
    crate::backend::hybrid::crossing::count()
}

#[cfg(not(any(feature = "vulkan-hybrid", feature = "metal-hybrid")))]
fn crossing_count() -> usize {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_requires_sixteen_strict_unsigned_completion_ids() {
        let sixteen = (0..TOKEN_COUNT)
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(
            parse("IDS", &sixteen, Some(TOKEN_COUNT)).unwrap().len(),
            TOKEN_COUNT
        );
        for value in ["", "1,", "1,,2", "-1", "+1", "1, 2", "4294967296"] {
            assert!(parse("IDS", value, None).is_err(), "{value:?}");
        }
        assert!(parse("IDS", "1,2", Some(TOKEN_COUNT)).is_err());
    }

    #[test]
    fn finite_ranking_breaks_ties_by_lower_token_id() {
        assert_eq!(ranked_logits(vec![1.0, 2.0, 2.0]).unwrap(), [1, 2, 0]);
        assert!(ranked_logits(vec![1.0, f32::NAN]).is_err());
        assert!(ranked_logits(vec![f32::INFINITY, 1.0]).is_err());
    }
}
