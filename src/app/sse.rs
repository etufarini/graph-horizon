/*
 * GH Zero app - shared SSE chat envelope
 * Owns the OpenAI-compatible chat-completion chunk shape shared by the headless
 * server and web surfaces: delta chunks, usage chunk, final stop chunk and the
 * [DONE] sentinel. This module does egress formatting only; request parsing,
 * engine events and web-only extensions stay in their surface modules.
 */

use gh_zero_engine::GenerationStats;
use serde::Serialize;

const MODEL_NAME: &str = "gh-zero";
const CHUNK_ID: &str = "chatcmpl-gh-zero";
const OBJECT: &str = "chat.completion.chunk";

#[derive(Serialize)]
struct Chunk {
    id: &'static str,
    object: &'static str,
    created: u64,
    model: &'static str,
    choices: [ChunkChoice; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    prefill_ms: u64,
    decode_ms: u64,
}

#[derive(Serialize)]
struct ChunkChoice {
    index: u32,
    delta: Delta,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize, Default)]
pub(crate) struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) content: Option<String>,
}

pub(crate) fn delta_line(delta: Delta) -> String {
    chunk_line(delta, None, None, created_now())
}

pub(crate) fn final_line() -> String {
    chunk_line(Delta::default(), Some("stop"), None, created_now())
}

pub(crate) fn usage_line(stats: &GenerationStats) -> String {
    usage_line_at(stats, created_now())
}

pub(crate) fn done_line() -> String {
    "data: [DONE]\n\n".to_string()
}

pub(crate) fn data_line<T: Serialize>(value: &T) -> String {
    let json = serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string());
    format!("data: {json}\n\n")
}

fn created_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn chunk_line(
    delta: Delta,
    finish_reason: Option<&'static str>,
    usage: Option<Usage>,
    created: u64,
) -> String {
    let chunk = Chunk {
        id: CHUNK_ID,
        object: OBJECT,
        created,
        model: MODEL_NAME,
        choices: [ChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
        usage,
    };
    data_line(&chunk)
}

fn usage_line_at(stats: &GenerationStats, created: u64) -> String {
    chunk_line(
        Delta::default(),
        None,
        Some(Usage {
            prompt_tokens: stats.prompt_tokens,
            completion_tokens: stats.completion_tokens,
            prefill_ms: stats.prefill_ms,
            decode_ms: stats.decode_ms,
        }),
        created,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_line_matches_server_chunk_shape() {
        let line = chunk_line(
            Delta {
                content: Some("ciao".to_string()),
            },
            None,
            None,
            7,
        );
        assert_eq!(
            line,
            "data: {\"id\":\"chatcmpl-gh-zero\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"gh-zero\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ciao\"},\"finish_reason\":null}]}\n\n"
        );
    }

    #[test]
    fn final_line_matches_server_stop_chunk_shape() {
        assert_eq!(
            chunk_line(Delta::default(), Some("stop"), None, 7),
            "data: {\"id\":\"chatcmpl-gh-zero\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"gh-zero\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
        );
    }

    #[test]
    fn usage_line_matches_server_usage_chunk_shape() {
        let stats = GenerationStats {
            prompt_tokens: 128,
            completion_tokens: 42,
            prefill_ms: 400,
            decode_ms: 875,
        };
        assert_eq!(
            usage_line_at(&stats, 7),
            "data: {\"id\":\"chatcmpl-gh-zero\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"gh-zero\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":128,\"completion_tokens\":42,\"prefill_ms\":400,\"decode_ms\":875}}\n\n"
        );
    }

    #[test]
    fn done_line_is_exact() {
        assert_eq!(done_line(), "data: [DONE]\n\n");
    }
}
