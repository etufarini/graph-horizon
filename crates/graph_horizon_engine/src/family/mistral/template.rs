/*
 * graph_horizon_engine — Ministral fixed chat renderer
 * Validates the text-only chat sequence and inserts only structural Tekken
 * control token ids. User/system/assistant content is untrusted ordinary text:
 * it is encoded through the tokenizer and never interpreted as control syntax
 * or as the embedded `tokenizer.chat_template` Jinja payload. Only the two
 * markers in the release-owned implicit Reasoning block use their existing
 * GGUF IDs, and only when no System message is explicit.
*/

use color_eyre::eyre::{Result, bail};

use crate::api::message::{Message, Role};

use super::tokenizer::TekkenTokenizer;
use super::version::REASONING_SYSTEM_PROMPT;

pub fn render(
    messages: &[Message],
    tokenizer: &TekkenTokenizer,
    context: usize,
) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    out.push(tokenizer.bos_id());
    let explicit_system = messages
        .first()
        .filter(|message| message.role == Role::System);
    let implicit_system = explicit_system.is_none() && tokenizer.uses_reasoning_profile();
    let system = explicit_system.map_or_else(
        || implicit_system.then_some(REASONING_SYSTEM_PROMPT),
        |message| Some(message.content.as_str()),
    );
    let mut i = usize::from(explicit_system.is_some());
    if let Some(content) = system {
        out.push(tokenizer.system_open_id());
        if implicit_system {
            out.extend(tokenizer.encode_reasoning_system(content));
        } else {
            out.extend(tokenizer.encode(content));
        }
        out.push(tokenizer.system_close_id());
    }
    if i >= messages.len() {
        bail!("E10 invalid chat sequence");
    }
    let mut expect_user = true;
    while i < messages.len() {
        let m = &messages[i];
        match (expect_user, m.role) {
            (true, Role::User) => {
                out.push(tokenizer.inst_open_id());
                out.extend(tokenizer.encode(m.content.as_str()));
                out.push(tokenizer.inst_close_id());
                expect_user = false;
            }
            (false, Role::Assistant) if i + 1 < messages.len() && !m.content.is_empty() => {
                out.extend(tokenizer.encode(m.content.as_str()));
                out.push(tokenizer.eos_id());
                expect_user = true;
            }
            _ => bail!("E10 invalid chat sequence"),
        }
        i += 1;
    }
    if !expect_user {
        if out.len() > context {
            bail!(
                "E11 prompt has {} tokens but context is {context}",
                out.len()
            );
        }
        Ok(out)
    } else {
        bail!("E10 invalid chat sequence")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::mistral::tokenizer::tests::{mini_reasoning_tokenizer, mini_tokenizer};

    fn msg(role: Role, content: &str) -> Message {
        Message {
            role,
            content: content.into(),
        }
    }

    #[test]
    fn instruct_template_prompt_ids_are_unchanged() {
        let tok = mini_tokenizer();
        let out = render(
            &[
                msg(Role::System, "s"),
                msg(Role::User, "h"),
                msg(Role::Assistant, "a"),
                msg(Role::User, "i"),
            ],
            &tok,
            32,
        )
        .unwrap();
        assert_eq!(
            out,
            vec![
                tok.bos_id(),
                tok.system_open_id(),
                b's' as u32,
                tok.system_close_id(),
                tok.inst_open_id(),
                b'h' as u32,
                tok.inst_close_id(),
                b'a' as u32,
                tok.eos_id(),
                tok.inst_open_id(),
                b'i' as u32,
                tok.inst_close_id()
            ]
        );

        let no_system = render(&[msg(Role::User, "h")], &tok, 8).unwrap();
        assert_eq!(
            no_system,
            vec![
                tok.bos_id(),
                tok.inst_open_id(),
                b'h' as u32,
                tok.inst_close_id()
            ]
        );
    }

    #[test]
    fn reasoning_template_injects_default_system() {
        let tok = mini_reasoning_tokenizer();
        let out = render(&[msg(Role::User, "h")], &tok, 1024).unwrap();
        let mut expected = vec![tok.bos_id(), tok.system_open_id()];
        expected.extend(tok.encode_reasoning_system(REASONING_SYSTEM_PROMPT));
        expected.extend([
            tok.system_close_id(),
            tok.inst_open_id(),
            b'h' as u32,
            tok.inst_close_id(),
        ]);
        assert_eq!(out, expected);
        assert!(out.contains(&262));
        assert!(out.contains(&263));
    }

    #[test]
    fn reasoning_template_explicit_system_replaces_default() {
        let tok = mini_reasoning_tokenizer();
        for content in ["custom", "", "[THINK]x[/THINK]"] {
            let out = render(
                &[msg(Role::System, content), msg(Role::User, "h")],
                &tok,
                32,
            )
            .unwrap();
            let mut expected = vec![tok.bos_id(), tok.system_open_id()];
            expected.extend(tok.encode(content));
            expected.extend([
                tok.system_close_id(),
                tok.inst_open_id(),
                b'h' as u32,
                tok.inst_close_id(),
            ]);
            assert_eq!(out, expected);
            assert_ne!(out, tok.encode(REASONING_SYSTEM_PROMPT));
            assert!(!out.contains(&262));
            assert!(!out.contains(&263));
        }
    }

    #[test]
    fn reasoning_template_context_counts_implicit_system() {
        let tok = mini_reasoning_tokenizer();
        let actual = 1 + 1 + tok.encode_reasoning_system(REASONING_SYSTEM_PROMPT).len() + 1 + 3;
        let error = render(&[msg(Role::User, "h")], &tok, actual - 1)
            .unwrap_err()
            .to_string();
        assert_eq!(
            error,
            format!(
                "E11 prompt has {actual} tokens but context is {}",
                actual - 1
            )
        );
    }

    #[test]
    fn error_matrix_e10_e11_rejects_sequence_and_context_overflow() {
        let tok = mini_tokenizer();
        let err = render(&[msg(Role::Assistant, "x")], &tok, 32)
            .unwrap_err()
            .to_string();
        assert!(err.contains("E10 invalid chat sequence"));
        let err = render(&[msg(Role::User, "hi")], &tok, 3)
            .unwrap_err()
            .to_string();
        assert!(err.contains("E11 prompt has 5 tokens but context is 3"));
    }
}
