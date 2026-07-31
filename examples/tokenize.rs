/*
 * GH Zero tokenize example
 * Opens one GGUF read-only, renders one fixed Ministral chat prompt, and prints
 * token IDs with decoded byte pieces. It allocates no inference backend and
 * treats prompt text as ordinary content.
 */

use std::path::Path;

use color_eyre::eyre::{Result, eyre};
use gh_zero_engine::{GgufFile, Message, MistralConfig, Role, TekkenTokenizer, render_chat_prompt};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .ok_or_else(|| eyre!("usage: tokenize <model.gguf> <prompt> [system]"))?;
    let prompt = args
        .next()
        .ok_or_else(|| eyre!("usage: tokenize <model.gguf> <prompt> [system]"))?;
    let system = args.next();
    if args.next().is_some() {
        return Err(eyre!("usage: tokenize <model.gguf> <prompt> [system]"));
    }

    let file = GgufFile::open(Path::new(&model)).map_err(|_| eyre!("invalid GGUF file"))?;
    let tokenizer = TekkenTokenizer::from_metadata(file.metadata())?;
    let config = MistralConfig::from_metadata(
        file.metadata(),
        tokenizer.vocab_size(),
        tokenizer.bos_id(),
        tokenizer.eos_id(),
    )?;
    let mut messages = Vec::new();
    if let Some(content) = system {
        messages.push(Message {
            role: Role::System,
            content,
        });
    }
    messages.push(Message {
        role: Role::User,
        content: prompt,
    });
    let ids = render_chat_prompt(&messages, &tokenizer, config.context_length)?;
    println!("prompt_ids: {ids:?}");
    for id in ids {
        let bytes = tokenizer.decode_bytes(&[id]);
        println!("{id}: {:?}", String::from_utf8_lossy(&bytes));
    }
    Ok(())
}
