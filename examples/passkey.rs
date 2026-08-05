/*
 * Graph Orizon passkey example
 * Exercises long text-context recall through the final public chat API for one
 * explicit model/context/KV configuration. It reports correctness only when
 * the generated text contains the supplied passkey.
 */

use std::path::Path;

use color_eyre::eyre::{Result, bail, eyre};
use graph_orizon_engine::{
    Engine, EngineConfig, Event, KvQuant, Message, Request, Role, SamplingParams,
};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .ok_or_else(|| eyre!("usage: passkey <model.gguf> <context> <f16|int8> [passkey]"))?;
    let context = args
        .next()
        .ok_or_else(|| eyre!("missing context"))?
        .parse::<usize>()?;
    let kv = args
        .next()
        .and_then(|value| KvQuant::parse(&value))
        .ok_or_else(|| eyre!("KV must be f16 or int8"))?;
    let passkey = args.next().unwrap_or_else(|| "GHZ-314159".into());
    if context == 0 || args.next().is_some() {
        bail!("invalid passkey arguments");
    }

    let engine = Engine::new(
        Path::new(&model),
        EngineConfig {
            context_tokens: Some(context),
            kv_quant: kv,
            ..EngineConfig::default()
        },
    )?;
    let filler = "The archive contains ordinary text. ".repeat((context / 12).max(8));
    let prompt = format!("Remember this passkey: {passkey}.\n{filler}\nReturn only the passkey.");
    let mut response = String::new();
    let mut failure = None;
    engine.generate(
        Request {
            messages: vec![Message {
                role: Role::User,
                content: prompt,
            }],
            sampling: SamplingParams::greedy(),
            max_tokens: 32,
        },
        &mut |event| {
            match event {
                Event::TextDelta(text) => response.push_str(&text),
                Event::Error(message) => failure = Some(message),
                Event::Finished(_) => {}
            }
            true
        },
    );
    if let Some(message) = failure {
        bail!("{message}");
    }
    if !response.contains(&passkey) {
        bail!("passkey was not recovered");
    }
    println!("pass: recovered {passkey}");
    Ok(())
}
