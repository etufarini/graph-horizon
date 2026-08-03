/*
 * GH Zero chat benchmark
 * Measures family-neutral public Engine prefill/decode timing for one explicit
 * model, context, KV, and optional hybrid placement. It rejects incomplete
 * measurements and defines no performance threshold or fallback.
 */

use std::path::Path;

use color_eyre::eyre::{Result, bail, eyre};
use gh_zero_engine::harness::throughput::{self, BenchConfig};
use gh_zero_engine::{Engine, EngineConfig, KvQuant};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().ok_or_else(|| {
        eyre!("usage: bench <model.gguf> --context N --kv f16|int8 [--weights-percent 0..100] [--prompt TEXT] [--max-tokens N] [--warmup N] [--reps N]")
    })?;
    let mut context = None;
    let mut kv = None;
    let mut prompt = "Ciao".to_string();
    let mut max_tokens = 32;
    let mut warmup = 1;
    let mut reps = 3;
    let mut weights_percent = None;
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| eyre!("missing value for {flag}"))?;
        match flag.as_str() {
            "--context" => context = Some(value.parse::<usize>()?),
            "--kv" => kv = KvQuant::parse(&value),
            "--weights-percent" => weights_percent = Some(value.parse::<u8>()?),
            "--prompt" => prompt = value,
            "--max-tokens" => max_tokens = value.parse()?,
            "--warmup" => warmup = value.parse()?,
            "--reps" => reps = value.parse()?,
            _ => bail!("unknown option: {flag}"),
        }
    }
    let context = context
        .filter(|value| *value > 0)
        .ok_or_else(|| eyre!("--context must be >= 1"))?;
    let kv = kv.ok_or_else(|| eyre!("--kv must be f16 or int8"))?;
    if weights_percent.is_some_and(|value| value > 100) {
        bail!("--weights-percent must be in 0..100");
    }
    if max_tokens < 2 || reps == 0 || prompt.is_empty() {
        bail!("prompt must be nonempty, max tokens >= 2, and reps >= 1");
    }
    let engine = Engine::new(
        Path::new(&model),
        EngineConfig {
            context_tokens: Some(context),
            vram_weights_percent: weights_percent,
            kv_quant: kv,
            ..EngineConfig::default()
        },
    )?;
    let report = throughput::run(
        &engine,
        &BenchConfig {
            preset: format!(
                "context={context},kv={},weights_percent={}",
                kv.name(),
                weights_percent.map_or_else(|| "none".into(), |value| value.to_string())
            ),
            prompt,
            gen_tokens: max_tokens,
            reps,
            warmup,
        },
    )?;
    let ttft_ms = report.ttft_seconds.mean * 1000.0;
    let prompt_tps = report.prompt_tps.mean;
    let decode_tps = report
        .tg
        .ok_or_else(|| eyre!("benchmark produced fewer than two decoded tokens"))?
        .mean;
    if !ttft_ms.is_finite() || !prompt_tps.is_finite() || !decode_tps.is_finite() {
        bail!("benchmark produced a non-finite metric");
    }
    println!(
        "prompt_tokens={} decoded_tokens={} prompt_tps={:.2} ttft_ms={:.2} decode_tps={:.2}",
        report.n_prompt, report.decoded_tokens, prompt_tps, ttft_ms, decode_tps
    );
    Ok(())
}
