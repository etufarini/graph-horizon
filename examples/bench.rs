/*
 * GH Zero chat benchmark
 * Measures family-neutral public Engine prefill/decode timing for one explicit
 * model, context and KV scheme. It does not inspect graph internals or define
 * performance thresholds.
 */

use std::path::Path;

use color_eyre::eyre::{Result, bail, eyre};
use gh_zero_engine::harness::throughput::{self, BenchConfig};
use gh_zero_engine::{Engine, EngineConfig, KvQuant};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().ok_or_else(|| {
        eyre!("usage: bench <model.gguf> --context N --kv f16|int8 [--prompt TEXT] [--max-tokens N] [--warmup N] [--reps N]")
    })?;
    let mut context = None;
    let mut kv = None;
    let mut prompt = "Ciao".to_string();
    let mut max_tokens = 32;
    let mut warmup = 1;
    let mut reps = 3;
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| eyre!("missing value for {flag}"))?;
        match flag.as_str() {
            "--context" => context = Some(value.parse::<usize>()?),
            "--kv" => kv = KvQuant::parse(&value),
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
    let engine = Engine::new(
        Path::new(&model),
        EngineConfig {
            context_tokens: Some(context),
            kv_quant: kv,
            ..EngineConfig::default()
        },
    )?;
    let report = throughput::run(
        &engine,
        &BenchConfig {
            preset: format!("context={context},kv={}", kv.name()),
            prompt,
            gen_tokens: max_tokens,
            reps,
            warmup,
        },
    )?;
    println!(
        "prompt_tokens={} prompt_tps={:.2}±{} ttft_ms={:.2} decode_tps={}",
        report.n_prompt,
        report.prompt_tps.mean,
        report
            .prompt_tps
            .stddev
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "n/a".into()),
        report.ttft_seconds.mean * 1000.0,
        report
            .tg
            .map(|value| format!("{:.2}", value.mean))
            .unwrap_or_else(|| "n/a".into())
    );
    Ok(())
}
