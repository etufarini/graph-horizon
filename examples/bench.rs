/*
 * GH Zero single-tuple chat benchmark
 * Owns CLI validation and rendering for one family-neutral public Engine run.
 * It owns no artifact authentication, comparison, performance verdict,
 * fallback, or phase-internal timing.
 */

use std::{path::Path, process::ExitCode};

use gh_zero_engine::harness::throughput::{self, BenchConfig};
use gh_zero_engine::{Engine, EngineConfig, KvQuant};

enum BenchFailure {
    Usage(&'static str),
    Execution(&'static str),
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(failure) => {
            let (message, code) = match failure {
                BenchFailure::Usage(message) => (message, 2),
                BenchFailure::Execution(message) => (message, 1),
            };
            eprintln!("{message}");
            ExitCode::from(code)
        }
    }
}

fn run() -> Result<(), BenchFailure> {
    let args = std::env::args_os()
        .skip(1)
        .map(|arg| {
            arg.into_string()
                .map_err(|_| BenchFailure::Usage("bench: invalid arguments"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (model, options) = args
        .split_first()
        .ok_or(BenchFailure::Usage("bench: invalid arguments"))?;
    if model.starts_with("--") {
        return Err(BenchFailure::Usage("bench: invalid arguments"));
    }
    let mut context = None;
    let mut kv = None;
    let mut prompt = "Ciao".to_string();
    let mut max_tokens = 32;
    let mut warmup = 1;
    let mut reps = 3;
    let mut weights_percent = None;
    let mut seen = 0_u8;
    let number = |value: &str, minimum, message| {
        value
            .parse::<usize>()
            .ok()
            .filter(|value| *value >= minimum)
            .ok_or(BenchFailure::Usage(message))
    };
    for pair in options.chunks(2) {
        let flag = &pair[0];
        // Each recognized option owns one bit, so duplicates fail before a
        // later value can overwrite the validated tuple.
        let (message, bit) = match flag.as_str() {
            "--context" => ("bench: invalid --context", 1),
            "--kv" => ("bench: invalid --kv", 2),
            "--weights-percent" => ("bench: invalid --weights-percent", 4),
            "--prompt" => ("bench: invalid --prompt", 8),
            "--max-tokens" => ("bench: invalid --max-tokens", 16),
            "--warmup" => ("bench: invalid --warmup", 32),
            "--reps" => ("bench: invalid --reps", 64),
            _ => return Err(BenchFailure::Usage("bench: invalid arguments")),
        };
        if seen & bit != 0 {
            return Err(BenchFailure::Usage(message));
        }
        seen |= bit;
        let value = pair.get(1).ok_or(BenchFailure::Usage(message))?;
        match flag.as_str() {
            "--context" => context = Some(number(value, 1, message)?),
            "--kv" => kv = Some(KvQuant::parse(&value).ok_or(BenchFailure::Usage(message))?),
            "--weights-percent" => {
                weights_percent = value.parse::<u8>().ok().filter(|value| *value <= 100);
                if weights_percent.is_none() {
                    return Err(BenchFailure::Usage(message));
                }
            }
            "--prompt" if !value.is_empty() => prompt = value.clone(),
            "--prompt" => return Err(BenchFailure::Usage(message)),
            "--max-tokens" => max_tokens = number(value, 2, message)?,
            "--warmup" => warmup = number(value, 0, message)?,
            "--reps" => reps = number(value, 1, message)?,
            _ => unreachable!(),
        }
    }
    let context = context.ok_or(BenchFailure::Usage("bench: invalid --context"))?;
    let kv = kv.ok_or(BenchFailure::Usage("bench: invalid --kv"))?;
    let engine = Engine::new(
        Path::new(model),
        EngineConfig {
            context_tokens: Some(context),
            vram_weights_percent: weights_percent,
            kv_quant: kv,
            ..EngineConfig::default()
        },
    )
    .map_err(|_| BenchFailure::Execution("bench: model initialization failed"))?;
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
    )
    .map_err(|_| BenchFailure::Execution("bench: measurement failed"))?;
    let ttft_ms = report.ttft_seconds.mean * 1000.0;
    let prompt_tps = report.prompt_tps.mean;
    let decode_tps = report
        .tg
        .ok_or(BenchFailure::Execution("bench: invalid measurement"))?
        .mean;
    if !ttft_ms.is_finite() || !prompt_tps.is_finite() || !decode_tps.is_finite() {
        return Err(BenchFailure::Execution("bench: invalid measurement"));
    }
    println!(
        "prompt_tokens={} decoded_tokens={} prompt_tps={:.2} ttft_ms={:.2} decode_tps={:.2}",
        report.n_prompt, report.decoded_tokens, prompt_tps, ttft_ms, decode_tps
    );
    Ok(())
}
