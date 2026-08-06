/*
 * Graph Horizon single-tuple chat benchmark
 * Owns CLI validation and rendering for one family-neutral public Engine run.
 * It owns no artifact authentication, comparison, performance verdict,
 * fallback, or phase-internal timing.
 */

use std::{collections::HashSet, path::Path};

use graph_horizon_engine::harness::throughput::{self, BenchConfig};
use graph_horizon_engine::{Engine, EngineConfig, KvQuant};

enum BenchFailure {
    Usage(&'static str),
    Execution(&'static str),
}

const INVALID_ARGUMENTS: BenchFailure = BenchFailure::Usage("bench: invalid arguments");
const INVALID_MEASUREMENT: BenchFailure = BenchFailure::Execution("bench: invalid measurement");

fn main() {
    if let Err(failure) = run() {
        let (message, code) = match failure {
            BenchFailure::Usage(message) => (message, 2),
            BenchFailure::Execution(message) => (message, 1),
        };
        eprintln!("{message}");
        std::process::exit(code);
    }
}

fn run() -> Result<(), BenchFailure> {
    let args = std::env::args_os()
        .skip(1)
        .map(|arg| arg.into_string().map_err(|_| INVALID_ARGUMENTS))
        .collect::<Result<Vec<_>, _>>()?;
    let (model, options) = args.split_first().ok_or(INVALID_ARGUMENTS)?;
    if model.starts_with("--") {
        return Err(INVALID_ARGUMENTS);
    }
    let mut context = None;
    let mut kv = None;
    let mut prompt = "Ciao".to_string();
    let mut max_tokens = 32;
    let mut warmup = 1;
    let mut reps = 3;
    let mut weights_percent = None;
    let mut seen = HashSet::new();
    let usage = BenchFailure::Usage;
    let number = |value: &str, minimum, maximum, message| match value.parse::<usize>() {
        Ok(value) if (minimum..=maximum).contains(&value) => Ok(value),
        _ => Err(usage(message)),
    };
    for pair in options.chunks(2) {
        let flag = &pair[0];
        let message = match flag.as_str() {
            "--context" => "bench: invalid --context",
            "--kv" => "bench: invalid --kv",
            "--weights-percent" => "bench: invalid --weights-percent",
            "--prompt" => "bench: invalid --prompt",
            "--max-tokens" => "bench: invalid --max-tokens",
            "--warmup" => "bench: invalid --warmup",
            "--reps" => "bench: invalid --reps",
            _ => return Err(INVALID_ARGUMENTS),
        };
        // Duplicate rejection precedes parsing, so a later value never
        // overwrites a validated tuple field.
        if !seen.insert(flag) {
            return Err(usage(message));
        }
        let value = pair.get(1).ok_or(usage(message))?;
        match flag.as_str() {
            "--context" => context = Some(number(value, 1, usize::MAX, message)?),
            "--kv" => kv = Some(KvQuant::parse(value).ok_or(usage(message))?),
            "--weights-percent" => weights_percent = Some(number(value, 0, 100, message)? as u8),
            "--prompt" if !value.is_empty() => prompt = value.clone(),
            "--prompt" => return Err(usage(message)),
            "--max-tokens" => max_tokens = number(value, 2, usize::MAX, message)?,
            "--warmup" => warmup = number(value, 0, usize::MAX, message)?,
            "--reps" => reps = number(value, 1, usize::MAX, message)?,
            _ => unreachable!(),
        }
    }
    let context = context.ok_or(usage("bench: invalid --context"))?;
    let kv = kv.ok_or(usage("bench: invalid --kv"))?;
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
    if report.decoded_tokens < 2 {
        return Err(INVALID_MEASUREMENT);
    }
    let metric = |mean: f64, stddev: Option<f64>, scale: f64| {
        let shown_mean = mean * scale;
        let shown_stddev = stddev.map(|value| value * scale);
        let cv = stddev.map(|value| value / mean);
        if !shown_mean.is_finite()
            || shown_mean <= 0.0
            || shown_stddev.is_some_and(|value| !value.is_finite() || value < 0.0)
            || cv.is_some_and(|value| !value.is_finite())
        {
            return Err(INVALID_MEASUREMENT);
        }
        Ok([
            format!("{shown_mean:.2}"),
            shown_stddev.map_or_else(|| "n/a".into(), |value| format!("{value:.2}")),
            cv.map_or_else(|| "n/a".into(), |value| format!("{value:.4}")),
        ])
    };
    let [p_mean, p_sd, p_cv] = metric(report.prompt_tps.mean, report.prompt_tps.stddev, 1.0)?;
    let [t_mean, t_sd, t_cv] =
        metric(report.ttft_seconds.mean, report.ttft_seconds.stddev, 1000.0)?;
    let decode = report.tg.ok_or(INVALID_MEASUREMENT)?;
    let [d_mean, d_sd, d_cv] = metric(decode.mean, decode.stddev, 1.0)?;
    let prompt_tokens = report.n_prompt;
    let decoded_tokens = report.decoded_tokens;
    println!(
        "prompt_tokens={prompt_tokens} decoded_tokens={decoded_tokens} prompt_tps_mean={p_mean} prompt_tps_stddev={p_sd} prompt_tps_cv={p_cv} ttft_ms_mean={t_mean} ttft_ms_stddev={t_sd} ttft_cv={t_cv} decode_tps_mean={d_mean} decode_tps_stddev={d_sd} decode_tps_cv={d_cv}"
    );
    Ok(())
}
