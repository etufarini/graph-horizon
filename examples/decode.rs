/*
 * Graph Horizon steady-state decode benchmark
 * Primes one exact caller-keyed prompt prefix, then measures repeated decode
 * requests at the same KV depth. It owns only argument validation, sampling,
 * and text reporting; engine routing and timing remain production behavior.
 */

use std::path::Path;
use std::time::Instant;

use color_eyre::eyre::{Result, bail, eyre};
use graph_horizon_engine::{
    Engine, EngineConfig, Event, GenerationStats, KvQuant, Message, Request, Role, SamplingParams,
};

const CACHE_KEY: [u8; 16] = [0xad; 16];

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().ok_or_else(usage)?;
    let depth = parse(&args.next().ok_or_else(usage)?, "depth")?;
    let context = parse(&args.next().ok_or_else(usage)?, "context")?;
    let kv = args
        .next()
        .and_then(|value| KvQuant::parse(&value))
        .ok_or_else(usage)?;
    let tokens = parse(&args.next().ok_or_else(usage)?, "tokens")?;
    let reps = parse(&args.next().ok_or_else(usage)?, "reps")?;
    if depth < 4 || depth + tokens > context || tokens < 2 || reps == 0 || args.next().is_some() {
        bail!(usage());
    }

    let engine = Engine::new(
        Path::new(&model),
        EngineConfig {
            context_tokens: Some(context),
            kv_quant: kv,
            ..EngineConfig::default()
        },
    )?;
    let memory = engine.memory();
    println!(
        "depth={depth} context={context} kv={} weight_bytes={} kv_bytes={}",
        kv.name(),
        memory.weights,
        memory.kv
    );

    // Ministral's no-system chat template contributes exactly BOS, INST, and
    // /INST. Repeated ` a` is one token, making the requested depth explicit.
    let prompt = " a".repeat(depth - 3);
    let (prime, prime_wall) = generate(&engine, &prompt, tokens)?;
    if prime.prompt_tokens != depth {
        bail!(
            "decode: requested depth {depth}, tokenizer produced {}",
            prime.prompt_tokens
        );
    }
    println!(
        "prime_prompt_tokens={} prime_prefill_tokens={} prime_prefill_ms={} prime_decode_ms={} prime_wall_ms={:.3}",
        prime.prompt_tokens, prime.prefill_tokens, prime.prefill_ms, prime.decode_ms, prime_wall
    );

    let mut token_rates = Vec::with_capacity(reps);
    let mut token_ms = Vec::with_capacity(reps);
    let mut wall_ms = Vec::with_capacity(reps);
    let mut attempts = 0;
    while token_rates.len() < reps {
        attempts += 1;
        if attempts > reps.saturating_mul(4) {
            bail!("decode: too many early terminal samples");
        }
        let (stats, wall) = generate(&engine, &prompt, tokens)?;
        if stats.prompt_tokens != depth || stats.prefill_tokens != 1 {
            bail!("decode: keyed prefix was not reused at the requested depth");
        }
        if stats.completion_tokens != tokens {
            println!(
                "attempt={attempts} skipped_completion_tokens={}",
                stats.completion_tokens
            );
            continue;
        }
        if stats.decode_ms == 0 {
            bail!("decode: zero decode duration");
        }
        let rate = stats.completion_tokens as f64 * 1000.0 / stats.decode_ms as f64;
        let per_token = stats.decode_ms as f64 / stats.completion_tokens as f64;
        let sample = token_rates.len();
        println!(
            "sample={sample} prompt_tokens={} prefill_tokens={} completion_tokens={} prefill_ms={} decode_ms={} ms_per_token={per_token:.4} decode_tps={rate:.4} wall_ms={wall:.3}",
            stats.prompt_tokens,
            stats.prefill_tokens,
            stats.completion_tokens,
            stats.prefill_ms,
            stats.decode_ms,
        );
        token_rates.push(rate);
        token_ms.push(per_token);
        wall_ms.push(wall);
    }

    let rates = summarize(&token_rates);
    let latency = summarize(&token_ms);
    let walls = summarize(&wall_ms);
    println!(
        "depth={depth} decode_tps_mean={:.4} decode_tps_median={:.4} decode_tps_stddev={:.4} decode_tps_cv={:.6} ms_per_token_mean={:.4} ms_per_token_median={:.4} ms_per_token_stddev={:.4} ms_per_token_cv={:.6} wall_ms_mean={:.3}",
        rates.mean,
        rates.median,
        rates.stddev,
        rates.stddev / rates.mean,
        latency.mean,
        latency.median,
        latency.stddev,
        latency.stddev / latency.mean,
        walls.mean,
    );
    Ok(())
}

fn usage() -> color_eyre::Report {
    eyre!("usage: decode <model.gguf> <depth> <context> <f16|int8> <tokens> <reps>")
}

fn parse(value: &str, field: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|_| eyre!("decode: invalid {field}"))
}

fn generate(engine: &Engine, prompt: &str, tokens: usize) -> Result<(GenerationStats, f64)> {
    let request = Request {
        messages: vec![Message {
            role: Role::User,
            content: prompt.into(),
        }],
        sampling: SamplingParams::greedy(),
        max_tokens: tokens,
    };
    let start = Instant::now();
    let mut stats = None;
    let mut failure = None;
    engine.generate_cached(CACHE_KEY, request, &mut |event| {
        match event {
            Event::Finished(value) => stats = Some(value),
            Event::Error(message) => failure = Some(message),
            Event::Phase(_) | Event::TextDelta(_) => {}
        }
        true
    });
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    if let Some(message) = failure {
        bail!(message);
    }
    Ok((
        stats.ok_or_else(|| eyre!("decode: generation did not finish"))?,
        wall_ms,
    ))
}

struct Summary {
    mean: f64,
    median: f64,
    stddev: f64,
}

fn summarize(values: &[f64]) -> Summary {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len().saturating_sub(1).max(1)) as f64;
    Summary {
        mean,
        median,
        stddev: variance.sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::summarize;

    #[test]
    fn summary_reports_sample_dispersion_and_median() {
        let summary = summarize(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(summary.mean, 2.5);
        assert_eq!(summary.median, 2.5);
        assert!((summary.stddev - 1.290_994_448_735_805_6).abs() < 1e-12);
    }
}
