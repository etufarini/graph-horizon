/*
 * Graph Horizon runtime profile example
 * Reports immutable placement/memory fields and one up-to-32-token timing sample for
 * an explicit tuple. It rejects incomplete metrics and never retries another
 * backend, context, KV scheme, or placement.
 */

use std::path::Path;

use color_eyre::eyre::{Result, eyre};
use graph_horizon_engine::harness::throughput::{self, BenchConfig};
use graph_horizon_engine::{BackendMemory, Engine, EngineConfig, KvQuant};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .ok_or_else(|| eyre!("usage: profile <model.gguf> <context> <f16|int8>"))?;
    let context = args
        .next()
        .ok_or_else(|| eyre!("missing context"))?
        .parse::<usize>()?;
    let kv = args
        .next()
        .and_then(|value| KvQuant::parse(&value))
        .ok_or_else(|| eyre!("KV must be f16 or int8"))?;
    let weights_percent = match (args.next(), args.next()) {
        (None, None) => None,
        (Some(flag), Some(value)) if flag == "--weights-percent" => Some(
            value
                .parse::<u8>()
                .map_err(|_| eyre!("weight percentage must be in 0..100"))?,
        ),
        _ => return Err(eyre!("invalid profile arguments")),
    };
    if context == 0 || weights_percent.is_some_and(|value| value > 100) || args.next().is_some() {
        return Err(eyre!("invalid profile arguments"));
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
    if let Some(plan) = engine.placement() {
        println!(
            "mode={} cpu_layers={} gpu_layers={}",
            plan.mode, plan.cpu_layers, plan.gpu_layers
        );
        print_memory("cpu", plan.cpu);
        print_memory("gpu", plan.gpu);
    } else {
        println!("placement: pure backend build (no hybrid plan)");
    }
    let report = throughput::run(
        &engine,
        &BenchConfig {
            preset: format!("context={context},kv={}", kv.name()),
            prompt: "Ciao".into(),
            gen_tokens: 32,
            reps: 1,
            warmup: 0,
        },
    )?;
    let ttft_ms = report.ttft_seconds.mean * 1000.0;
    let prompt_tps = report.prompt_tps.mean;
    let decode_tps = report
        .tg
        .ok_or_else(|| eyre!("profile produced fewer than two decoded tokens"))?
        .mean;
    if !ttft_ms.is_finite() || !prompt_tps.is_finite() || !decode_tps.is_finite() {
        return Err(eyre!("profile produced a non-finite metric"));
    }
    println!(
        "prompt_tokens={} decoded_tokens={} prompt_tps={:.2} ttft_ms={:.2} decode_tps={:.2}",
        report.n_prompt, report.decoded_tokens, prompt_tps, ttft_ms, decode_tps
    );
    Ok(())
}

fn print_memory(side: &str, bytes: BackendMemory) {
    println!(
        "{side}: weights={} kv={} scratch={} fixed={} staging={} crossing={} reserve={} total={}",
        bytes.weights,
        bytes.kv,
        bytes.scratch,
        bytes.fixed,
        bytes.staging,
        bytes.crossing,
        bytes.reserve,
        bytes.total
    );
}
