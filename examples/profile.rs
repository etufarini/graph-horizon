/*
 * GH Zero runtime profile example
 * Reports the immutable public placement/memory fields and one family-neutral
 * chat timing sample. It does not trace kernels or retry another backend,
 * context, or KV scheme.
 */

use std::path::Path;

use color_eyre::eyre::{Result, eyre};
use gh_zero_engine::harness::throughput::{self, BenchConfig};
use gh_zero_engine::{BackendMemory, Engine, EngineConfig, KvQuant};

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
            // One pinned-parity token measures TTFT without deliberately
            // ending inside a later multi-token UTF-8 code point.
            prompt: "Hello".into(),
            gen_tokens: 1,
            reps: 1,
            warmup: 0,
        },
    )?;
    println!(
        "prompt_tokens={} prompt_tps={:.2} ttft_ms={:.2} decoded_tokens={}",
        report.n_prompt,
        report.prompt_tps.mean,
        report.ttft_seconds.mean * 1000.0,
        report.decoded_tokens
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
