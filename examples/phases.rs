/*
 * GH Zero isolated phase benchmark row
 * Parses one immutable model/profile tuple, runs the family-neutral phase and
 * public-control harnesses, and emits one bounded JSON record. All CLI values
 * are untrusted; this file owns no artifact lookup, matrix order, or comparison.
 */

use std::path::Path;

use gh_zero_engine::harness::phases::{self, PhaseConfig, PhaseFixture, PhaseReport};
use gh_zero_engine::harness::throughput::{self, BenchConfig, ThroughputReport};
use gh_zero_engine::{Engine, EngineConfig, KvQuant, PlacementReport};
use serde::Serialize;

#[rustfmt::skip]
#[derive(Clone)]
struct Args {
    model: String, context: usize, kv: KvQuant, fixture: PhaseFixture,
    weights_percent: Option<u8>, model_id: String, artifact_bytes: u64,
    artifact_sha256: String, revision: String, hardware_id: String, driver_id: String,
}

fn main() {
    let args = match parse(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(()) => {
            eprintln!("benchmark: invalid arguments");
            std::process::exit(2);
        }
    };
    println!("{}", execute(&args));
}

fn execute(args: &Args) -> String {
    let engine = match Engine::new(
        Path::new(&args.model),
        EngineConfig {
            context_tokens: Some(args.context),
            vram_weights_percent: args.weights_percent,
            kv_quant: args.kv,
            ..EngineConfig::default()
        },
    ) {
        Ok(engine) => engine,
        Err(error) => {
            let external = error.to_string().contains("backend is unavailable");
            let terminal = if external {
                ("external verification", "device unavailable")
            } else {
                ("fail", "execution failed")
            };
            return row(args, terminal.0, terminal.1, None, None, None);
        }
    };
    let placement = engine.placement();
    if cfg!(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))
        && !placement.is_some_and(|value| {
            value.mode == "mixed" && value.cpu_layers > 0 && value.gpu_layers > 0
        })
    {
        return row(args, "fail", "placement mismatch", None, None, placement);
    }
    let config = PhaseConfig {
        fixture: args.fixture,
    };
    let report = match phases::run(&engine, &config) {
        Ok(report) => report,
        Err(error) => {
            let reason = match error.to_string().as_str() {
                "invalid fixture" => "invalid fixture",
                "invalid measurement" => "invalid measurement",
                _ => "execution failed",
            };
            return row(args, "fail", reason, None, None, placement);
        }
    };
    let control_config = BenchConfig {
        preset: "phase-control".into(),
        prompt: "Ciao".into(),
        gen_tokens: 32,
        reps: phases::REPETITIONS,
        warmup: phases::WARMUP,
    };
    let control = match throughput::run(&engine, &control_config) {
        Ok(control) => control,
        Err(_) => return row(args, "fail", "execution failed", None, None, placement),
    };
    row(args, "pass", "ok", Some(&report), Some(&control), placement)
}

#[rustfmt::skip]
fn row(
    args: &Args,
    status: &str,
    reason: &str,
    report: Option<&PhaseReport>,
    control: Option<&ThroughputReport>,
    placement: Option<PlacementReport>,
) -> String {
    // Terminal rows expose only the immutable requested tuple; placement and
    // every other inference result become visible only after a successful run.
    let successful = status == "pass";
    let measured_placement = successful.then_some(placement).flatten();
    let mode = measured_placement.map_or(
        if args.weights_percent.is_some() { "mixed" } else { "pure" },
        |value| value.mode,
    );
    let report = successful.then_some(report).flatten();
    let control = successful.then_some(control).flatten();
    format!(
        concat!(
            "{{\"schema_version\":1,\"status\":{},\"reason\":{},\"revision\":{},",
            "\"backend_profile\":{},\"family\":\"mistral3\",\"model_id\":{},\"variant\":\"instruct\",",
            "\"artifact_bytes\":{},\"artifact_sha256\":{},\"kv\":{},\"placement_mode\":{},",
            "\"cpu_layers\":{},\"gpu_layers\":{},\"weights_percent\":{},\"context\":{},",
            "\"fixture\":{},\"fixture_digest\":{},\"hardware_id\":{},\"driver_id\":{},",
            "\"warmup\":{},\"repetitions\":{},\"prompt_tokens\":{},\"decode_steps\":{},",
            "\"prefill_mean_ns\":{},\"prefill_tps_mean\":{},\"prefill_tps_stddev\":{},\"prefill_tps_cv\":{},",
            "\"first_sample_mean_ns\":{},\"decode_p50_mean_ns\":{},\"decode_p95_mean_ns\":{},",
            "\"decode_tps_mean\":{},\"decode_tps_stddev\":{},\"decode_tps_cv\":{},",
            "\"public_ttft_ms\":{},\"public_decode_tps\":{},\"cpu_memory_total\":{},\"gpu_memory_total\":{}}}"
        ),
        json(status), json(reason), json(&args.revision), json(profile()), json(&args.model_id),
        args.artifact_bytes, json(&args.artifact_sha256), json(args.kv.name()), json(mode),
        option(measured_placement.map(|v| v.cpu_layers)), option(measured_placement.map(|v| v.gpu_layers)),
        option(args.weights_percent), args.context, json(fixture_name(args.fixture)),
        option(report.map(|v| v.fixture_digest.as_str())), json(&args.hardware_id), json(&args.driver_id),
        phases::WARMUP, phases::REPETITIONS, option(report.map(|v| v.prompt_tokens)),
        option(report.map(|v| v.decode_steps)), option(report.map(|v| v.prefill_mean_ns)),
        option(report.map(|v| v.prefill_tps_mean)), option(report.map(|v| v.prefill_tps_stddev)),
        option(report.map(|v| v.prefill_tps_cv)), option(report.map(|v| v.first_sample_mean_ns)),
        option(report.map(|v| v.decode_p50_mean_ns)), option(report.map(|v| v.decode_p95_mean_ns)),
        option(report.map(|v| v.decode_tps_mean)), option(report.map(|v| v.decode_tps_stddev)),
        option(report.map(|v| v.decode_tps_cv)), option(control.map(|v| v.ttft_seconds.mean * 1_000.0)),
        option(control.and_then(|v| v.tg.as_ref().map(|stat| stat.mean))),
        option(measured_placement.map(|v| v.cpu.total)), option(measured_placement.map(|v| v.gpu.total)),
    )
}

#[rustfmt::skip]
fn parse(mut input: impl Iterator<Item = String>) -> Result<Args, ()> {
    let model = input.next().filter(|value| !value.is_empty()).ok_or(())?;
    let mut values = std::collections::BTreeMap::new();
    while let Some(flag) = input.next() {
        if !flag.starts_with("--") || values.insert(flag, input.next().ok_or(())?).is_some() {
            return Err(());
        }
    }
    let take = |name: &str| values.get(name).map(String::as_str).ok_or(());
    let context = decimal(take("--context")?)? as usize;
    let kv = KvQuant::parse(take("--kv")?).ok_or(())?;
    let fixture = match take("--fixture")? { "short" => PhaseFixture::Short,
        "long" => PhaseFixture::Long, _ => return Err(()) };
    let weights_percent = values.get("--weights-percent")
        .map(|value| u8::try_from(decimal(value)?).map_err(|_| ()))
        .transpose()?;
    let model_id = take("--model-id")?.to_owned();
    if take("--variant")? != "instruct" { return Err(()); }
    let artifact_bytes = decimal(take("--artifact-bytes")?)?;
    let artifact_sha256 = take("--artifact-sha256")?.to_owned();
    let revision = take("--revision")?.to_owned();
    let hardware_id = take("--hardware-id")?.to_owned();
    let driver_id = take("--driver-id")?.to_owned();
    let expected = 10 + usize::from(weights_percent.is_some());
    if values.len() != expected || context != 4096 || artifact_bytes == 0
        || !id(&model_id) || !id(&hardware_id) || !id(&driver_id)
        || !hex(&artifact_sha256, 64) || !hex(&revision, 40)
        || cfg!(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))
            != (weights_percent == Some(25))
    { return Err(()); }
    Ok(Args {
        model, context, kv, fixture, weights_percent, model_id, artifact_bytes,
        artifact_sha256, revision, hardware_id, driver_id,
    })
}

#[rustfmt::skip]
fn decimal(value: &str) -> Result<u64, ()> {
    if value.is_empty() || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit()) { return Err(()); }
    value.parse().map_err(|_| ())
}

#[rustfmt::skip]
fn id(value: &str) -> bool { (1..=96).contains(&value.len()) && value.bytes()
    .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte)) }

fn hex(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
fn json(value: &str) -> String {
    serde_json::to_string(value).expect("strings serialize")
}
fn option(value: Option<impl Serialize>) -> String {
    serde_json::to_string(&value).expect("metrics serialize")
}

fn fixture_name(value: PhaseFixture) -> &'static str {
    match value {
        PhaseFixture::Short => "short",
        PhaseFixture::Long => "long",
    }
}

#[rustfmt::skip]
fn profile() -> &'static str {
    if cfg!(feature = "cpu") { "cpu" } else if cfg!(feature = "vulkan") { "vulkan" }
    else if cfg!(feature = "vulkan-hybrid") { "vulkan-hybrid" }
    else if cfg!(feature = "metal") { "metal" } else { "metal-hybrid" }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gh_zero_engine::BackendMemory;

    fn valid() -> Vec<String> {
        let mut args = "model.gguf --context 4096 --kv f16 --fixture short --model-id 3b-instruct --variant instruct --artifact-bytes 1 --artifact-sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa --revision bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb --hardware-id host --driver-id driver"
            .split_whitespace().map(str::to_owned).collect::<Vec<_>>();
        if cfg!(any(feature = "vulkan-hybrid", feature = "metal-hybrid")) {
            args.extend(["--weights-percent".into(), "25".into()]);
        }
        args
    }

    #[test]
    fn strict_cli_accepts_only_one_complete_canonical_tuple() {
        assert!(parse(valid().into_iter()).is_ok());
        for extra in ["--context", "--unknown", "trailing"] {
            let mut args = valid();
            args.extend([extra.into(), "1".into()]);
            assert!(parse(args.into_iter()).is_err());
        }
        for invalid in ["04096", "0", "+4096", "4097"] {
            let mut args = valid();
            let index = args.iter().position(|v| v == "4096").unwrap();
            args[index] = invalid.into();
            assert!(parse(args.into_iter()).is_err());
        }
    }

    #[test]
    fn pure_profile_rejects_hybrid_percentage_and_json_order_is_fixed() {
        let args = parse(valid().into_iter()).unwrap();
        let json = row(&args, "fail", "execution failed", None, None, None);
        assert!(json.starts_with("{\"schema_version\":1,\"status\":\"fail\",\"reason\":"));
        assert!(serde_json::from_str::<serde_json::Value>(&json).is_ok());
        if !cfg!(any(feature = "vulkan-hybrid", feature = "metal-hybrid")) {
            let mut invalid = valid();
            invalid.extend(["--weights-percent".into(), "25".into()]);
            assert!(parse(invalid.into_iter()).is_err());
        }
    }

    #[test]
    fn non_pass_rows_hide_all_inference_derived_fields() {
        let args = parse(valid().into_iter()).unwrap();
        let placement = PlacementReport {
            mode: "mixed",
            cpu_layers: 1,
            gpu_layers: 2,
            cpu: BackendMemory {
                weights: 3,
                kv: 4,
                scratch: 5,
                total: 12,
                ..BackendMemory::default()
            },
            gpu: BackendMemory {
                weights: 6,
                kv: 7,
                scratch: 8,
                total: 21,
                ..BackendMemory::default()
            },
        };
        for (status, reason) in [
            ("fail", "placement mismatch"),
            ("external verification", "device unavailable"),
        ] {
            let value: serde_json::Value =
                serde_json::from_str(&row(&args, status, reason, None, None, Some(placement)))
                    .unwrap();
            assert_eq!(
                value["placement_mode"],
                if args.weights_percent.is_some() {
                    "mixed"
                } else {
                    "pure"
                }
            );
            for field in [
                "cpu_layers",
                "gpu_layers",
                "fixture_digest",
                "prompt_tokens",
                "decode_steps",
                "prefill_mean_ns",
                "prefill_tps_mean",
                "prefill_tps_stddev",
                "prefill_tps_cv",
                "first_sample_mean_ns",
                "decode_p50_mean_ns",
                "decode_p95_mean_ns",
                "decode_tps_mean",
                "decode_tps_stddev",
                "decode_tps_cv",
                "public_ttft_ms",
                "public_decode_tps",
                "cpu_memory_total",
                "gpu_memory_total",
            ] {
                assert!(value[field].is_null(), "{field} must be null");
            }
        }
    }
}
