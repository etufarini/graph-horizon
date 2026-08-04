/*
 * GH Zero performance evidence boundary
 * Reads bounded untrusted JSONL, validates the fixed matrix schema and order,
 * and owns immutable A/B tuple checks. It owns no CLI or decision policy and
 * never exposes evidence paths or raw parser diagnostics.
 */
use std::{collections::BTreeMap, fs};

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

#[rustfmt::skip]
mod strict {
use super::*;

const ROW_KEYS: [&str; 38] = [
    "schema_version", "status", "reason", "revision", "backend_profile", "family", "model_id", "variant",
    "artifact_bytes", "artifact_sha256", "kv", "placement_mode", "cpu_layers", "gpu_layers", "weights_percent",
    "context", "fixture", "fixture_digest", "hardware_id", "driver_id", "warmup", "repetitions", "prompt_tokens",
    "decode_steps", "prefill_mean_ns", "prefill_tps_mean", "prefill_tps_stddev", "prefill_tps_cv",
    "first_sample_mean_ns", "decode_p50_mean_ns", "decode_p95_mean_ns", "decode_tps_mean", "decode_tps_stddev",
    "decode_tps_cv", "public_ttft_ms", "public_decode_tps", "cpu_memory_total", "gpu_memory_total",
];
const DERIVED_KEYS: [&str; 19] = [
    "cpu_layers", "gpu_layers", "fixture_digest", "prompt_tokens", "decode_steps", "prefill_mean_ns",
    "prefill_tps_mean", "prefill_tps_stddev", "prefill_tps_cv", "first_sample_mean_ns", "decode_p50_mean_ns",
    "decode_p95_mean_ns", "decode_tps_mean", "decode_tps_stddev", "decode_tps_cv", "public_ttft_ms",
    "public_decode_tps", "cpu_memory_total", "gpu_memory_total",
];
const PROFILES: [&str; 3] = ["cpu", "metal", "metal-hybrid"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Requested {
    pub(crate) profile: String, pub(crate) family: String, pub(crate) model: String,
    pub(crate) variant: String, pub(crate) bytes: u64, pub(crate) sha: String,
    pub(crate) kv: String, pub(crate) percent: Option<u64>, pub(crate) context: u64,
    pub(crate) fixture: String, pub(crate) hardware: String, pub(crate) driver: String,
    pub(crate) warmup: u64, pub(crate) repetitions: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct Row {
    pub(crate) status: String, pub(crate) reason: String, pub(crate) revision: String,
    pub(crate) requested: Requested, pub(crate) placement: String, pub(crate) digest: Option<String>,
    pub(crate) prompt_tokens: Option<u64>, pub(crate) decode_steps: Option<u64>,
    pub(crate) prefill_ns: Option<f64>, pub(crate) prefill_tps: Option<f64>, pub(crate) prefill_cv: Option<f64>,
    pub(crate) first_sample_ns: Option<f64>, pub(crate) decode_p50_ns: Option<f64>,
    pub(crate) decode_p95_ns: Option<f64>, pub(crate) decode_tps: Option<f64>,
    pub(crate) decode_cv: Option<f64>, pub(crate) public_ttft_ms: Option<f64>,
    pub(crate) public_decode_tps: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Counts { pub(crate) pass: u64, pub(crate) fail: u64, pub(crate) external: u64 }
pub(crate) struct Evidence { pub(crate) revision: String, pub(crate) rows: Vec<Row>, pub(crate) counts: Counts }

struct Entries(Vec<(String, Value)>);
impl<'de> Deserialize<'de> for Entries {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct EntriesVisitor;
        impl<'de> Visitor<'de> for EntriesVisitor {
            type Value = Entries;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("JSON object") }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Entries, A::Error> {
                let mut values = Vec::new();
                while let Some(entry) = map.next_entry()? { values.push(entry); }
                Ok(Entries(values))
            }
        }
        d.deserialize_map(EntriesVisitor)
    }
}

pub(crate) fn load(path: &str) -> Result<Evidence, ()> {
    let metadata = fs::metadata(path).map_err(|_| ())?;
    if !metadata.is_file() || metadata.len() > 1_048_576 { return Err(()); }
    let bytes = fs::read(path).map_err(|_| ())?;
    if bytes.len() > 1_048_576 || !bytes.ends_with(b"\n") { return Err(()); }
    parse_bytes(&bytes)
}

pub(super) fn parse_bytes(bytes: &[u8]) -> Result<Evidence, ()> {
    let text = std::str::from_utf8(bytes).map_err(|_| ())?;
    let lines = text.split_terminator('\n').collect::<Vec<_>>();
    if lines.len() != 13 || lines.iter().any(|line| line.len() > 32_768) { return Err(()); }
    let mut rows = Vec::with_capacity(12);
    for (index, line) in lines[..12].iter().enumerate() { rows.push(parse_row(line, index)?); }
    let (revision, counts) = parse_summary(lines[12])?;
    let actual = Counts { pass: rows.iter().filter(|row| row.status == "pass").count() as u64, fail: rows.iter().filter(|row| row.status == "fail").count() as u64, external: rows.iter().filter(|row| row.status == "external verification").count() as u64 };
    if counts.pass != actual.pass || counts.fail != actual.fail || counts.external != actual.external
        || rows.iter().any(|row| row.revision != revision) { return Err(()); }
    let first = &rows[0].requested;
    for row in &rows {
        if row.requested.hardware != first.hardware || row.requested.driver != first.driver { return Err(()); }
        let peer = rows.iter().find(|peer| peer.requested.model == row.requested.model).ok_or(())?;
        if row.requested.bytes != peer.requested.bytes || row.requested.sha != peer.requested.sha { return Err(()); }
    }
    Ok(Evidence { revision, rows, counts })
}

pub(crate) fn comparable_tuples(left: &Evidence, right: &Evidence) -> Result<(), ()> {
    for (baseline, candidate) in left.rows.iter().zip(&right.rows) {
        if baseline.requested != candidate.requested { return Err(()); }
        let (a, b) = (baseline.digest.as_ref(), candidate.digest.as_ref());
        if a.is_some() && b.is_some() && a != b { return Err(()); }
        if both_differ(baseline.prompt_tokens, candidate.prompt_tokens) || both_differ(baseline.decode_steps, candidate.decode_steps) {
            return Err(());
        }
    }
    Ok(())
}

fn both_differ<T: PartialEq>(left: Option<T>, right: Option<T>) -> bool { matches!((left, right), (Some(a), Some(b)) if a != b) }

fn parse_row(line: &str, index: usize) -> Result<Row, ()> {
    let map = object(line, &ROW_KEYS)?;
    if integer(&map, "schema_version")? != 2 { return Err(()); }
    let status = string(&map, "status")?; let reason = string(&map, "reason")?;
    let revision = string(&map, "revision")?; let profile = string(&map, "backend_profile")?;
    let requested = Requested {
        profile, family: string(&map, "family")?, model: string(&map, "model_id")?,
        variant: string(&map, "variant")?, bytes: integer(&map, "artifact_bytes")?,
        sha: string(&map, "artifact_sha256")?, kv: string(&map, "kv")?,
        percent: optional_integer(&map, "weights_percent")?, context: integer(&map, "context")?,
        fixture: string(&map, "fixture")?, hardware: string(&map, "hardware_id")?,
        driver: string(&map, "driver_id")?, warmup: integer(&map, "warmup")?,
        repetitions: integer(&map, "repetitions")?,
    };
    let row = Row {
        status, reason, revision, placement: string(&map, "placement_mode")?, requested,
        digest: optional_string(&map, "fixture_digest")?, prompt_tokens: optional_integer(&map, "prompt_tokens")?,
        decode_steps: optional_integer(&map, "decode_steps")?, prefill_ns: optional_number(&map, "prefill_mean_ns")?,
        prefill_tps: optional_number(&map, "prefill_tps_mean")?, prefill_cv: optional_number(&map, "prefill_tps_cv")?,
        first_sample_ns: optional_number(&map, "first_sample_mean_ns")?,
        decode_p50_ns: optional_number(&map, "decode_p50_mean_ns")?, decode_p95_ns: optional_number(&map, "decode_p95_mean_ns")?,
        decode_tps: optional_number(&map, "decode_tps_mean")?, decode_cv: optional_number(&map, "decode_tps_cv")?,
        public_ttft_ms: optional_number(&map, "public_ttft_ms")?, public_decode_tps: optional_number(&map, "public_decode_tps")?,
    };
    validate_row(&map, &row, index)?;
    Ok(row)
}

fn validate_row(map: &BTreeMap<String, Value>, row: &Row, index: usize) -> Result<(), ()> {
    let (model, profile, kv, fixture) = expected(index);
    let req = &row.requested;
    let hybrid = profile.ends_with("-hybrid");
    if req.profile != profile || req.model != model || req.kv != kv || req.fixture != fixture || req.family != "mistral3" || req.variant != "instruct" || req.bytes == 0 || !hex(&req.sha, 64) || !hex(&row.revision, 40) || !id(&req.hardware) || !id(&req.driver) || req.context != 4096 || req.warmup != 1 || req.repetitions != 3 || req.percent != hybrid.then_some(25) || row.placement != if hybrid { "mixed" } else { "pure" } || !reason_ok(&row.status, &row.reason) {
        return Err(());
    }
    let derived_null = DERIVED_KEYS.iter().all(|key| map[*key].is_null());
    if row.status != "pass" && !derived_null { return Err(()); }
    if row.status != "pass" { return Ok(()); }
    let cpu_layers = optional_integer(map, "cpu_layers")?;
    let gpu_layers = optional_integer(map, "gpu_layers")?;
    let _cpu_memory = optional_integer(map, "cpu_memory_total")?;
    let _gpu_memory = optional_integer(map, "gpu_memory_total")?;
    if hybrid && !(cpu_layers.is_some_and(|v| v > 0) && gpu_layers.is_some_and(|v| v > 0)) || !hybrid && (cpu_layers.is_some() || gpu_layers.is_some()) {
        return Err(());
    }
    let prompt = if fixture == "short" { 16 } else { 512 };
    if row.digest.as_ref().is_none_or(|v| !hex(v, 64)) || row.prompt_tokens != Some(prompt) || row.decode_steps != Some(31) || [row.prefill_ns, row.prefill_tps, row.first_sample_ns, row.decode_p50_ns, row.decode_p95_ns, row.decode_tps].iter().any(|v| v.is_none_or(|n| n <= 0.0)) || [row.prefill_cv, row.decode_cv].iter().any(|v| v.is_none_or(|n| n < 0.0)) || [row.public_ttft_ms, row.public_decode_tps].iter().any(|v| v.is_some_and(|n| n <= 0.0)) || optional_number(map, "prefill_tps_stddev")?.is_none_or(|v| v < 0.0) || optional_number(map, "decode_tps_stddev")?.is_none_or(|v| v < 0.0) {
        return Err(());
    }
    Ok(())
}

pub(super) fn expected(index: usize) -> (&'static str, &'static str, &'static str, &'static str) {
    let slot = index % 4;
    ("3b-instruct", PROFILES[index / 4], if slot < 2 { "f16" } else { "int8" }, if slot % 2 == 0 { "short" } else { "long" })
}

fn parse_summary(line: &str) -> Result<(String, Counts), ()> {
    let keys = ["schema_version", "summary", "total", "pass", "fail", "external_verification", "revision"];
    let map = object(line, &keys)?;
    if integer(&map, "schema_version")? != 2 || map.get("summary") != Some(&Value::Bool(true)) || integer(&map, "total")? != 12 {
        return Err(());
    }
    let counts = Counts { pass: integer(&map, "pass")?, fail: integer(&map, "fail")?, external: integer(&map, "external_verification")? };
    if counts.pass + counts.fail + counts.external != 12 { return Err(()); }
    let revision = string(&map, "revision")?;
    if !hex(&revision, 40) { return Err(()); }
    Ok((revision, counts))
}

fn object(line: &str, keys: &[&str]) -> Result<BTreeMap<String, Value>, ()> {
    let Entries(entries) = serde_json::from_str(line).map_err(|_| ())?;
    let mut map = BTreeMap::new();
    for (key, value) in entries { if !keys.contains(&key.as_str()) || map.insert(key, value).is_some() { return Err(()); } }
    if map.len() != keys.len() { return Err(()); }
    Ok(map)
}
fn string(map: &BTreeMap<String, Value>, key: &str) -> Result<String, ()> { map.get(key).and_then(Value::as_str).map(str::to_owned).ok_or(()) }
fn optional_string(map: &BTreeMap<String, Value>, key: &str) -> Result<Option<String>, ()> { if map[key].is_null() { Ok(None) } else { string(map, key).map(Some) } }
fn integer(map: &BTreeMap<String, Value>, key: &str) -> Result<u64, ()> { map.get(key).and_then(Value::as_u64).ok_or(()) }
fn optional_integer(map: &BTreeMap<String, Value>, key: &str) -> Result<Option<u64>, ()> { if map[key].is_null() { Ok(None) } else { integer(map, key).map(Some) } }
fn optional_number(map: &BTreeMap<String, Value>, key: &str) -> Result<Option<f64>, ()> {
    if map[key].is_null() { return Ok(None); }
    map[key].as_f64().filter(|value| value.is_finite()).map(Some).ok_or(())
}
fn id(value: &str) -> bool { (1..=96).contains(&value.len()) && value.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)) }
fn hex(value: &str, len: usize) -> bool { value.len() == len && value.bytes().all(|b| b.is_ascii_hexdigit()) }
fn reason_ok(status: &str, reason: &str) -> bool {
    match status {
        "pass" => reason == "ok",
        "external verification" => matches!(reason, "artifact unavailable" | "tool unavailable" | "platform unavailable" | "device unavailable" | "driver unavailable" | "capacity unavailable"),
        "fail" => matches!(reason, "artifact mismatch" | "placement mismatch" | "invalid fixture" | "invalid measurement" | "execution failed"),
        _ => false,
    }
}
}

pub(crate) use strict::{Evidence, Row, comparable_tuples, load};

#[cfg(test)]
pub(crate) fn sample_evidence(revision: &str, factor: f64) -> Evidence {
    let mut rows = Vec::new();
    for index in 0..12 {
        let (model, profile, kv, fixture) = strict::expected(index);
        let hybrid = profile.ends_with("-hybrid");
        rows.push(Row {
            status: "pass".into(),
            reason: "ok".into(),
            revision: revision.into(),
            requested: strict::Requested {
                profile: profile.into(),
                family: "mistral3".into(),
                model: model.into(),
                variant: "instruct".into(),
                bytes: 1,
                sha: "a".repeat(64),
                kv: kv.into(),
                percent: hybrid.then_some(25),
                context: 4096,
                fixture: fixture.into(),
                hardware: "host".into(),
                driver: "driver".into(),
                warmup: 1,
                repetitions: 3,
            },
            placement: if hybrid { "mixed" } else { "pure" }.into(),
            digest: Some("c".repeat(64)),
            prompt_tokens: Some(if fixture == "short" { 16 } else { 512 }),
            decode_steps: Some(31),
            prefill_ns: Some(100.0 / factor),
            prefill_tps: Some(100.0 * factor),
            prefill_cv: Some(0.01),
            first_sample_ns: Some(100.0),
            decode_p50_ns: Some(100.0 / factor),
            decode_p95_ns: Some(100.0 / factor),
            decode_tps: Some(100.0 * factor),
            decode_cv: Some(0.01),
            public_ttft_ms: Some(100.0 / factor),
            public_decode_tps: Some(100.0 * factor),
        });
    }
    Evidence {
        revision: revision.into(),
        rows,
        counts: strict::Counts {
            pass: 12,
            fail: 0,
            external: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::strict::*;

    fn matrix(pass: bool) -> String {
        let mut output = String::new();
        for index in 0..12 {
            let (model, profile, kv, fixture) = expected(index);
            let hybrid = profile.ends_with("-hybrid");
            let status = if pass {
                "pass"
            } else {
                "external verification"
            };
            let reason = if pass { "ok" } else { "artifact unavailable" };
            let row = serde_json::json!({
                "schema_version":2,"status":status,"reason":reason,
                "revision":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","backend_profile":profile,
                "family":"mistral3","model_id":model,"variant":"instruct","artifact_bytes":1,
                "artifact_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","kv":kv,
                "placement_mode":if hybrid {"mixed"} else {"pure"},
                "cpu_layers":if pass && hybrid {Some(1)} else {None},
                "gpu_layers":if pass && hybrid {Some(1)} else {None},
                "weights_percent":if hybrid {Some(25)} else {None},"context":4096,"fixture":fixture,
                "fixture_digest":pass.then(|| "c".repeat(64)),"hardware_id":"host","driver_id":"driver",
                "warmup":1,"repetitions":3,
                "prompt_tokens":pass.then_some(if fixture == "short" {16} else {512}),
                "decode_steps":pass.then_some(31),"prefill_mean_ns":pass.then_some(1.0),
                "prefill_tps_mean":pass.then_some(1.0),"prefill_tps_stddev":pass.then_some(0.0),
                "prefill_tps_cv":pass.then_some(0.0),"first_sample_mean_ns":pass.then_some(1.0),
                "decode_p50_mean_ns":pass.then_some(1.0),"decode_p95_mean_ns":pass.then_some(1.0),
                "decode_tps_mean":pass.then_some(1.0),"decode_tps_stddev":pass.then_some(0.0),
                "decode_tps_cv":pass.then_some(0.0),"public_ttft_ms":pass.then_some(1.0),
                "public_decode_tps":pass.then_some(1.0),"cpu_memory_total":null,"gpu_memory_total":null
            });
            output.push_str(&serde_json::to_string(&row).unwrap());
            output.push('\n');
        }
        output.push_str(if pass {
            "{\"schema_version\":2,\"summary\":true,\"total\":12,\"pass\":12,\"fail\":0,\"external_verification\":0,\"revision\":\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"}\n"
        } else {
            "{\"schema_version\":2,\"summary\":true,\"total\":12,\"pass\":0,\"fail\":0,\"external_verification\":12,\"revision\":\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"}\n"
        });
        output
    }

    #[test]
    fn accepts_complete_canonical_matrix_and_rejects_shape_errors() {
        let valid = matrix(false);
        assert!(parse_bytes(valid.as_bytes()).is_ok());
        let version_one = valid.replace("\"schema_version\":2", "\"schema_version\":1");
        assert!(parse_bytes(version_one.as_bytes()).is_err());
        let mut missing = valid.lines().map(str::to_owned).collect::<Vec<_>>();
        missing.remove(10);
        assert!(parse_bytes(format!("{}\n", missing.join("\n")).as_bytes()).is_err());
        assert!(parse_bytes(format!("{valid}{{}}\n").as_bytes()).is_err());
        let mut reordered = valid.lines().map(str::to_owned).collect::<Vec<_>>();
        reordered.swap(0, 1);
        assert!(parse_bytes(format!("{}\n", reordered.join("\n")).as_bytes()).is_err());
        let duplicate = valid.replacen("\"status\":", "\"status\":\"fail\",\"status\":", 1);
        assert!(parse_bytes(duplicate.as_bytes()).is_err());
        let derived = valid.replacen("\"prompt_tokens\":null", "\"prompt_tokens\":16", 1);
        assert!(parse_bytes(derived.as_bytes()).is_err());
        let pass = matrix(true);
        assert!(parse_bytes(pass.as_bytes()).is_ok());
        let old_long = pass.replacen("\"prompt_tokens\":512", "\"prompt_tokens\":2048", 1);
        assert!(parse_bytes(old_long.as_bytes()).is_err());
        let old_repetitions = pass.replacen("\"repetitions\":3", "\"repetitions\":7", 1);
        assert!(parse_bytes(old_repetitions.as_bytes()).is_err());
    }

    #[test]
    fn immutable_tuple_and_present_fixture_fields_must_match() {
        let baseline = super::sample_evidence("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 1.0);
        let mut candidate =
            super::sample_evidence("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", 1.05);
        assert!(comparable_tuples(&baseline, &candidate).is_ok());
        candidate.rows[0].requested.hardware = "other".into();
        assert!(comparable_tuples(&baseline, &candidate).is_err());
        candidate.rows[0].requested.hardware = "host".into();
        candidate.rows[0].digest = Some("d".repeat(64));
        assert!(comparable_tuples(&baseline, &candidate).is_err());
    }
}
