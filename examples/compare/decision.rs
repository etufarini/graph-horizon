/*
 * GH Zero performance comparison policy
 * Evaluates already-validated A/B matrices using fixed variance, coverage,
 * regression, and geometric-mean rules. It performs no I/O, parsing, backend
 * work, or evidence mutation.
 */
use serde::Serialize;

use super::evidence::{Evidence, Row};

#[rustfmt::skip]
mod policy {
use super::*;

#[derive(Serialize)]
pub(crate) struct Decision<'a> {
    schema_version: u8,
    decision: &'a str,
    reason: &'a str,
    target: &'a str,
    attempt: u8,
    baseline_revision: &'a str,
    candidate_revision: &'a str,
    baseline_pass: u64,
    baseline_fail: u64,
    baseline_external_verification: u64,
    candidate_pass: u64,
    candidate_fail: u64,
    candidate_external_verification: u64,
    comparable_primary_rows: usize,
    comparable_scale_rows: usize,
    measured_cpu: bool,
    measured_pure_device: bool,
    prefill_geomean_ratio: Option<f64>,
    decode_geomean_ratio: Option<f64>,
    regression_row: Option<usize>,
    regression_metric: Option<&'a str>,
    regression_ratio: Option<f64>,
    capacity_regression: bool,
}

impl Decision<'_> {
    pub(crate) fn exit_code(&self) -> i32 {
        match self.decision {
            "keep" => 0,
            "revert" => 1,
            "repeat required" => 3,
            _ => 4,
        }
    }
}

#[derive(Clone, Copy)]
struct Regression {
    row: usize,
    metric: &'static str,
    ratio: Option<f64>,
    capacity: bool,
}

pub(crate) fn evaluate<'a>(baseline: &'a Evidence, candidate: &'a Evidence, target: &'a str, attempt: u8) -> Decision<'a> {
    let comparable = baseline.rows.iter().zip(&candidate.rows).enumerate().filter(|(_, (a, b))| a.status == "pass" && b.status == "pass").collect::<Vec<_>>();
    let primary = comparable.iter().filter(|(index, _)| *index < 20).collect::<Vec<_>>();
    let prefill = geomean(primary.iter().map(|(_, (a, b))| ratio(b.prefill_tps, a.prefill_tps)));
    let decode = geomean(primary.iter().map(|(_, (a, b))| ratio(b.decode_tps, a.decode_tps)));
    let measured_cpu = primary.iter().any(|(_, (_, row))| row.requested.profile == "cpu");
    let measured_pure_device = primary.iter().any(|(_, (_, row))| matches!(row.requested.profile.as_str(), "vulkan" | "metal"));
    let unstable = baseline.rows.iter().chain(&candidate.rows).filter(|row| row.status == "pass").any(|row| row.prefill_cv.is_some_and(|v| v > 0.05) || row.decode_cv.is_some_and(|v| v > 0.05));
    let regression = capacity_regression(baseline, candidate).or_else(|| first_regression(baseline, candidate));
    let target_met = match target {
        "prefill" => prefill.is_some_and(|v| v >= 1.05),
        "decode" => decode.is_some_and(|v| v >= 1.05),
        _ => prefill.is_some_and(|v| v >= 1.05) && decode.is_some_and(|v| v >= 1.05),
    };
    let (decision, reason) = if unstable && attempt == 1 {
        ("repeat required", "unstable measurement")
    } else if unstable {
        ("revert", "unstable measurement")
    } else if regression.is_some_and(|r| r.capacity) {
        ("revert", "capacity regression")
    } else if regression.is_some() {
        ("revert", "row regression")
    } else if !measured_cpu || !measured_pure_device {
        ("external verification", "insufficient measured hardware")
    } else if !target_met {
        ("revert", "target missed")
    } else {
        ("keep", "target met")
    };
    Decision {
        schema_version: 1,
        decision,
        reason,
        target,
        attempt,
        baseline_revision: &baseline.revision,
        candidate_revision: &candidate.revision,
        baseline_pass: baseline.counts.pass,
        baseline_fail: baseline.counts.fail,
        baseline_external_verification: baseline.counts.external,
        candidate_pass: candidate.counts.pass,
        candidate_fail: candidate.counts.fail,
        candidate_external_verification: candidate.counts.external,
        comparable_primary_rows: primary.len(),
        comparable_scale_rows: comparable.len() - primary.len(),
        measured_cpu,
        measured_pure_device,
        prefill_geomean_ratio: prefill,
        decode_geomean_ratio: decode,
        regression_row: regression.map(|r| r.row + 1),
        regression_metric: regression.map(|r| r.metric),
        regression_ratio: regression.and_then(|r| r.ratio),
        capacity_regression: regression.is_some_and(|r| r.capacity),
    }
}

fn capacity_regression(baseline: &Evidence, candidate: &Evidence) -> Option<Regression> {
    baseline
        .rows
        .iter()
        .zip(&candidate.rows)
        .enumerate()
        .find_map(|(row, (a, b))| (a.status == "pass" && b.reason == "capacity unavailable").then_some(Regression { row, metric: "capacity", ratio: None, capacity: true }))
}

fn first_regression(baseline: &Evidence, candidate: &Evidence) -> Option<Regression> {
    baseline.rows.iter().zip(&candidate.rows).enumerate().find_map(|(row, (a, b))| {
        let found = if a.status == "pass" && b.status != "pass" {
            Some(("status", None))
        } else if a.status == "pass" && b.status == "pass" && a.placement != b.placement {
            Some(("placement", None))
        } else if a.status != "pass" || b.status != "pass" {
            None
        } else {
            metric_regression(a, b)
        };
        found.map(|(metric, ratio)| Regression { row, metric, ratio, capacity: false })
    })
}

fn metric_regression(a: &Row, b: &Row) -> Option<(&'static str, Option<f64>)> {
    let checks = [
        ("prefill_tps", ratio(b.prefill_tps, a.prefill_tps), false),
        ("decode_tps", ratio(b.decode_tps, a.decode_tps), false),
        ("public_decode_tps", ratio(b.public_decode_tps, a.public_decode_tps), false),
        ("prefill_latency", ratio(b.prefill_ns, a.prefill_ns), true),
        ("first_sample_latency", ratio(b.first_sample_ns, a.first_sample_ns), true),
        ("decode_p50", ratio(b.decode_p50_ns, a.decode_p50_ns), true),
        ("decode_p95", ratio(b.decode_p95_ns, a.decode_p95_ns), true),
        ("public_ttft", ratio(b.public_ttft_ms, a.public_ttft_ms), true),
    ];
    checks.into_iter().find_map(|(name, value, latency)| value.filter(|ratio| if latency { *ratio > 1.02 } else { *ratio < 0.98 }).map(|ratio| (name, Some(ratio))))
}

fn ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> { Some(numerator? / denominator?) }
fn geomean(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let values = values.collect::<Option<Vec<_>>>()?;
    (!values.is_empty()).then(|| (values.iter().map(|value| value.ln()).sum::<f64>() / values.len() as f64).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::sample_evidence;

    const BASE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const CANDIDATE: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    #[test]
    fn target_pass_and_miss_follow_geometric_mean() {
        let baseline = sample_evidence(BASE, 1.0);
        let keep = sample_evidence(CANDIDATE, 1.05);
        let result = evaluate(&baseline, &keep, "both", 1);
        assert_eq!((result.decision, result.reason, result.exit_code()), ("keep", "target met", 0));
        let miss = sample_evidence(CANDIDATE, 1.04);
        let result = evaluate(&baseline, &miss, "both", 1);
        assert_eq!((result.decision, result.reason, result.exit_code()), ("revert", "target missed", 1));
    }

    #[test]
    fn row_and_capacity_regressions_have_fixed_precedence() {
        let baseline = sample_evidence(BASE, 1.0);
        let mut row = sample_evidence(CANDIDATE, 1.05);
        row.rows[4].decode_tps = Some(97.0);
        let result = evaluate(&baseline, &row, "decode", 1);
        assert_eq!((result.reason, result.regression_row, result.regression_metric), ("row regression", Some(5), Some("decode_tps")));

        let mut capacity = sample_evidence(CANDIDATE, 1.05);
        capacity.rows[9].status = "external verification".into();
        capacity.rows[9].reason = "capacity unavailable".into();
        capacity.counts.pass = 11; capacity.counts.external = 1;
        let result = evaluate(&baseline, &capacity, "decode", 1);
        assert_eq!((result.reason, result.regression_row, result.regression_metric), ("capacity regression", Some(10), Some("capacity")));
        assert!(result.capacity_regression);
    }

    #[test]
    fn instability_repeats_once_then_reverts() {
        let baseline = sample_evidence(BASE, 1.0);
        let mut candidate = sample_evidence(CANDIDATE, 1.05);
        candidate.rows[0].prefill_cv = Some(0.051);
        let first = evaluate(&baseline, &candidate, "prefill", 1);
        assert_eq!((first.decision, first.reason, first.exit_code()), ("repeat required", "unstable measurement", 3));
        let second = evaluate(&baseline, &candidate, "prefill", 2);
        assert_eq!((second.decision, second.reason, second.exit_code()), ("revert", "unstable measurement", 1));
    }

    #[test]
    fn missing_pure_device_coverage_is_external() {
        let mut baseline = sample_evidence(BASE, 1.0);
        let mut candidate = sample_evidence(CANDIDATE, 1.05);
        for index in 4..8 {
            baseline.rows[index].status = "external verification".into();
            baseline.rows[index].reason = "device unavailable".into();
            candidate.rows[index].status = "external verification".into();
            candidate.rows[index].reason = "device unavailable".into();
        }
        baseline.counts.pass = 8; baseline.counts.external = 4;
        candidate.counts.pass = 8; candidate.counts.external = 4;
        let result = evaluate(&baseline, &candidate, "both", 1);
        assert_eq!((result.decision, result.reason, result.exit_code()),
            ("external verification", "insufficient measured hardware", 4));
    }
}
}

pub(crate) use policy::evaluate;
