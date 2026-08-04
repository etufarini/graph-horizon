/*
 * gh_zero_engine — public isolated-phase measurement facade
 * Owns fixed warm-up/repetition orchestration, validates neutral runtime
 * samples, and publishes aggregate data for one fixture. It owns no fixture
 * token IDs, backend branches, model loading, JSON, or performance decisions.
 */

mod stats;

use color_eyre::eyre::{Result, bail};

use crate::Engine;
pub use crate::runtime::phases::PhaseFixture;
use crate::runtime::phases::{DECODE_STEPS, PhaseSample};

use stats::{mean_std_cv, mean_u64, nearest_rank, rate};

pub const WARMUP: usize = 1;
pub const REPETITIONS: usize = 3;

pub struct PhaseConfig {
    pub fixture: PhaseFixture,
}

pub struct PhaseReport {
    pub fixture: PhaseFixture,
    pub fixture_digest: String,
    pub prompt_tokens: usize,
    pub decode_steps: usize,
    pub warmup: usize,
    pub repetitions: usize,
    pub prefill_mean_ns: f64,
    pub prefill_tps_mean: f64,
    pub prefill_tps_stddev: f64,
    pub prefill_tps_cv: f64,
    pub first_sample_mean_ns: f64,
    pub decode_p50_mean_ns: f64,
    pub decode_p95_mean_ns: f64,
    pub decode_tps_mean: f64,
    pub decode_tps_stddev: f64,
    pub decode_tps_cv: f64,
    pub prefill_tps_repetitions: [f64; REPETITIONS],
    pub decode_tps_repetitions: [f64; REPETITIONS],
}

pub fn run(engine: &Engine, config: &PhaseConfig) -> Result<PhaseReport> {
    let (_, expected_digest) = engine.measure_phase(config.fixture)?;
    let mut samples = Vec::with_capacity(REPETITIONS);
    for _ in 0..REPETITIONS {
        let (sample, digest) = engine.measure_phase(config.fixture)?;
        if digest != expected_digest {
            bail!("invalid fixture");
        }
        samples.push(sample);
    }
    aggregate(config.fixture, expected_digest, &samples)
}

fn aggregate(
    fixture: PhaseFixture,
    fixture_digest: String,
    samples: &[PhaseSample],
) -> Result<PhaseReport> {
    if samples.len() != REPETITIONS {
        bail!("invalid measurement");
    }
    let prompt_tokens = samples[0].prompt_tokens;
    if prompt_tokens == 0
        || samples.iter().any(|sample| {
            sample.prompt_tokens != prompt_tokens
                || sample.decode_steps != DECODE_STEPS
                || sample.prefill_ns == 0
                || sample.first_sample_ns == 0
                || sample.decode_ns.contains(&0)
        })
    {
        bail!("invalid measurement");
    }

    let mut prefill_rates = [0.0; REPETITIONS];
    let mut decode_rates = [0.0; REPETITIONS];
    let mut p50 = [0; REPETITIONS];
    let mut p95 = [0; REPETITIONS];
    for (index, sample) in samples.iter().enumerate() {
        prefill_rates[index] = rate(prompt_tokens, sample.prefill_ns)?;
        let decode_total = sample.decode_ns.iter().try_fold(0u64, |sum, value| {
            sum.checked_add(*value)
                .ok_or_else(|| color_eyre::eyre::eyre!("invalid measurement"))
        })?;
        decode_rates[index] = rate(DECODE_STEPS, decode_total)?;
        p50[index] = nearest_rank(&sample.decode_ns, 0.50)?;
        p95[index] = nearest_rank(&sample.decode_ns, 0.95)?;
    }
    let (prefill_tps_mean, prefill_tps_stddev, prefill_tps_cv) = mean_std_cv(&prefill_rates)?;
    let (decode_tps_mean, decode_tps_stddev, decode_tps_cv) = mean_std_cv(&decode_rates)?;

    Ok(PhaseReport {
        fixture,
        fixture_digest,
        prompt_tokens,
        decode_steps: DECODE_STEPS,
        warmup: WARMUP,
        repetitions: REPETITIONS,
        prefill_mean_ns: mean_u64(&samples.iter().map(|s| s.prefill_ns).collect::<Vec<_>>())?,
        prefill_tps_mean,
        prefill_tps_stddev,
        prefill_tps_cv,
        first_sample_mean_ns: mean_u64(
            &samples
                .iter()
                .map(|s| s.first_sample_ns)
                .collect::<Vec<_>>(),
        )?,
        decode_p50_mean_ns: mean_u64(&p50)?,
        decode_p95_mean_ns: mean_u64(&p95)?,
        decode_tps_mean,
        decode_tps_stddev,
        decode_tps_cv,
        prefill_tps_repetitions: prefill_rates,
        decode_tps_repetitions: decode_rates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(value: u64) -> PhaseSample {
        PhaseSample {
            prefill_ns: value,
            first_sample_ns: value,
            decode_ns: [value; DECODE_STEPS],
            prompt_tokens: 16,
            decode_steps: DECODE_STEPS,
        }
    }

    #[test]
    fn three_valid_samples_are_aggregated_without_losing_cv_inputs() {
        assert_eq!(WARMUP, 1);
        assert_eq!(REPETITIONS, 3);
        let samples = (1..=REPETITIONS).map(|_| sample(1_000)).collect::<Vec<_>>();
        let report = aggregate(PhaseFixture::Short, "digest".into(), &samples).unwrap();
        assert_eq!(report.prompt_tokens, 16);
        assert_eq!(report.decode_p50_mean_ns, 1_000.0);
        assert_eq!(report.decode_p95_mean_ns, 1_000.0);
        assert_eq!(report.prefill_tps_cv, 0.0);
        assert_eq!(report.decode_tps_cv, 0.0);
        assert_eq!(report.prefill_tps_repetitions, [16_000_000.0; REPETITIONS]);
    }

    #[test]
    fn invalid_durations_and_decode_counts_are_rejected() {
        let mut samples = (1..=REPETITIONS).map(|_| sample(1)).collect::<Vec<_>>();
        samples[0].prefill_ns = 0;
        assert!(aggregate(PhaseFixture::Short, "digest".into(), &samples).is_err());
        samples[0] = sample(1);
        samples[0].decode_steps = 30;
        assert!(aggregate(PhaseFixture::Short, "digest".into(), &samples).is_err());
    }
}
