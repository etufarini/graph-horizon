/*
 * gh_zero_engine — pure isolated-phase statistics
 * Computes checked rates, sample deviation/CV, arithmetic means, and
 * nearest-rank percentiles. It owns no model, clock, fixture, I/O, or policy.
 */

use color_eyre::eyre::{Result, bail};

pub(super) fn rate(tokens: usize, nanoseconds: u64) -> Result<f64> {
    if tokens == 0 || nanoseconds == 0 {
        bail!("invalid measurement");
    }
    let value = tokens as f64 * 1_000_000_000.0 / nanoseconds as f64;
    if !value.is_finite() || value <= 0.0 {
        bail!("invalid measurement");
    }
    Ok(value)
}

pub(super) fn mean_u64(values: &[u64]) -> Result<f64> {
    if values.is_empty() || values.contains(&0) {
        bail!("invalid measurement");
    }
    Ok(values.iter().map(|&value| value as f64).sum::<f64>() / values.len() as f64)
}

pub(super) fn mean_std_cv(values: &[f64]) -> Result<(f64, f64, f64)> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        bail!("invalid measurement");
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if !mean.is_finite() || mean <= 0.0 {
        bail!("invalid measurement");
    }
    let variance = if values.len() == 1 {
        0.0
    } else {
        values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (values.len() - 1) as f64
    };
    let stddev = variance.sqrt();
    Ok((mean, stddev, stddev / mean))
}

pub(super) fn nearest_rank(values: &[u64], percentile: f64) -> Result<u64> {
    if values.is_empty()
        || values.contains(&0)
        || !percentile.is_finite()
        || !(0.0..=1.0).contains(&percentile)
        || percentile == 0.0
    {
        bail!("invalid measurement");
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (percentile * sorted.len() as f64).ceil() as usize;
    Ok(sorted[rank.saturating_sub(1).min(sorted.len() - 1)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_uses_one_based_ceiling_and_final_clamp() {
        let values = (1..=31).collect::<Vec<_>>();
        assert_eq!(nearest_rank(&values, 0.5).unwrap(), 16);
        assert_eq!(nearest_rank(&values, 0.95).unwrap(), 30);
        assert_eq!(nearest_rank(&values, 1.0).unwrap(), 31);
    }

    #[test]
    fn repeated_values_have_zero_sample_deviation_and_cv() {
        assert_eq!(mean_std_cv(&[4.0; 7]).unwrap(), (4.0, 0.0, 0.0));
    }

    #[test]
    fn invalid_inputs_never_create_zero_or_non_finite_metrics() {
        for values in [&[][..], &[0.0][..], &[f64::NAN][..], &[f64::INFINITY][..]] {
            assert!(mean_std_cv(values).is_err());
        }
        assert!(rate(1, 0).is_err());
        assert!(mean_u64(&[1, 0]).is_err());
        assert!(nearest_rank(&[1], 0.0).is_err());
    }
}
