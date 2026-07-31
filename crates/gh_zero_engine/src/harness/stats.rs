/*
 * gh_zero_engine — throughput statistics (harness, pure)
 * The pure aggregation and per-rep segment math behind the throughput bench:
 * mean + sample standard deviation, safe rates, the per-rep metric derivation and
 * the cross-rep aggregation. No model, no I/O — unit tested without a backend. The
 * data types (`Stat`, `RepSample`) live next to the measurement in `throughput`.
*/

use super::throughput::{RepSample, Stat};

// Mean + sample standard deviation of a non-empty set of per-rep values. With a
// single observation σ is undefined ⇒ `None` (n/a), never `0`. Called only on
// non-empty sets (the aggregation gathers `Some` values; the empty case is mapped
// to `None` upstream), but it guards the empty input defensively.
pub(super) fn mean_std(xs: &[f64]) -> Stat {
    let n = xs.len();
    if n == 0 {
        return Stat {
            mean: 0.0,
            stddev: None,
        };
    }
    let mean = xs.iter().sum::<f64>() / n as f64;
    let stddev = if n < 2 {
        None // one observation: dispersion not defined
    } else {
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        Some(var.sqrt())
    };
    Stat { mean, stddev }
}

// A rate `count / duration`, but only when the duration is finite and strictly
// positive; a degenerate (zero/non-finite) duration yields `None` rather than an
// infinity or NaN that would poison the aggregation.
pub(super) fn rate(count: f64, duration: f64) -> Option<f64> {
    if duration.is_finite() && duration > 0.0 {
        Some(count / duration)
    } else {
        None
    }
}

// Derives one rep's metrics from the delta `offsets` (seconds since `t_start`,
// monotonically non-decreasing) and `n_prompt`. Pure, so the segment math is unit
// tested without a model. Segments split the generation positions into two
// contiguous halves and reuse the already-collected timestamps:
//   m = d/2; tg_first = (m − 1)/(t_m − t_1); tg_last = (d − m − 1)/(t_d − t_{m+1}).
pub(super) fn rep_metrics(offsets: &[f64], n_prompt: usize) -> RepSample {
    let d = offsets.len();
    let ttft_s = offsets.first().copied();
    let prompt_tps = ttft_s.and_then(|t| rate(n_prompt as f64, t));

    // `tg` needs ≥ 2 deltas (one full inter-delta interval).
    let tg = if d >= 2 {
        rate((d - 1) as f64, offsets[d - 1] - offsets[0])
    } else {
        None
    };

    // Segments need ≥ 4 deltas (≥ 2 per half). First half: positions 1..m; last
    // half: positions m+1..d (1-based, as documented above).
    let (tg_first, tg_last) = if d >= 4 {
        let m = d / 2;
        let first = rate((m - 1) as f64, offsets[m - 1] - offsets[0]);
        let last = rate((d - m - 1) as f64, offsets[d - 1] - offsets[m]);
        (first, last)
    } else {
        (None, None)
    };

    RepSample {
        deltas: d,
        ttft_s,
        prompt_tps,
        tg,
        tg_first,
        tg_last,
    }
}

// Aggregates the `Some` values selected by `pick` across the samples into a
// `Stat`, or `None` when no rep contributed a value (empty set ⇒ not reportable).
pub(super) fn aggregate(
    samples: &[RepSample],
    pick: impl Fn(&RepSample) -> Option<f64>,
) -> Option<Stat> {
    let xs: Vec<f64> = samples.iter().filter_map(&pick).collect();
    if xs.is_empty() {
        None
    } else {
        Some(mean_std(&xs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
    }

    #[test]
    fn mean_std_basic() {
        // mean of [2,4,6] = 4; sample variance = ((4+0+4)/2) = 4 ⇒ σ = 2.
        let s = mean_std(&[2.0, 4.0, 6.0]);
        approx(s.mean, 4.0);
        approx(s.stddev.unwrap(), 2.0);
    }

    #[test]
    fn mean_std_single_observation_has_no_stddev() {
        let s = mean_std(&[7.0]);
        approx(s.mean, 7.0);
        assert!(s.stddev.is_none());
    }

    #[test]
    fn mean_std_empty_is_defined() {
        let s = mean_std(&[]);
        approx(s.mean, 0.0);
        assert!(s.stddev.is_none());
    }

    #[test]
    fn aggregate_empty_set_is_none() {
        // Two reps, neither carrying a `tg` ⇒ the aggregate is None.
        let samples = vec![
            RepSample {
                deltas: 1,
                ttft_s: Some(0.1),
                prompt_tps: Some(10.0),
                tg: None,
                tg_first: None,
                tg_last: None,
            },
            RepSample {
                deltas: 1,
                ttft_s: Some(0.2),
                prompt_tps: Some(5.0),
                tg: None,
                tg_first: None,
                tg_last: None,
            },
        ];
        assert!(aggregate(&samples, |s| s.tg).is_none());
        // ttft IS present on both ⇒ Some.
        assert!(aggregate(&samples, |s| s.ttft_s).is_some());
    }

    #[test]
    fn rep_metrics_one_delta_only_ttft() {
        // n_prompt=10, single delta at t=0.5 ⇒ ttft=0.5, prompt_tps=20, no tg/segments.
        let r = rep_metrics(&[0.5], 10);
        assert_eq!(r.deltas, 1);
        approx(r.ttft_s.unwrap(), 0.5);
        approx(r.prompt_tps.unwrap(), 20.0);
        assert!(r.tg.is_none());
        assert!(r.tg_first.is_none());
        assert!(r.tg_last.is_none());
    }

    #[test]
    fn rep_metrics_two_deltas_tg_no_segments() {
        // 2 deltas: tg = (2-1)/(t_last-t_first) = 1/1.0 = 1.0; segments need ≥4.
        let r = rep_metrics(&[1.0, 2.0], 4);
        approx(r.tg.unwrap(), 1.0);
        assert!(r.tg_first.is_none());
        assert!(r.tg_last.is_none());
    }

    #[test]
    fn rep_metrics_segments_split() {
        // 4 deltas at 1,2,4,8. d=4, m=2.
        //   tg       = (4-1)/(8-1)       = 3/7
        //   tg_first = (m-1)/(t_2 - t_1) = (1)/(2-1)   = 1.0
        //   tg_last  = (d-m-1)/(t_4-t_3) = (1)/(8-4)   = 0.25
        let r = rep_metrics(&[1.0, 2.0, 4.0, 8.0], 100);
        approx(r.tg.unwrap(), 3.0 / 7.0);
        approx(r.tg_first.unwrap(), 1.0);
        approx(r.tg_last.unwrap(), 0.25);
    }

    #[test]
    fn rep_metrics_zero_deltas_is_empty() {
        let r = rep_metrics(&[], 10);
        assert_eq!(r.deltas, 0);
        assert!(r.ttft_s.is_none());
        assert!(r.prompt_tps.is_none());
        assert!(r.tg.is_none());
    }

    #[test]
    fn rate_rejects_degenerate_duration() {
        assert!(rate(3.0, 0.0).is_none());
        assert!(rate(3.0, f64::NAN).is_none());
        approx(rate(3.0, 1.5).unwrap(), 2.0);
    }
}
