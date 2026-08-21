/*
 * graph_horizon_engine — throughput statistics (harness, pure)
 * The pure aggregation and per-rep segment math behind the throughput bench:
 * mean + sample standard deviation, safe rates, the per-rep metric derivation and
 * the cross-rep aggregation. No model, no I/O — unit tested without a backend. The
 * data types (`Stat`, `RepSample`) live next to the measurement in `throughput`.
*/

use super::throughput::{RepSample, Stat};

// Mean, median, and sample standard deviation of non-empty per-rep values. With a
// single observation σ is undefined ⇒ `None` (n/a), never `0`. Called only on
// non-empty sets (the aggregation gathers `Some` values; the empty case is mapped
// to `None` upstream), but it guards the empty input defensively.
pub(super) fn mean_std(xs: &[f64]) -> Stat {
    let n = xs.len();
    if n == 0 {
        return Stat {
            mean: 0.0,
            median: 0.0,
            stddev: None,
        };
    }
    let mean = xs.iter().sum::<f64>() / n as f64;
    let mut ordered = xs.to_vec();
    ordered.sort_by(f64::total_cmp);
    let median = if n.is_multiple_of(2) {
        (ordered[n / 2 - 1] + ordered[n / 2]) / 2.0
    } else {
        ordered[n / 2]
    };
    let stddev = if n < 2 {
        None // one observation: dispersion not defined
    } else {
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        Some(var.sqrt())
    };
    Stat {
        mean,
        median,
        stddev,
    }
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
// monotonically non-decreasing) and `n_prompt`. Pure, so the segment math is
// unit tested without a model.
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

    let segment = |start: usize, end: usize| {
        (end > start + 1)
            .then(|| rate((end - start - 1) as f64, offsets[end - 1] - offsets[start]))?
    };
    let (tg_begin, tg_middle, tg_end) = if d >= 6 {
        let first = d / 3;
        let second = 2 * d / 3;
        (
            segment(0, first),
            segment(first, second),
            segment(second, d),
        )
    } else {
        (None, None, None)
    };
    let delta_intervals = offsets
        .windows(2)
        .filter_map(|pair| {
            let duration = pair[1] - pair[0];
            (duration.is_finite() && duration > 0.0).then_some(duration)
        })
        .collect();

    RepSample {
        deltas: d,
        ttft_s,
        prompt_tps,
        tg,
        model_tg: None,
        tg_begin,
        tg_middle,
        tg_end,
        delta_intervals,
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
        approx(s.median, 4.0);
        approx(s.stddev.unwrap(), 2.0);
    }

    #[test]
    fn mean_std_even_median_uses_middle_pair() {
        let s = mean_std(&[8.0, 2.0, 4.0, 6.0]);
        approx(s.median, 5.0);
    }

    #[test]
    fn mean_std_single_observation_has_no_stddev() {
        let s = mean_std(&[7.0]);
        approx(s.mean, 7.0);
        approx(s.median, 7.0);
        assert!(s.stddev.is_none());
    }

    #[test]
    fn mean_std_empty_is_defined() {
        let s = mean_std(&[]);
        approx(s.mean, 0.0);
        approx(s.median, 0.0);
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
                model_tg: None,
                tg_begin: None,
                tg_middle: None,
                tg_end: None,
                delta_intervals: Vec::new(),
            },
            RepSample {
                deltas: 1,
                ttft_s: Some(0.2),
                prompt_tps: Some(5.0),
                tg: None,
                model_tg: None,
                tg_begin: None,
                tg_middle: None,
                tg_end: None,
                delta_intervals: Vec::new(),
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
    }

    #[test]
    fn rep_metrics_two_deltas_tg() {
        // 2 deltas: tg = (2-1)/(t_last-t_first) = 1/1.0 = 1.0.
        let r = rep_metrics(&[1.0, 2.0], 4);
        approx(r.tg.unwrap(), 1.0);
    }

    #[test]
    fn rep_metrics_reports_transition_dispersion_and_thirds() {
        let r = rep_metrics(&[1.0, 2.0, 3.0, 5.0, 7.0, 10.0], 100);
        assert_eq!(r.delta_intervals, [1.0, 1.0, 2.0, 2.0, 3.0]);
        approx(r.tg_begin.unwrap(), 1.0);
        approx(r.tg_middle.unwrap(), 0.5);
        approx(r.tg_end.unwrap(), 1.0 / 3.0);
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
