/*
 * graph_horizon_engine — token sampling
 * Picks the next token from raw logits and owns the small deterministic PRNG
 * used by stochastic sampling. Filters are applied in a fixed, documented order:
 *   repeat penalty → temperature → top-k → top-p (nucleus) → min-p → sample.
 * With temperature 0 the function short-circuits to a deterministic argmax; with a temperature set
 * the choice is deterministic for a given seed (small in-house xorshift RNG, no
 * external dependency).
*/

use crate::api::request::SamplingParams;

// Minimal xorshift64* state. A request owns one instance, so stochastic runs are
// reproducible for the same seed and greedy runs pay no random-number cost.
pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Rng {
        // Xorshift would remain stuck at zero, so every seed maps to a non-zero state.
        Rng {
            state: (seed ^ 0x9E37_79B9_7F4A_7C15) | 1,
        }
    }

    // Uniform in [0, 1) using the top 24 bits, which are exact in f32.
    pub(crate) fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32
    }
}

// Returns the chosen token id. `logits` is mutated in place (penalty/temperature
// are applied to it). `recent` are the token ids the repeat penalty acts on.
pub(crate) fn sample(logits: &mut [f32], p: &SamplingParams, recent: &[u32], rng: &mut Rng) -> u32 {
    // 1) Repeat penalty: divide positive logits, multiply negative ones, so a
    //    recently seen token is always pushed toward less likely.
    if p.repeat_penalty != 1.0 {
        for &t in recent {
            if let Some(l) = logits.get_mut(t as usize) {
                *l = if *l > 0.0 {
                    *l / p.repeat_penalty
                } else {
                    *l * p.repeat_penalty
                };
            }
        }
    }

    // 2) Greedy gate: temperature 0 ⇒ exact argmax (deterministic validation path).
    if p.temperature <= 0.0 {
        return argmax(logits);
    }

    // 3) Temperature, kept on a candidate list of finite logits only.
    let inv = 1.0 / p.temperature;
    let mut cand: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .filter(|&(_, &l)| l.is_finite())
        .map(|(i, &l)| (i as u32, l * inv))
        .collect();
    if cand.is_empty() {
        // Everything was filtered (all -inf): fall back to the raw maximum.
        return argmax(logits);
    }
    // Total order: logit descending, ties broken by ascending vocab index. Indices
    // are unique so no two candidates compare equal; the order is total. This both
    // reproduces the prior tiebreak (chosen token unchanged) and makes sort
    // stability irrelevant, so the sorts below use the faster `sort_unstable_by`
    // (no temp allocation) with an identical result.
    let order = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));

    // 4) Top-k: keep the k highest logits. When k is set we only need a partial
    //    selection (O(n)) plus a sort of the k survivors — a full sort over the
    //    whole vocabulary (O(n log n)) would be wasted every token.
    if p.top_k > 0 && (p.top_k as usize) < cand.len() {
        let k = p.top_k as usize;
        cand.select_nth_unstable_by(k - 1, order);
        cand.truncate(k);
    }
    cand.sort_unstable_by(order);

    // 4b–6) Shared tail: softmax → top-p → min-p → inverse-CDF draw.
    sample_from_candidates(cand, p, rng)
}

// Shared sampling tail (steps 4b–6 of `sample`): softmax over the survivors,
// top-p (nucleus), min-p, then the inverse-CDF draw. `cand` MUST be non-empty and
// already sorted under `(scaled logit desc, index asc)` — the temperature scaling
// and the top-k selection happened before this call (in `sample`, or in
// `forward::sample_step` for the device top-k path). Factored out so the device
// path and the host path draw through identical code (bit-exact by construction).
pub(crate) fn sample_from_candidates(
    cand: Vec<(u32, f32)>,
    p: &SamplingParams,
    rng: &mut Rng,
) -> u32 {
    // Unreachable for the real callers (they pass a non-empty list); guards the
    // `cand[0]` access rather than panicking if ever called empty.
    if cand.is_empty() {
        return 0;
    }
    let mut cand = cand;

    // Softmax over the survivors (stable: subtract the max).
    let max = cand[0].1;
    let mut probs: Vec<f32> = cand.iter().map(|&(_, l)| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for pr in probs.iter_mut() {
        *pr /= sum;
    }

    // 5) Top-p (nucleus): smallest prefix whose cumulative mass reaches top_p.
    if p.top_p < 1.0 {
        let mut cum = 0.0;
        let mut cut = probs.len();
        for (i, &pr) in probs.iter().enumerate() {
            cum += pr;
            if cum >= p.top_p {
                cut = i + 1;
                break;
            }
        }
        cand.truncate(cut);
        probs.truncate(cut);
    }

    // 6) Min-p: drop tokens below min_p * peak. probs is sorted desc, so the kept
    //    set is the leading run (at least the top token survives).
    if p.min_p > 0.0 {
        let thr = p.min_p * probs[0];
        let keep = probs.iter().take_while(|&&pr| pr >= thr).count().max(1);
        cand.truncate(keep);
        probs.truncate(keep);
    }

    // Renormalize the survivors and draw one by inverse-CDF.
    let sum: f32 = probs.iter().sum();
    let r = rng.next_f32() * sum;
    let mut acc = 0.0;
    for (i, &pr) in probs.iter().enumerate() {
        acc += pr;
        if r < acc {
            return cand[i].0;
        }
    }
    cand[cand.len() - 1].0
}

// Upper bound on k for the device top-k path. Above it, sampling falls back to the
// full host readback (still bit-exact). MUST equal `reduce::MAX_K` (Rust) and the
// `MAXK` constant of `topk_partial.comp`. 128 covers realistic `top_k` values
// while keeping the shader/merge small.
pub(crate) const MAX_DEVICE_K: usize = 128;

// Which sampling path the decode hot-path takes, decided ONCE per request from the
// (constant) `SamplingParams`. The device paths read only a minimal reduction of
// the logits; `Fallback` reads the whole vocab and runs `sample` unchanged.
#[derive(Clone, Copy)]
pub(crate) enum SamplePath {
    // temperature <= 0 && repeat_penalty == 1: device argmax == greedy `sample`.
    Greedy,
    // temperature > 0 && 0 < top_k <= MAX_DEVICE_K && repeat_penalty == 1: device
    // top-k + the shared tail == `sample` on the full logits.
    TopK(usize),
    // Everything else (repeat penalty active, no/over-cap top-k, nucleus-only):
    // full readback + `sample`.
    Fallback,
}

// Classifies the sampling path. The device paths require `repeat_penalty == 1.0`:
// with no penalty `sample` skips the `recent` loop entirely, so the device path
// (which never reads `recent`) is bit-identical.
pub(crate) fn plan(p: &SamplingParams) -> SamplePath {
    if p.repeat_penalty != 1.0 {
        return SamplePath::Fallback;
    }
    if p.temperature <= 0.0 {
        return SamplePath::Greedy;
    }
    if p.top_k > 0 && (p.top_k as usize) <= MAX_DEVICE_K {
        return SamplePath::TopK(p.top_k as usize);
    }
    SamplePath::Fallback
}

// Exact argmax over the logits (first occurrence on ties).
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l > best_val {
            best_val = l;
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn greedy() -> SamplingParams {
        SamplingParams::greedy()
    }

    #[test]
    fn temperature_zero_is_exact_argmax() {
        let mut logits = [0.1, 2.5, -1.0, 2.4];
        let mut rng = Rng::new(0);
        assert_eq!(sample(&mut logits, &greedy(), &[], &mut rng), 1);
    }

    #[test]
    fn sampling_is_deterministic_for_a_seed() {
        let p = SamplingParams {
            temperature: 1.0,
            ..greedy()
        };
        let draw = |seed| {
            let mut l = [1.0f32, 2.0, 3.0, 0.5];
            sample(&mut l, &p, &[], &mut Rng::new(seed))
        };
        // Same seed ⇒ same token; the run is reproducible.
        assert_eq!(draw(42), draw(42));
    }

    #[test]
    fn top_k_one_collapses_to_argmax() {
        let p = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            ..greedy()
        };
        let mut l = [0.0f32, 5.0, 1.0];
        assert_eq!(sample(&mut l, &p, &[], &mut Rng::new(7)), 1);
    }

    // `sample` must equal building the candidate list and
    // calling the shared tail `sample_from_candidates` on it, for the same seed.
    #[test]
    fn sample_equals_shared_tail_on_same_candidates() {
        let p = SamplingParams {
            temperature: 0.8,
            top_p: 0.95,
            min_p: 0.05,
            ..greedy()
        };
        let logits = [1.0f32, 2.0, 3.0, 0.5, -1.0, 4.0, 2.5];
        let seed = 123u64;

        let mut l = logits;
        let via_sample = sample(&mut l, &p, &[], &mut Rng::new(seed));

        // Same candidate construction `sample` does on the repeat_penalty==1 path:
        // finite logits scaled by 1/temperature, sorted under (logit desc, idx asc).
        let inv = 1.0 / p.temperature;
        let mut cand: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v.is_finite())
            .map(|(i, &v)| (i as u32, v * inv))
            .collect();
        cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let via_tail = sample_from_candidates(cand, &p, &mut Rng::new(seed));

        assert_eq!(via_sample, via_tail);
    }

    // `plan` routes greedy / in-range top-k to the device, everything else to the
    // full readback.
    #[test]
    fn plan_routes_paths() {
        assert!(matches!(plan(&greedy()), SamplePath::Greedy));
        let tk = SamplingParams {
            temperature: 1.0,
            top_k: 40,
            ..greedy()
        };
        assert!(matches!(plan(&tk), SamplePath::TopK(40)));
        // Over the cap → fallback.
        let big = SamplingParams {
            temperature: 1.0,
            top_k: (MAX_DEVICE_K + 1) as u32,
            ..greedy()
        };
        assert!(matches!(plan(&big), SamplePath::Fallback));
        // Repeat penalty active → fallback even when greedy-like.
        let pen = SamplingParams {
            repeat_penalty: 1.1,
            ..greedy()
        };
        assert!(matches!(plan(&pen), SamplePath::Fallback));
        // Temperature>0 but no top-k (nucleus only) → fallback.
        let nuc = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            ..greedy()
        };
        assert!(matches!(plan(&nuc), SamplePath::Fallback));
    }
}
