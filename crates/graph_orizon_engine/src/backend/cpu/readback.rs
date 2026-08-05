/*
 * graph_orizon_engine — CPU backend host read-back
 * Implements bounded FP32 logits read-back plus the two host reductions used by
 * sampling. `read_argmax` and `read_topk` share the ordering contract in
 * `sampling.rs`: logit descending, then token index ascending.
*/

use color_eyre::eyre::{Result, bail};

use super::buffer::CpuBuffer;

// Reads the FP32 logits to host. Errors (never reads out of bounds) if the buffer
// is smaller than the requested vocab.
pub(super) fn read_logits(logits: &CpuBuffer, vocab: usize) -> Result<Vec<f32>> {
    if logits.byte_len() < vocab * 4 {
        bail!("cpu: logits buffer smaller than requested vocab");
    }
    let mut v = logits.read_f32();
    v.truncate(vocab);
    Ok(v)
}

// Argmax over the host-resident FP32 logits, same tiebreak as `sampling.rs`
// (strict `>`, so the lowest index wins on ties). No reduction, no transfer.
pub(super) fn read_argmax(logits: &CpuBuffer, vocab: usize) -> Result<u32> {
    if vocab == 0 {
        bail!("cpu: read_argmax on empty logits");
    }
    Ok(argmax_f32(&read_logits(logits, vocab)?))
}

// Top-k over the host-resident FP32 logits under the total order
// `(logit desc, index asc)`, returning `min(k, vocab)` `(index, raw logit)` pairs in
// that order. `k > vocab` clamps; identical comparator to `sampling.rs`.
pub(super) fn read_topk(logits: &CpuBuffer, vocab: usize, k: usize) -> Result<Vec<(u32, f32)>> {
    if vocab == 0 {
        bail!("cpu: read_topk on empty logits");
    }
    Ok(topk_f32(&read_logits(logits, vocab)?, k))
}

// Argmax with `sampling.rs`'s tiebreak: strict `>`, so the lowest index wins on
// equal values. Free function so the reduction logic is unit-tested without a
// fully-loaded backend.
fn argmax_f32(l: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in l.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best as u32
}

// Top-k under the total order `(logit desc, index asc)`, returning `min(k, len)`
// `(index, raw logit)` pairs in that order. Same comparator as `sampling.rs`.
fn topk_f32(l: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut cand: Vec<(u32, f32)> = l.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
    let order = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    let k = k.min(cand.len());
    if k < cand.len() {
        cand.select_nth_unstable_by(k - 1, order);
        cand.truncate(k);
    }
    cand.sort_unstable_by(order);
    cand
}

#[cfg(test)]
mod tests {
    use super::{argmax_f32, topk_f32};

    // argmax matches the scalar reference and breaks ties on the lowest index.
    #[test]
    fn argmax_tiebreak_lowest_index() {
        assert_eq!(argmax_f32(&[0.1, 2.5, -1.0, 2.4]), 1);
        // Two equal maxima: the lowest index wins (strict `>`).
        assert_eq!(argmax_f32(&[3.0, 1.0, 3.0]), 0);
    }

    // top-k returns exactly k pairs under (logit desc, index asc), in order, with
    // the raw logits as values; ties at the border are broken by ascending index.
    #[test]
    fn topk_order_and_tiebreak() {
        let l = [0.1f32, 2.5, 2.5, -1.0, 2.4];
        let top = topk_f32(&l, 3);
        assert_eq!(top, vec![(1, 2.5), (2, 2.5), (4, 2.4)]);
        // k > len clamps to the whole, fully ordered vocabulary.
        let all = topk_f32(&l, 99);
        assert_eq!(all.len(), 5);
        assert_eq!(all[0], (1, 2.5));
        assert_eq!(all[4], (3, -1.0));
    }
}
