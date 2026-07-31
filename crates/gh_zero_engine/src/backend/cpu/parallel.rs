/*
 * gh_zero_engine — CPU kernel parallelism
 * The single place in the CPU module where `available_parallelism` and the
 * worker count live: the compute-bound kernels never touch threads directly,
 * they only call `for_units`. The count is resolved once (cached in a OnceLock),
 * seeded by `Engine::new` from `cfg.cpu_threads` (CLI flag, no env). `for_units`
 * splits the output into disjoint contiguous chunks and runs them on the
 * persistent worker pool (`pool`), which spawns its workers once and reuses them
 * across the ~150 matmuls/token instead of re-spawning per call. Each chunk is a
 * non-overlapping sub-slice, so workers never alias; the per-chunk `&mut` is
 * rebuilt from a shared base pointer (a small, documented `unsafe`, sound because
 * the slots partition the output and the caller blocks until every slot finishes).
 * The cardinal rule still holds — "parallelize only over independent outputs,
 * never over the reduction dimension": splitting output rows/heads across threads
 * does not touch the fixed f32 accumulation order, so results stay bit-identical
 * to the single-thread path.
*/

use std::sync::OnceLock;

// Worker count, resolved exactly once and cached. Seeded by `Engine::new` via
// `set_threads(cfg.cpu_threads)`; when unseeded (no flag, or a module used outside
// `Engine::new` such as examples/tests) it falls back to `available_parallelism()`,
// else `1`. Always `>= 1`.
static THREADS: OnceLock<usize> = OnceLock::new();

// Seeds the worker count from the CLI (`cfg.cpu_threads`). A `Some(n)` with `n > 0`
// fixes the count; `None` (no flag) or a degenerate value leaves it unseeded so
// `threads()` resolves to the system parallelism — today's behaviour. Set-once:
// `Engine::new` runs twice (model + embed), so the second call is a no-op and never
// panics. Must be called BEFORE the first `threads()`/`pool::global()` use, which
// `Engine::new` guarantees (it seeds before the load).
pub(crate) fn set_threads(n: Option<usize>) {
    if let Some(n) = n
        && n > 0
    {
        let _ = THREADS.set(n);
    }
}

// Logical cores, not physical: with the per-call thread spawn removed (the
// persistent `pool`), the SMT siblings are worth using — on an 8c/16t host decode
// tg keeps climbing past 8 workers (8→10.3, 14→13.0, 16→12.7 tok/s), so the old
// physical-core cap (which made sense only while spawn overhead dominated) now
// leaves ~18% on the table. Bit-exact: the worker count only changes how the
// disjoint output rows are chunked, never the per-row accumulation order.
pub(crate) fn threads() -> usize {
    *THREADS.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    })
}

// Splits `out` into contiguous chunks aligned to `stride` (elements per
// independent unit: `1` for matmul/logits rows, `head_dim` for attention heads)
// and runs `f(u0, sub)` on each, where `u0` is the absolute index of the chunk's
// first unit and `sub` is its output slice. The chunks come from `chunks_mut`, so
// they are pairwise disjoint — each worker writes only its own slice, no data
// race, no lock. If at most one chunk is needed (single unit, empty output, or
// one worker) the closure runs inline with no thread spawned, identical to the
// old single-thread path. `f` is `Sync` because several threads call it at once.
pub(crate) fn for_units<T, F>(out: &mut [T], stride: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let units = out.len() / stride;
    let n = threads().min(units);
    if n <= 1 {
        // Inline degradation: no scope, no spawn — bit-for-bit the prior behavior.
        f(0, out);
        return;
    }
    // Units per chunk (last chunk shorter); every full chunk holds exactly
    // `chunk_units` units, so chunk `ci` starts at unit `ci * chunk_units`.
    let chunk_units = units.div_ceil(n);
    let chunk_elems = chunk_units * stride;
    // The number of *non-empty* chunks can be smaller than `n` (e.g. 100 units
    // over 16 workers → 7 units/chunk → 15 chunks), matching `chunks_mut`. Dispatch
    // exactly that many slots so no slot ever indexes past the output.
    let nslots = units.div_ceil(chunk_units);
    // Dispatch the `n` disjoint chunks to the persistent worker pool (slot 0 runs
    // on the caller). Reusing the workers avoids the per-call thread spawn+join —
    // decode issues ~150 matmuls/token. Bit-exact: same disjoint chunks, same
    // per-row order; only which thread runs a chunk changes. Each slot rebuilds
    // its own non-overlapping `&mut` sub-slice from the base pointer, so no two
    // workers ever alias (identical disjointness to the old `chunks_mut`).
    // The base pointer is shared (read-only as a value) across the worker slots;
    // each slot derives a disjoint sub-slice, so sharing it is sound. Raw pointers
    // are `!Sync`, so wrap it to cross the dispatch boundary.
    struct Base<T>(*mut T);
    // SAFETY: every slot indexes a disjoint range, no two slots alias.
    unsafe impl<T> Sync for Base<T> {}
    let base = Base(out.as_mut_ptr());
    let base = &base; // capture the whole Sync wrapper, not the bare `*mut f32` field
    let len = out.len();
    super::pool::global().dispatch(nslots, |slot| {
        let start = slot * chunk_elems;
        let end = (start + chunk_elems).min(len);
        // SAFETY: slots partition 0..len into disjoint [start,end) ranges, so this
        // is the only live `&mut` over this sub-slice for the dispatch's duration.
        let chunk = unsafe { std::slice::from_raw_parts_mut(base.0.add(start), end - start) };
        f(slot * chunk_units, chunk);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fills every unit with a value derived from its absolute index; the result
    // must match the serial expectation regardless of how `out` was chunked.
    fn fill_by_index(len: usize, stride: usize) -> Vec<f32> {
        let mut out = vec![0f32; len * stride];
        for_units(&mut out, stride, |u0, chunk| {
            for j in 0..chunk.len() / stride {
                let u = u0 + j;
                for d in 0..stride {
                    chunk[j * stride + d] = (u * 10 + d) as f32;
                }
            }
        });
        out
    }

    fn expected(len: usize, stride: usize) -> Vec<f32> {
        let mut v = vec![0f32; len * stride];
        for u in 0..len {
            for d in 0..stride {
                v[u * stride + d] = (u * 10 + d) as f32;
            }
        }
        v
    }

    #[test]
    fn for_units_covers_every_unit_disjointly() {
        // stride 1 (matmul rows) and stride 3 (attention-like heads), several
        // lengths including more units than workers and the single-unit case.
        for &stride in &[1usize, 3] {
            for &len in &[0usize, 1, 2, 7, 100] {
                assert_eq!(fill_by_index(len, stride), expected(len, stride));
            }
        }
    }

    #[test]
    fn threads_is_at_least_one() {
        assert!(threads() >= 1);
    }
}
