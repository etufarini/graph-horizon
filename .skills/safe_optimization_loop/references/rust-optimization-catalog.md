<!--
This catalog classifies Rust candidates by risk and required gate; it neither
authorizes changes nor assigns performance results.
-->

# Rust Optimization Catalog (inference hot paths)

Candidate optimizations for prefill and decode, sorted by correctness risk.
Always profile first (catalog ≠ permission to guess). Exact candidates require
an exact gate; numeric candidates require a predeclared bounded gate that
matches their risk.

The decode loop runs once per generated token and is latency-sensitive, so per-iteration overhead dominates there. Prefill runs once over the prompt and is usually compute/throughput-bound, so it rewards better arithmetic intensity and parallelism. Match the optimization to the path you profiled.

The retained public harness separates TTFT, prompt throughput, decode
throughput, memory, and placement. Backend-internal profilers are intentionally
not permanent production interfaces; add focused temporary attribution only
when the public measurements cannot rank the next candidate.

---

## Tier 1 — Bit-exact-preserving (default-safe, proceed)

These do not reorder or re-round floating-point arithmetic, so they should keep token IDs identical. Still gate every one — a logic bug can break bit-exactness even in a "safe" change.

**Hot-path allocation removal (decode).** Per-token `Vec`/`String`/`Box`
allocations are a classic decode cost. Pre-allocate existing buffers and reuse
them with `Vec::clear()` where ownership stays obvious. Do not add a dependency
for a hypothetical allocation win.

**Avoid copies and redundant clones.** Replace `.clone()` with borrows where lifetimes allow; pass slices instead of owned buffers; use `Cow` where appropriate. Each avoided memcpy is free latency.

**Hoist loop-invariants out of the decode loop.** Anything recomputed every token that depends only on the model/config (shapes, strides, scale factors, masks) should be computed once before the loop.

**Memory access / contiguity.** Ensure existing hot tensors are accessed with
unit stride and group already adjacent fields when representation and arithmetic
remain unchanged. A KV representation or layout change moves to Tier 2 because
it needs byte-layout, capacity, and quality gates.

**Cache blocking / tiling that preserves accumulation order.** Tiling a matmul or attention pass for cache locality is bit-exact *only if the summation order within each dot product is unchanged*. If your tiling changes the order of additions in a reduction, it moves to Tier 2.

**Reduce synchronization and contention.** Cut lock scope, replace a `Mutex` on the hot path with per-thread state merged deterministically, avoid atomics where a local suffices. Removing a lock doesn't change results; just keep any reduction order deterministic.

**Bounds-check elimination (carefully).** Restructure indexing so the compiler proves bounds (iterators, `chunks_exact`, slicing once up front) to drop bounds checks. Verify via `cargo asm` and the test suite. Avoid `unsafe`/`get_unchecked` unless profiling proves it matters and the invariant is airtight — and then gate hard.

**Cut per-token overhead in sampler/tokenizer/plumbing.** Greedy argmax can be a single pass; avoid sorting the whole vocab when you only need the max; cache tokenizer lookups; avoid per-token logging/formatting.

**Thread pinning / affinity.** Pinning worker threads can cut latency variance without touching math. Keep the thread *count* fixed (it can affect reduction order — see Tier 2).

**Build profile tuning.** This repo ships a `fast` profile
(`inherits = "release"`, `lto = true`, `codegen-units = 1`), used by
`support/install.sh --profile fast`; that's the optimized build to baseline on.
It does not add `target-cpu=native`, target features, or fast-math. Re-capture
nothing, but re-baseline timings.

---

## Tier 2 — Numerics-changing (requires an approved bounded gate)

These can change rounding even when "mathematically equivalent", so they cannot
claim bit identity. Retain them only when the explicit performance scope allows
the change and a predeclared numeric, quality, layout, and lifecycle gate covers
its actual risk.

**SIMD horizontal reductions.** Hand-vectorized or autovectorized dot products/sums change the order of floating-point additions vs a scalar left-to-right sum → different bits. Vectorization that preserves the exact reduction order (or that only vectorizes bit-exact integer/elementwise ops) can be Tier 1; a reordered float reduction is Tier 2.

**Rayon / multi-thread parallel reductions.** Splitting a reduction across threads and combining changes summation order and is often nondeterministic across thread counts. Strongly numerics-affecting under bit-exact.

**FMA contraction.** `a * b + c` fused into one `mul_add` rounds once instead of twice → different result. Separately supplied `target-cpu=native` or `target-feature=+fma` flags can change available instructions; the repository's `fast` profile does not supply them. Fix `RUSTFLAGS` before capturing records and never vary it within a loop run.

**Swapping GEMM / BLAS backend** (e.g., to a faster matmul lib, or different kernel). Different accumulation, blocking, and FMA usage → different bits. High value, but a numerics change.

**Lower precision / mixed precision** (f32→bf16/f16 anywhere it wasn't already). Changes results by construction.

**Quantization** (weights/KV cache to int8/int4/fp8). Largest wins, largest numerics change. It requires explicit scope plus quality, layout, memory, and lifecycle gates; it is never classified as bit-exact.

**Fast-math-style flags.** Rust doesn't expose `-ffast-math` broadly (good), but any feature that relaxes IEEE semantics belongs here.

---

## How to use this with the loop

1. Profile → find the bottleneck and which path (prefill vs decode).
2. Pick the **lowest-tier** candidate that targets it. Exhaust Tier 1 before considering Tier 2.
3. For Tier 1: implement, run the selected exact gate, then measure performance
   and keep or revert.
4. For Tier 2: declare the bounded gate before implementation, run correctness
   before performance, then keep or revert from evidence. Without an adequate
   approved gate, remove the candidate from the tree.

A good default ordering for inference engines under bit-exact: cut the per-token decode round-trip / readback overhead → remove decode-loop allocations and copies → fix KV-cache/memory layout → hoist invariants → cut sampler/tokenizer overhead → reduce contention → bounds-check/codegen tuning → (only then, with sign-off) the numerics-changing matmul/SIMD/quant work.

Reminder on gates: `parity-check.sh` proves exact prompt IDs and a bounded
teacher-forced top-two criterion; it does not by itself prove bit identity.
Choose an additional exact gate for a Tier 1 promise, and the risk-specific
bounded gates for Tier 2.
