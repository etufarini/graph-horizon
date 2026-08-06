# Rust Optimization Catalog (inference hot paths)

Candidate optimizations for prefill and decode, **sorted by risk under a bit-exact policy**. Always profile first (catalog ≠ permission to guess). Each entry notes whether it preserves bit-exact output.

The decode loop runs once per generated token and is latency-sensitive, so per-iteration overhead dominates there. Prefill runs once over the prompt and is usually compute/throughput-bound, so it rewards better arithmetic intensity and parallelism. Match the optimization to the path you profiled.

In this engine the tracers attribute time to fixed axes — prefill: Projections / SsmScan / Attn / Transfers / CpuGlue; decode: the same GPU-compute axes plus **`ReadbackSampling`** (per-token host readback of logits + host sampling). The standing decode suspect is the **per-token D2H round-trip** (one `submit_wait` + readback per generated token), so decode candidates that cut or amortise that readback are high-value (see the perf-gap notes). Pick the candidate that targets the axis your trace flagged.

---

## Tier 1 — Bit-exact-preserving (default-safe, proceed)

These do not reorder or re-round floating-point arithmetic, so they should keep token IDs identical. Still gate every one — a logic bug can break bit-exactness even in a "safe" change.

**Hot-path allocation removal (decode).** Per-token `Vec`/`String`/`Box` allocations are a classic decode killer. Pre-allocate buffers once and reuse them across tokens; use `Vec::clear()` + reuse instead of re-allocating; consider `SmallVec`/arena for tiny transient buffers. Eliminate `format!`/`to_string` on the hot path.

**Avoid copies and redundant clones.** Replace `.clone()` with borrows where lifetimes allow; pass slices instead of owned buffers; use `Cow` where appropriate. Each avoided memcpy is free latency.

**Hoist loop-invariants out of the decode loop.** Anything recomputed every token that depends only on the model/config (shapes, strides, scale factors, masks) should be computed once before the loop.

**Memory layout / contiguity.** Ensure hot tensors are contiguous and accessed with unit stride; improve KV-cache layout so writes/reads are sequential; group fields touched together (struct-of-arrays vs array-of-structs) to improve cache behavior. Layout changes don't change the math, only access patterns.

**Cache blocking / tiling that preserves accumulation order.** Tiling a matmul or attention pass for cache locality is bit-exact *only if the summation order within each dot product is unchanged*. If your tiling changes the order of additions in a reduction, it moves to Tier 2.

**Reduce synchronization and contention.** Cut lock scope, replace a `Mutex` on the hot path with per-thread state merged deterministically, avoid atomics where a local suffices. Removing a lock doesn't change results; just keep any reduction order deterministic.

**Bounds-check elimination (carefully).** Restructure indexing so the compiler proves bounds (iterators, `chunks_exact`, slicing once up front) to drop bounds checks. Verify via `cargo asm` and the test suite. Avoid `unsafe`/`get_unchecked` unless profiling proves it matters and the invariant is airtight — and then gate hard.

**Cut per-token overhead in sampler/tokenizer/plumbing.** Greedy argmax can be a single pass; avoid sorting the whole vocab when you only need the max; cache tokenizer lookups; avoid per-token logging/formatting.

**Thread pinning / affinity.** Pinning worker threads can cut latency variance without touching math. Keep the thread *count* fixed (it can affect reduction order — see Tier 2).

**Build profile tuning.** This repo already ships a `fast` profile (`inherits = "release"`, `lto = true`, `codegen-units = 1`), used by `support/install.sh --profile fast`; that's the optimized build to baseline on. These don't change FP semantics (unlike fast-math). Re-capture nothing, but re-baseline timings. (`target-cpu=native`, added by `support/install.sh --profile fast`, is Tier-2-adjacent — see the FMA warning below.)

---

## Tier 2 — Numerics-changing (non committabile senza oracle approvato)

These can change rounding even when "mathematically equivalent", so under a
bit-exact policy they will (or may) fail Gate A. Do not commit them in the
current loop: the milestone has no approved tolerant numeric oracle.

**SIMD horizontal reductions.** Hand-vectorized or autovectorized dot products/sums change the order of floating-point additions vs a scalar left-to-right sum → different bits. Vectorization that preserves the exact reduction order (or that only vectorizes bit-exact integer/elementwise ops) can be Tier 1; a reordered float reduction is Tier 2.

**Rayon / multi-thread parallel reductions.** Splitting a reduction across threads and combining changes summation order and is often nondeterministic across thread counts. Strongly numerics-affecting under bit-exact.

**FMA contraction.** `a * b + c` fused into one `mul_add` rounds once instead of twice → different result. This can be introduced implicitly by `target-cpu=native` / `target-feature=+fma` enabling the compiler to contract. This is why `RUSTFLAGS` must be fixed *before* capturing golden outputs and never varied within a loop run.

**Swapping GEMM / BLAS backend** (e.g., to a faster matmul lib, or different kernel). Different accumulation, blocking, and FMA usage → different bits. High value, but a numerics change.

**Lower precision / mixed precision** (f32→bf16/f16 anywhere it wasn't already). Changes results by construction.

**Quantization** (weights/KV cache to int8/int4/fp8). Largest wins, largest numerics change. Always a quality-gated, user-approved change, never part of the bit-exact loop.

**Fast-math-style flags.** Rust doesn't expose `-ffast-math` broadly (good), but any feature that relaxes IEEE semantics belongs here.

---

## How to use this with the loop

1. Profile → find the bottleneck and which path (prefill vs decode).
2. Pick the **lowest-tier** candidate that targets it. Exhaust Tier 1 before considering Tier 2.
3. For Tier 1: implement, gate (Gate A bit-exact, then Gate B perf), keep or revert.
4. For Tier 2: record the proposal and expected gain, then remove it from the
   tree. A separate specification must define the numeric oracle.

A good default ordering for inference engines under bit-exact: cut the per-token decode round-trip / readback overhead → remove decode-loop allocations and copies → fix KV-cache/memory layout → hoist invariants → cut sampler/tokenizer overhead → reduce contention → bounds-check/codegen tuning → (only then, with sign-off) the numerics-changing matmul/SIMD/quant work.

Reminder on gates: a Tier 1 candidate must pass the exact `parity-check.sh`
gate. Tier 2 has no commit gate in the current product and remains unmerged.
