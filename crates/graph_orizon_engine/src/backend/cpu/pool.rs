/*
 * graph_orizon_engine — persistent fork-join worker pool
 * Decode issues ~150 matmuls per token; the previous design spawned a fresh set
 * of OS threads (`std::thread::scope`) on every one, paying spawn+join latency
 * each call. This pool spawns the workers once (lazily, on first parallel call)
 * and reuses them: each `dispatch` publishes one job, wakes the workers via a
 * condvar (no busy-spin — workers block when idle, so an idle process burns no
 * CPU), runs slot 0 on the caller, then blocks until every worker slot reports
 * done. Workers live for the process lifetime as daemon threads.
 *
 * Correctness: a job is split into `n` *disjoint* output slots (built by the
 * caller, `parallel::for_units`); slot `s` is one contiguous chunk, so the raw
 * `&mut` reconstructions never alias — the same invariant the scoped version
 * relied on. The published job is a borrowed trait object whose lifetime is
 * erased to `'static`; this is sound only because the caller blocks on
 * `remaining == 0` before returning, so the borrow strictly outlives every
 * worker's use of it. A worker closure panic is caught and re-raised on the
 * caller after the join, mirroring the old scope-join panic propagation (no
 * partial result is consumed: the kernel's final write happens after dispatch
 * returns). Numerics are untouched — this only changes *which* thread runs a
 * slot, not the per-row accumulation order, so output stays bit-identical.
*/

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, Once, OnceLock};

// Type-erased borrowed job. `run` points at a `&(dyn Fn(usize) + Sync)` whose
// real lifetime is the `dispatch` call; we hold it only while the caller is
// blocked on the job, so the erased lifetime never outlives the borrow.
#[derive(Clone, Copy)]
struct Job {
    run: *const (dyn Fn(usize) + Sync + 'static),
    n: usize, // number of active slots; worker `id` runs only if `id < n`.
}
// SAFETY: `run` is `Sync` behind the pointer and is only dereferenced while the
// caller holds the job alive (blocked until all workers finish).
unsafe impl Send for Job {}

struct Shared {
    job: Option<Job>,
    generation: u64,  // bumped per dispatch so workers detect a new job
    remaining: usize, // worker slots (1..n) not yet finished this dispatch
    panicked: AtomicBool,
}

pub(super) struct Pool {
    workers: usize,           // total participants incl. the caller (== threads())
    dispatch_lock: Mutex<()>, // serializes concurrent callers (one job at a time)
    state: Mutex<Shared>,
    wake: Condvar, // workers wait here for the next generation
    done: Condvar, // caller waits here for remaining == 0
}

static POOL: OnceLock<Pool> = OnceLock::new();

// The shared pool, sized to the resolved worker count. Built once and stored so
// its address is stable, then `workers - 1` daemon threads are spawned exactly
// once against that `'static` reference (the caller is the workers-th participant).
pub(super) fn global() -> &'static Pool {
    let pool = POOL.get_or_init(|| Pool {
        workers: super::parallel::threads(),
        dispatch_lock: Mutex::new(()),
        state: Mutex::new(Shared {
            job: None,
            generation: 0,
            remaining: 0,
            panicked: AtomicBool::new(false),
        }),
        wake: Condvar::new(),
        done: Condvar::new(),
    });
    static SPAWNED: Once = Once::new();
    SPAWNED.call_once(|| {
        for id in 1..pool.workers {
            std::thread::Builder::new()
                .name(format!("graph-orizon-cpu-{id}"))
                .spawn(move || worker_loop(pool, id))
                .expect("spawn cpu worker");
        }
    });
    pool
}

// A worker blocks until a generation it has not seen, runs its slot if active,
// decrements `remaining`, and notifies the caller when it drains to zero.
fn worker_loop(pool: &'static Pool, id: usize) {
    let mut seen = 0u64;
    loop {
        let job = {
            let mut s = pool.state.lock().unwrap();
            while s.generation == seen {
                s = pool.wake.wait(s).unwrap();
            }
            seen = s.generation;
            s.job
        };
        let Some(job) = job else { continue };
        if id < job.n {
            // SAFETY: the caller keeps the borrowed closure alive until it sees
            // remaining == 0, which happens strictly after this call returns.
            let f = unsafe { &*job.run };
            if catch_unwind(AssertUnwindSafe(|| f(id))).is_err() {
                pool.state
                    .lock()
                    .unwrap()
                    .panicked
                    .store(true, Ordering::Relaxed);
            }
            let mut s = pool.state.lock().unwrap();
            s.remaining -= 1;
            if s.remaining == 0 {
                pool.done.notify_one();
            }
        }
    }
}

impl Pool {
    // Runs `f(slot)` for slot in 0..n: slot 0 inline on the caller, slots 1..n on
    // the persistent workers. Blocks until all complete. `f` must drive disjoint
    // outputs per slot (guaranteed by the only caller, `for_units`).
    pub(super) fn dispatch<F: Fn(usize) + Sync>(&self, n: usize, f: F) {
        // One job at a time: the shared slot holds a single job, so concurrent
        // callers (e.g. the test harness running cases in parallel) must serialize.
        // The engine's forward is single-threaded, so this never contends in the
        // hot path; it only guarantees correctness if a caller ever overlaps.
        let _serial = self.dispatch_lock.lock().unwrap();
        // Publish the borrowed job with its lifetime erased to 'static (sound: we
        // block below until every worker is done with it).
        let erased: *const (dyn Fn(usize) + Sync) = &f;
        let erased: *const (dyn Fn(usize) + Sync + 'static) =
            unsafe { std::mem::transmute(erased) };
        {
            let mut s = self.state.lock().unwrap();
            s.panicked.store(false, Ordering::Relaxed);
            s.remaining = n - 1; // slots 1..n on workers; slot 0 is the caller
            s.job = Some(Job { run: erased, n });
            s.generation += 1;
        }
        self.wake.notify_all();
        // Caller runs slot 0 itself, then waits for the workers.
        let caller_panic = catch_unwind(AssertUnwindSafe(|| f(0))).is_err();
        {
            let mut s = self.state.lock().unwrap();
            while s.remaining != 0 {
                s = self.done.wait(s).unwrap();
            }
            s.job = None;
            if s.panicked.load(Ordering::Relaxed) || caller_panic {
                panic!("graph-orizon cpu worker panicked");
            }
        }
    }
}
