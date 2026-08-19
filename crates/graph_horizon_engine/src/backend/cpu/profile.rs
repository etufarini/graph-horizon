/*
 * graph_horizon_engine — opt-in CPU operation profiler
 * Aggregates elapsed wall time and call counts at the CPU backend boundary for
 * one process. It exists only under `cpu-profile`, prints once when the backend
 * drops, and never participates in normal CPU or any GPU/backend build.
 */

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Default)]
struct Entry {
    calls: u64,
    elapsed: Duration,
}

static ENTRIES: OnceLock<Mutex<BTreeMap<String, Entry>>> = OnceLock::new();

pub(super) fn measure<T>(label: impl Into<String>, operation: impl FnOnce() -> T) -> T {
    let label = label.into();
    let start = Instant::now();
    let result = operation();
    let mut entries = ENTRIES
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .expect("CPU profile lock poisoned");
    let entry = entries.entry(label).or_default();
    entry.calls += 1;
    entry.elapsed += start.elapsed();
    result
}

pub(super) fn report() {
    let Some(entries) = ENTRIES.get() else {
        return;
    };
    let entries = entries.lock().expect("CPU profile lock poisoned");
    let total = entries
        .values()
        .map(|entry| entry.elapsed)
        .sum::<Duration>();
    eprintln!("cpu_profile total_ms={:.3}", total.as_secs_f64() * 1000.0);
    for (label, entry) in entries.iter() {
        let milliseconds = entry.elapsed.as_secs_f64() * 1000.0;
        let fraction = if total.is_zero() {
            0.0
        } else {
            entry.elapsed.as_secs_f64() / total.as_secs_f64()
        };
        eprintln!(
            "cpu_profile operation={label} calls={} total_ms={milliseconds:.3} fraction={fraction:.6}",
            entry.calls
        );
    }
}
