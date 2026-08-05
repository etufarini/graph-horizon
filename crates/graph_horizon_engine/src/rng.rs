/*
 * graph_horizon_engine — deterministic sampling PRNG
 * Minimal xorshift64* generator seeded per request, used by the stochastic
 * sampling path so a run is reproducible for a given seed. No external dependency.
 * Kept private to the crate and re-exported as `sampling::Rng`, the only path
 * callers use.
*/

// Minimal deterministic RNG (xorshift64*). Seeded per request so the stochastic
// path is reproducible; unused on the greedy path.
pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Rng {
        // Force a non-zero state (xorshift is stuck at 0 otherwise).
        Rng {
            state: (seed ^ 0x9E37_79B9_7F4A_7C15) | 1,
        }
    }

    // Public: the passkey gate derives its 5-digit keys from raw draws
    // (D29: `10000 + next_u64() % 90000`).
    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    // Uniform in [0, 1) using the top 24 bits (exact in f32).
    pub(crate) fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}
