/*
 * gh_zero_engine — model-neutral phase timing driver
 * Times one plain prefill, its separate first greedy selection, and exactly 31
 * subsequent token-plus-argmax steps on a caller-owned runtime session. It owns
 * no model, fixture, backend, aggregation, cancellation, or EOS policy.
 */

use std::time::Instant;

use color_eyre::eyre::Result;

use super::RuntimeSession;

pub(crate) const DECODE_STEPS: usize = 31;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseFixture {
    Short,
    Long,
}

pub(crate) struct PhaseSample {
    pub(crate) prefill_ns: u64,
    pub(crate) first_sample_ns: u64,
    pub(crate) decode_ns: [u64; DECODE_STEPS],
    pub(crate) prompt_tokens: usize,
    pub(crate) decode_steps: usize,
}

pub(crate) fn measure<S: RuntimeSession>(
    session: &S,
    prompt: &[u32],
    vocab: usize,
) -> Result<PhaseSample> {
    let prefill_start = Instant::now();
    session.prefill(prompt, &mut || Ok(()))?;
    let prefill_ns = nanos(prefill_start);

    // The first sample closes prefill but is deliberately outside both phase
    // measurements so the public prefill boundary remains unchanged.
    let first_start = Instant::now();
    let mut token = session.argmax(vocab)?;
    let first_sample_ns = nanos(first_start);

    let mut decode_ns = [0; DECODE_STEPS];
    for (step, elapsed) in decode_ns.iter_mut().enumerate() {
        let start = Instant::now();
        token = session.token_argmax(token, prompt.len() + step, vocab)?;
        *elapsed = nanos(start);
    }

    Ok(PhaseSample {
        prefill_ns,
        first_sample_ns,
        decode_ns,
        prompt_tokens: prompt.len(),
        decode_steps: DECODE_STEPS,
    })
}

fn nanos(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use super::*;
    use crate::backend::Backend;
    use crate::backend::hybrid::weights::runtime::RuntimeShape;
    use crate::kv_cache::Kv;
    use crate::runtime::contract::LayeredGraph;

    struct TestGraph;
    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    struct TestBatch<'a, B: Backend>(std::marker::PhantomData<&'a B>);

    impl LayeredGraph for TestGraph {
        type Config = ();
        #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
        type Batch<'a, B: Backend>
            = TestBatch<'a, B>
        where
            B: 'a;

        #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
        const BATCH_ROWS: usize = 1;

        fn shape(_: &Self::Config) -> RuntimeShape {
            unreachable!("the timing test owns no graph")
        }

        fn token<B: Backend>(
            _: &B,
            _: &Self::Config,
            _: &Kv<B::Buffer>,
            _: u32,
            _: usize,
        ) -> Result<()> {
            unreachable!("the timing test owns no graph")
        }

        #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
        fn embedding<B: Backend>(_: &B, _: &B::Encoder, _: &Self::Config, _: u32) -> Result<()> {
            unreachable!("the timing test owns no graph")
        }

        #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
        fn range<B: Backend>(
            _: &B,
            _: &B::Encoder,
            _: &Self::Config,
            _: &Kv<B::Buffer>,
            _: std::ops::Range<usize>,
            _: usize,
        ) -> Result<()> {
            unreachable!("the timing test owns no graph")
        }

        #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
        fn tail<B: Backend>(_: &B, _: &B::Encoder, _: &Self::Config) {
            unreachable!("the timing test owns no graph")
        }

        fn prefill<B: Backend>(
            _: &B,
            _: &Self::Config,
            _: &Kv<B::Buffer>,
            _: &[u32],
            _: &mut dyn FnMut() -> Result<()>,
        ) -> Result<()> {
            unreachable!("the timing test owns no graph")
        }

        #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
        fn batch<'a, B: Backend>(
            _: &'a B,
            _: &Self::Config,
            _: usize,
        ) -> Result<Self::Batch<'a, B>> {
            unreachable!("the timing test owns no graph")
        }

        #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
        fn batch_residual<'a, B: Backend>(_: &'a Self::Batch<'_, B>) -> &'a B::Buffer {
            unreachable!("the timing test owns no graph")
        }

        #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
        fn record_batch<B: Backend>(
            _: &B,
            _: &Self::Config,
            _: &Kv<B::Buffer>,
            _: &Self::Batch<'_, B>,
            _: &[u32],
            _: usize,
            _: bool,
            _: bool,
        ) -> Result<()> {
            unreachable!("the timing test owns no graph")
        }
    }

    #[derive(Default)]
    struct TestSession {
        prefills: Cell<usize>,
        argmaxes: Cell<usize>,
        tokens: RefCell<Vec<(u32, usize)>>,
    }

    impl RuntimeSession for TestSession {
        type Graph = TestGraph;

        fn prefill(&self, _: &[u32], before: &mut dyn FnMut() -> Result<()>) -> Result<()> {
            before()?;
            self.prefills.set(self.prefills.get() + 1);
            Ok(())
        }

        fn token(&self, token: u32, position: usize) -> Result<()> {
            self.tokens.borrow_mut().push((token, position));
            Ok(())
        }

        fn logits(&self, _: usize) -> Result<Vec<f32>> {
            unreachable!()
        }

        fn argmax(&self, _: usize) -> Result<u32> {
            let next = self.argmaxes.get() + 1;
            self.argmaxes.set(next);
            Ok(next as u32)
        }

        fn topk(&self, _: usize, _: usize) -> Result<Vec<(u32, f32)>> {
            unreachable!()
        }
    }

    #[test]
    fn separates_first_sample_and_runs_all_decode_steps_even_for_eos() {
        let session = TestSession::default();
        let prompt = [7, 8];
        let sample = measure(&session, &prompt, 10).unwrap();

        assert_eq!(session.prefills.get(), 1);
        assert_eq!(session.argmaxes.get(), DECODE_STEPS + 1);
        assert_eq!(session.tokens.borrow().len(), DECODE_STEPS);
        assert_eq!(session.tokens.borrow()[0], (1, 2));
        assert_eq!(session.tokens.borrow()[DECODE_STEPS - 1], (31, 32));
        assert_eq!(sample.prompt_tokens, prompt.len());
        assert_eq!(sample.decode_steps, DECODE_STEPS);
        assert_eq!(sample.decode_ns.len(), DECODE_STEPS);
    }

    #[test]
    fn default_token_argmax_stops_before_readback_when_token_fails() {
        struct FailedToken(TestSession);
        impl RuntimeSession for FailedToken {
            type Graph = TestGraph;
            fn prefill(&self, _: &[u32], _: &mut dyn FnMut() -> Result<()>) -> Result<()> {
                Ok(())
            }
            fn token(&self, _: u32, _: usize) -> Result<()> {
                color_eyre::eyre::bail!("token failed")
            }
            fn logits(&self, _: usize) -> Result<Vec<f32>> {
                unreachable!()
            }
            fn argmax(&self, vocab: usize) -> Result<u32> {
                self.0.argmax(vocab)
            }
            fn topk(&self, _: usize, _: usize) -> Result<Vec<(u32, f32)>> {
                unreachable!()
            }
        }
        let session = FailedToken(TestSession::default());
        assert_eq!(
            session.token_argmax(1, 0, 2).unwrap_err().to_string(),
            "token failed"
        );
        assert_eq!(session.0.argmaxes.get(), 0);
    }
}
