/*
 * gh_zero_engine — final Ministral request lifecycle
 * Renders one text chat, owns request KV through a backend-neutral session,
 * streams UTF-8 deltas, and preserves terminal and load-time placement invariants.
 */

use color_eyre::eyre::Result;

use super::decode::TextDecoder;
#[cfg(any(test, not(feature = "hybrid")))]
use super::graph::{forward, prefill};
use super::{MistralConfig, RuntimeModel, template};
use crate::api::event::{GenerationStats, Terminal};
use crate::api::request::{EventSink, Request};
#[cfg(not(feature = "hybrid"))]
use crate::backend::Backend;
#[cfg(not(feature = "hybrid"))]
use crate::kv_cache::{self, Kv};
use crate::sampling::{self, Rng};

trait Session {
    fn prefill(&self, prompt: &[u32], before: &mut dyn FnMut() -> Result<()>) -> Result<()>;
    fn token(&self, token: u32, pos: usize) -> Result<()>;
    fn logits(&self, vocab: usize) -> Result<Vec<f32>>;
    fn argmax(&self, vocab: usize) -> Result<u32>;
    fn topk(&self, vocab: usize, k: usize) -> Result<Vec<(u32, f32)>>;
}

#[cfg(not(feature = "hybrid"))]
struct BackendSession<'a, B: Backend> {
    backend: &'a B,
    config: &'a MistralConfig,
    kv: Option<Kv<B::Buffer>>,
}

#[cfg(not(feature = "hybrid"))]
impl<'a, B: Backend> BackendSession<'a, B> {
    fn new(model: &'a RuntimeModel, backend: &'a B) -> Result<Self> {
        let cfg = &model.config;
        let kv = kv_cache::alloc_shape(
            backend,
            cfg.block_count,
            model.context,
            cfg.kv_head_count,
            cfg.key_length,
            cfg.value_length,
            model.scheme,
        )?;
        Ok(Self {
            backend,
            config: cfg,
            kv: Some(kv),
        })
    }

    fn kv(&self) -> &Kv<B::Buffer> {
        self.kv.as_ref().expect("request KV exists until drop")
    }
}

#[cfg(not(feature = "hybrid"))]
impl<B: Backend> Session for BackendSession<'_, B> {
    fn prefill(&self, prompt: &[u32], before: &mut dyn FnMut() -> Result<()>) -> Result<()> {
        prefill::prefill_with(self.backend, self.config, self.kv(), prompt, before)
    }

    fn token(&self, token: u32, pos: usize) -> Result<()> {
        forward::token(self.backend, self.config, self.kv(), token, pos)
    }

    fn logits(&self, vocab: usize) -> Result<Vec<f32>> {
        self.backend
            .read_logits(&self.backend.buffers().logits, vocab)
    }

    fn argmax(&self, vocab: usize) -> Result<u32> {
        self.backend
            .read_argmax(&self.backend.buffers().logits, vocab)
    }

    fn topk(&self, vocab: usize, k: usize) -> Result<Vec<(u32, f32)>> {
        self.backend
            .read_topk(&self.backend.buffers().logits, vocab, k)
    }
}

#[cfg(not(feature = "hybrid"))]
impl<B: Backend> Drop for BackendSession<'_, B> {
    fn drop(&mut self) {
        if let Some(kv) = self.kv.take() {
            kv_cache::free(self.backend, kv);
        }
    }
}

#[cfg(feature = "hybrid")]
struct HybridSession<'a> {
    config: &'a MistralConfig,
    kv: super::hybrid::forward::RequestKv<'a>,
}

#[cfg(feature = "hybrid")]
impl Session for HybridSession<'_> {
    fn prefill(&self, prompt: &[u32], before: &mut dyn FnMut() -> Result<()>) -> Result<()> {
        super::hybrid::prefill::run(&self.kv, self.config, prompt, before)
    }

    fn token(&self, token: u32, pos: usize) -> Result<()> {
        super::hybrid::forward::token(&self.kv, self.config, token, pos)
    }

    fn logits(&self, vocab: usize) -> Result<Vec<f32>> {
        super::hybrid::forward::read_logits(&self.kv, vocab)
    }

    fn argmax(&self, vocab: usize) -> Result<u32> {
        super::hybrid::forward::read_argmax(&self.kv, vocab)
    }

    fn topk(&self, vocab: usize, k: usize) -> Result<Vec<(u32, f32)>> {
        super::hybrid::forward::read_topk(&self.kv, vocab, k)
    }
}

pub(crate) fn generate(model: &RuntimeModel, request: Request, sink: &mut dyn EventSink) {
    let mut terminal = Terminal::new(sink);
    match execute(model, &request, &mut terminal) {
        Ok(Some(stats)) => terminal.finish(stats),
        Ok(None) => {}
        Err(_) => terminal.fail(),
    }
}

fn execute(
    model: &RuntimeModel,
    request: &Request,
    terminal: &mut Terminal<'_>,
) -> Result<Option<GenerationStats>> {
    let prompt = template::render(&request.messages, &model.tokenizer, model.context)?;
    #[cfg(not(feature = "hybrid"))]
    let session = BackendSession::new(model, &model.backend)?;
    #[cfg(feature = "hybrid")]
    let session = HybridSession {
        config: &model.config,
        kv: super::hybrid::forward::RequestKv::new(
            &model.backend,
            &model.config,
            model.context,
            model.scheme,
        )?,
    };
    drive(model, request, &prompt, &session, terminal)
}

fn drive(
    model: &RuntimeModel,
    request: &Request,
    prompt: &[u32],
    session: &dyn Session,
    terminal: &mut Terminal<'_>,
) -> Result<Option<GenerationStats>> {
    let prefill_start = std::time::Instant::now();
    session.prefill(prompt, &mut || {
        (!terminal.cancelled())
            .then_some(())
            .ok_or_else(|| color_eyre::eyre::eyre!("generation cancelled"))
    })?;
    let decode_start = std::time::Instant::now();
    let prefill_ms = decode_start.duration_since(prefill_start).as_millis() as u64;
    let mut decoder = TextDecoder::default();
    let mut recent = prompt.to_vec();
    let mut rng = Rng::new(request.sampling.seed);
    let mut produced = 0;
    // Immutable request parameters select one readback path for the whole decode.
    let sample_path = sampling::plan(&request.sampling);

    while produced < request.max_tokens && prompt.len() + produced < model.context {
        if terminal.cancelled() {
            return Ok(None);
        }
        let token = match sample_path {
            sampling::SamplePath::Greedy => session.argmax(model.config.vocab_size)?,
            sampling::SamplePath::TopK(k) => {
                let mut candidates = session.topk(model.config.vocab_size, k)?;
                let inv_temperature = 1.0 / request.sampling.temperature;
                for (_, logit) in &mut candidates {
                    *logit *= inv_temperature;
                }
                sampling::sample_from_candidates(candidates, &request.sampling, &mut rng)
            }
            sampling::SamplePath::Fallback => {
                let mut logits = session.logits(model.config.vocab_size)?;
                sampling::sample(&mut logits, &request.sampling, &recent, &mut rng)
            }
        };
        if token == model.tokenizer.eos_id() {
            break;
        }
        if let Some(text) = decoder.push(&model.tokenizer.decode_bytes(&[token]))
            && !terminal.delta(text)
        {
            return Ok(None);
        }
        recent.push(token);
        produced += 1;
        if produced < request.max_tokens && prompt.len() + produced < model.context {
            session.token(token, prompt.len() + produced - 1)?;
        }
    }
    decoder.finish();
    Ok(Some(GenerationStats {
        prompt_tokens: prompt.len(),
        completion_tokens: produced,
        prefill_ms,
        decode_ms: decode_start.elapsed().as_millis() as u64,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::event::Event;
    #[cfg(feature = "cpu")]
    use crate::api::message::{Message, Role};
    use crate::backend::Backend;
    #[cfg(feature = "cpu")]
    use crate::backend::cpu::CpuBackend;
    use crate::family::mistral::MistralModel;
    use crate::family::mistral::graph::shape::{ShapeBackend, config};
    #[cfg(feature = "cpu")]
    use crate::family::mistral::tokenizer::TekkenTokenizer;
    #[cfg(feature = "cpu")]
    use crate::family::mistral::{MistralContract, template};
    #[cfg(feature = "cpu")]
    use crate::gguf::loader::GgufFile;
    use crate::kv_cache::scheme::KvQuant;
    use crate::kv_cache::{self, Kv};
    use color_eyre::eyre::bail;
    use std::cell::Cell;
    #[cfg(feature = "cpu")]
    use std::process::Command;
    use std::rc::Rc;

    struct Request {
        prompt: Vec<u32>,
        max_tokens: usize,
        context: usize,
        kv_quant: crate::kv_cache::scheme::KvQuant,
    }

    #[derive(Default)]
    struct Events {
        cancelled: bool,
        values: Vec<Event>,
    }

    impl EventSink for Events {
        fn cancelled(&self) -> bool {
            self.cancelled
        }

        fn emit(&mut self, event: Event) -> bool {
            self.values.push(event);
            !self.cancelled
        }
    }

    #[test]
    fn error_matrix_e18_e19_has_one_terminal_or_silent_cancel() {
        let stats = GenerationStats {
            prompt_tokens: 2,
            completion_tokens: 1,
            prefill_ms: 3,
            decode_ms: 4,
        };
        let mut success = Events::default();
        let mut terminal = Terminal::new(&mut success);
        assert!(terminal.delta("x".into()));
        terminal.finish(stats);
        terminal.finish(stats);
        assert_eq!(
            success.values,
            [Event::TextDelta("x".into()), Event::Finished(stats)]
        );

        let mut failure = Events::default();
        let mut terminal = Terminal::new(&mut failure);
        terminal.fail();
        terminal.finish(stats);
        assert_eq!(failure.values, [Event::Error("generation failed".into())]);

        let mut cancelled = Events {
            cancelled: true,
            values: Vec::new(),
        };
        let mut terminal = Terminal::new(&mut cancelled);
        terminal.fail();
        terminal.finish(stats);
        assert!(cancelled.values.is_empty());
    }

    #[test]
    fn api_cancellation_discards_incomplete_utf8() {
        let mut decoder = TextDecoder::default();
        assert_eq!(decoder.push(&[0xe2, 0x82]), None);
        drop(decoder);
    }

    #[test]
    fn reasoning_output_tags_remain_text_deltas() {
        let mut events = Events::default();
        let mut terminal = Terminal::new(&mut events);
        let mut decoder = TextDecoder::default();
        for bytes in [
            b"[TH".as_slice(),
            b"INK]x[/".as_slice(),
            b"THINK]".as_slice(),
        ] {
            if let Some(text) = decoder.push(bytes) {
                assert!(terminal.delta(text));
            }
        }
        decoder.finish();
        terminal.finish(GenerationStats {
            prompt_tokens: 1,
            completion_tokens: 3,
            prefill_ms: 0,
            decode_ms: 0,
        });

        let text = events
            .values
            .iter()
            .filter_map(|event| match event {
                Event::TextDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(text, "[THINK]x[/THINK]");
        assert!(!text.contains('\u{fffd}'));
        assert_eq!(
            events
                .values
                .iter()
                .filter(|event| matches!(event, Event::Finished(_)))
                .count(),
            1
        );
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Boundary {
        BeforePrefill,
        BeforeDecode,
    }

    trait Cancel {
        fn cancelled(&self, boundary: Boundary) -> bool;
    }

    struct NeverCancel;

    impl Cancel for NeverCancel {
        fn cancelled(&self, _: Boundary) -> bool {
            false
        }
    }

    struct Tracker(Rc<Cell<usize>>);

    impl Tracker {
        fn new() -> Self {
            Self(Rc::new(Cell::new(0)))
        }

        fn live(&self) -> usize {
            self.0.get()
        }
    }

    struct OwnedKv<'a, B: Backend> {
        backend: &'a B,
        kv: Option<Kv<B::Buffer>>,
        live: Rc<Cell<usize>>,
    }

    impl<'a, B: Backend> OwnedKv<'a, B> {
        fn new(
            backend: &'a B,
            cfg: &MistralConfig,
            req: &Request,
            tracker: &Tracker,
        ) -> Result<Self> {
            let kv = kv_cache::alloc_shape(
                backend,
                cfg.block_count,
                req.context,
                cfg.kv_head_count,
                cfg.key_length,
                cfg.value_length,
                req.kv_quant,
            )?;
            tracker.0.set(tracker.0.get() + 1);
            Ok(Self {
                backend,
                kv: Some(kv),
                live: Rc::clone(&tracker.0),
            })
        }
    }

    impl<B: Backend> Drop for OwnedKv<'_, B> {
        fn drop(&mut self) {
            if let Some(kv) = self.kv.take() {
                kv_cache::free(self.backend, kv);
                self.live.set(self.live.get() - 1);
            }
        }
    }

    fn generate<B: Backend, C: Cancel>(
        model: &MistralModel<B>,
        req: &Request,
        cancel: &C,
        tracker: &Tracker,
        fail_at: Option<Boundary>,
    ) -> Result<Vec<u32>> {
        generate_with_prefill(model, req, cancel, tracker, fail_at, true)
    }

    fn generate_with_prefill<B: Backend, C: Cancel>(
        model: &MistralModel<B>,
        req: &Request,
        cancel: &C,
        tracker: &Tracker,
        fail_at: Option<Boundary>,
        batched: bool,
    ) -> Result<Vec<u32>> {
        let required = req
            .prompt
            .len()
            .checked_add(req.max_tokens.saturating_sub(1));
        if req.prompt.is_empty()
            || req.prompt.len() > req.context
            || required.is_none_or(|n| n > req.context)
        {
            bail!("E11 invalid generation request");
        }
        let owned = OwnedKv::new(&model.backend, &model.config, req, tracker)?;
        if batched {
            prefill::prefill_with(
                &model.backend,
                &model.config,
                owned.kv.as_ref().unwrap(),
                &req.prompt,
                || boundary(cancel, fail_at, Boundary::BeforePrefill),
            )?;
        } else {
            boundary(cancel, fail_at, Boundary::BeforePrefill)?;
            for (pos, &token) in req.prompt.iter().enumerate() {
                forward::token(
                    &model.backend,
                    &model.config,
                    owned.kv.as_ref().unwrap(),
                    token,
                    pos,
                )?;
            }
        }
        let mut tokens = Vec::new();
        for step in 0..req.max_tokens {
            boundary(cancel, fail_at, Boundary::BeforeDecode)?;
            let next = model
                .backend
                .read_argmax(&model.backend.buffers().logits, model.config.vocab_size)?;
            tokens.push(next);
            let pos = req.prompt.len() + step;
            if step + 1 < req.max_tokens && pos + 1 < req.context {
                forward::token(
                    &model.backend,
                    &model.config,
                    owned.kv.as_ref().unwrap(),
                    next,
                    pos,
                )?;
            }
        }
        Ok(tokens)
    }

    #[cfg(feature = "cpu")]
    fn teacher_forced_top2<B: Backend>(
        model: &MistralModel<B>,
        req: &Request,
        reference: &[u32],
        tracker: &Tracker,
    ) -> Result<Vec<Vec<u32>>> {
        let required = req
            .prompt
            .len()
            .checked_add(reference.len().saturating_sub(1));
        if req.prompt.is_empty() || reference.is_empty() || required.is_none_or(|n| n > req.context)
        {
            bail!("E11 invalid teacher-forcing request");
        }
        let owned = OwnedKv::new(&model.backend, &model.config, req, tracker)?;
        prefill::prefill_with(
            &model.backend,
            &model.config,
            owned.kv.as_ref().unwrap(),
            &req.prompt,
            || Ok(()),
        )?;
        let mut top2 = Vec::with_capacity(reference.len());
        for (step, &token) in reference.iter().enumerate() {
            let candidates = model
                .backend
                .read_topk(&model.backend.buffers().logits, model.config.vocab_size, 2)?
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            top2.push(candidates);
            if step + 1 < reference.len() {
                forward::token(
                    &model.backend,
                    &model.config,
                    owned.kv.as_ref().unwrap(),
                    token,
                    req.prompt.len() + step,
                )?;
            }
        }
        Ok(top2)
    }

    fn boundary<C: Cancel>(cancel: &C, fail: Option<Boundary>, at: Boundary) -> Result<()> {
        if cancel.cancelled(at) || fail == Some(at) {
            bail!("generation stopped");
        }
        Ok(())
    }

    struct CancelAt(Boundary);

    impl Cancel for CancelAt {
        fn cancelled(&self, boundary: Boundary) -> bool {
            self.0 == boundary
        }
    }

    fn model() -> MistralModel<ShapeBackend> {
        let config = config(1, 4096, 512);
        model_with(config.clone(), ShapeBackend::new(&config, false))
    }

    fn model_with(config: MistralConfig, backend: ShapeBackend) -> MistralModel<ShapeBackend> {
        MistralModel { backend, config }
    }

    fn request(max_tokens: usize) -> Request {
        Request {
            prompt: vec![1, 2],
            max_tokens,
            context: 8,
            kv_quant: KvQuant::F16,
        }
    }

    #[test]
    fn prompt_overflow_fails_before_allocation() {
        let model = model();
        let tracker = Tracker::new();
        let mut req = request(1);
        req.context = 1;
        assert!(generate(&model, &req, &NeverCancel, &tracker, None).is_err());
        assert_eq!(tracker.live(), 0);
    }

    #[test]
    fn zero_tokens_prefills_and_releases_kv() {
        let model = model();
        let tracker = Tracker::new();
        assert!(generate(&model, &request(0), &NeverCancel, &tracker, None).is_ok());
        assert_eq!(tracker.live(), 0);
    }

    #[test]
    fn error_matrix_e18_e19_failures_and_cancellation_release_kv() {
        for fail_at in [Some(Boundary::BeforePrefill), Some(Boundary::BeforeDecode)] {
            let model = model();
            let tracker = Tracker::new();
            assert!(generate(&model, &request(1), &NeverCancel, &tracker, fail_at).is_err());
            assert_eq!(tracker.live(), 0);
        }
        let model = model();
        let tracker = Tracker::new();
        assert!(
            generate(
                &model,
                &request(1),
                &CancelAt(Boundary::BeforeDecode),
                &tracker,
                None
            )
            .is_err()
        );
        assert_eq!(tracker.live(), 0);
    }

    #[test]
    fn allocation_and_numeric_failures_release_every_buffer() {
        for allocation in [0, 1, 4] {
            let config = config(1, 4096, 512);
            let backend = ShapeBackend::new(&config, false);
            backend.fail_allocation(allocation);
            let model = model_with(config, backend);
            assert!(generate(&model, &request(0), &NeverCancel, &Tracker::new(), None).is_err());
            assert_eq!(model.backend.live_allocations(), 0);
        }

        let model = model();
        model.backend.fail_embed();
        assert!(generate(&model, &request(0), &NeverCancel, &Tracker::new(), None).is_err());
        assert_eq!(model.backend.live_allocations(), 0);
    }

    struct CancelCall {
        boundary: Boundary,
        call: usize,
        seen: Cell<usize>,
    }

    impl Cancel for CancelCall {
        fn cancelled(&self, boundary: Boundary) -> bool {
            if boundary != self.boundary {
                return false;
            }
            let seen = self.seen.get();
            self.seen.set(seen + 1);
            seen == self.call
        }
    }

    #[test]
    fn cancellation_is_checked_before_each_batch_and_decode_step() {
        for (boundary, call, mut req) in [
            (Boundary::BeforePrefill, 1, request(0)),
            (Boundary::BeforeDecode, 1, request(3)),
        ] {
            if boundary == Boundary::BeforePrefill {
                req.prompt = vec![1, 2, 3, 4, 5];
            }
            let model = model();
            let cancel = CancelCall {
                boundary,
                call,
                seen: Cell::new(0),
            };
            assert!(generate(&model, &req, &cancel, &Tracker::new(), None).is_err());
            assert_eq!(cancel.seen.get(), 2);
            assert_eq!(model.backend.live_allocations(), 0);
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    #[ignore]
    fn real_greedy_parity() {
        let path = std::env::var("GH_ZERO_MODEL").expect("GH_ZERO_MODEL required");
        let context = std::env::var("GH_ZERO_CONTEXT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(4096);
        let scheme = std::env::var("GH_ZERO_KV")
            .ok()
            .and_then(|value| KvQuant::parse(&value))
            .unwrap_or(KvQuant::F16);
        let file = GgufFile::open(std::path::Path::new(&path)).expect("open GGUF");
        let contract = MistralContract::from_gguf(&file).expect("Ministral contract");
        let content = "Scrivi una riga con 20 emoji diverse, senza parole.";
        let prompt = template::render(
            &[Message {
                role: Role::User,
                content: content.into(),
            }],
            &contract.tokenizer,
            context,
        )
        .unwrap();
        let rendered = format!("[INST]{content}[/INST]");
        let reference_prompt = reference_ids(&path, &rendered, false);
        assert_eq!(prompt, reference_prompt);

        let max_tokens = 16;
        let reference = reference_completion(&path, &rendered, context, max_tokens);
        let model = MistralModel::<CpuBackend>::load(&file, context).expect("load CPU model");
        let request = Request {
            prompt,
            max_tokens,
            context,
            kv_quant: scheme,
        };
        let sequential_ids =
            generate_with_prefill(&model, &request, &NeverCancel, &Tracker::new(), None, false)
                .unwrap();
        let batched_ids = generate(&model, &request, &NeverCancel, &Tracker::new(), None).unwrap();
        let sequential = completion_bytes(&contract.tokenizer, &sequential_ids);
        let batched = completion_bytes(&contract.tokenizer, &batched_ids);
        println!(
            "kv={} prompt_ids={reference_prompt:?} reference={reference:?} sequential_ids={sequential_ids:?} batched_ids={batched_ids:?}",
            scheme.name(),
        );
        assert_eq!(sequential, reference, "sequential prefill");
        assert_eq!(batched, reference, "batched prefill");
    }

    #[cfg(feature = "cpu")]
    #[test]
    #[ignore = "requires an approved Ministral model and externally supplied oracle IDs"]
    fn real_ministral_parity() {
        use crate::family::mistral::parity;

        let path = std::env::var("GH_ZERO_MODEL").expect("GH_ZERO_MODEL required");
        let context = std::env::var("GH_ZERO_CONTEXT")
            .expect("GH_ZERO_CONTEXT required")
            .parse::<usize>()
            .expect("GH_ZERO_CONTEXT must be an unsigned decimal");
        assert_eq!(context, parity::CONTEXT, "GH_ZERO_CONTEXT must be 4096");
        let scheme = std::env::var("GH_ZERO_KV")
            .ok()
            .and_then(|value| KvQuant::parse(&value))
            .expect("GH_ZERO_KV must be f16 or int8");
        let reference = parity::reference_vectors();
        let file = GgufFile::open(std::path::Path::new(&path)).expect("open approved GGUF");
        let contract = MistralContract::from_gguf(&file).expect("Ministral contract");
        let prompt = template::render(
            &[Message {
                role: Role::User,
                content: parity::USER_CONTENT.into(),
            }],
            &contract.tokenizer,
            context,
        )
        .expect("Ministral prompt");
        parity::assert_exact("prompt IDs", &prompt, &reference.prompt);

        let model = MistralModel::<CpuBackend>::load(&file, context).expect("load CPU model");
        let request = Request {
            prompt: prompt.clone(),
            max_tokens: parity::TOKEN_COUNT,
            context,
            kv_quant: scheme,
        };
        let sequential =
            generate_with_prefill(&model, &request, &NeverCancel, &Tracker::new(), None, false)
                .expect("sequential completion");
        let batched = generate(&model, &request, &NeverCancel, &Tracker::new(), None)
            .expect("batched completion");
        let top2 = teacher_forced_top2(&model, &request, &reference.completion, &Tracker::new())
            .expect("teacher-forced top two");
        println!(
            "profile=Q4_K_M kv={} prompt_ids={prompt:?} reference_completion_ids={:?} sequential_ids={sequential:?} batched_ids={batched:?} teacher_top2={top2:?}",
            scheme.name(),
            reference.completion
        );
        assert_eq!(sequential.len(), parity::TOKEN_COUNT);
        assert_eq!(batched.len(), parity::TOKEN_COUNT);
        parity::assert_exact("sequential/batched local greedy IDs", &sequential, &batched);
        parity::assert_oracle_top2(&top2, &reference.completion);
        println!(
            "ministral-parity: local_ids={} oracle_top2=pass",
            parity::csv(&batched)
        );
    }

    #[cfg(feature = "cpu")]
    fn completion_bytes(tokenizer: &TekkenTokenizer, tokens: &[u32]) -> Vec<u8> {
        tokens
            .iter()
            .flat_map(|&token| tokenizer.decode_bytes(&[token]))
            .collect()
    }

    #[cfg(feature = "cpu")]
    fn reference_completion(model: &str, prompt: &str, context: usize, tokens: usize) -> Vec<u8> {
        let binary =
            std::env::var("GH_ZERO_REFERENCE_CLI").expect("GH_ZERO_REFERENCE_CLI required");
        let output = Command::new("timeout")
            .arg("300")
            .arg(binary)
            .args([
                "-m",
                model,
                "-p",
                prompt,
                "-c",
                &context.to_string(),
                "-n",
                &tokens.to_string(),
                "--temp",
                "0",
                "--top-k",
                "1",
                "--top-p",
                "1",
                "--min-p",
                "0",
                "--repeat-penalty",
                "1",
                "--no-display-prompt",
                "--no-warmup",
                "--no-perf",
                "--no-conversation",
                "--single-turn",
                "--no-jinja",
            ])
            .output()
            .expect("run llama-completion");
        assert!(output.status.success(), "llama-completion failed");
        let mut bytes = output.stdout;
        assert!(
            bytes.ends_with(b"\n\n"),
            "llama-completion output terminator changed"
        );
        bytes.truncate(bytes.len() - 2);
        bytes
    }

    #[cfg(feature = "cpu")]
    fn reference_ids(model: &str, text: &str, no_bos: bool) -> Vec<u32> {
        let binary = std::env::var("GH_ZERO_REFERENCE_TOKENIZE")
            .expect("GH_ZERO_REFERENCE_TOKENIZE required");
        let mut command = Command::new(binary);
        command.args(["--log-disable", "--ids"]);
        if no_bos {
            command.arg("--no-bos");
        }
        let output = command
            .args(["-m", model, "-p", text])
            .output()
            .expect("run llama-tokenize");
        assert!(output.status.success(), "llama-tokenize failed");
        std::str::from_utf8(&output.stdout)
            .unwrap()
            .trim()
            .trim_start_matches('[')
            .trim_end_matches(']')
            .split(',')
            .filter_map(|value| value.trim().parse().ok())
            .collect()
    }
}

#[cfg(test)]
#[cfg(feature = "cpu")]
pub(crate) mod numeric {
    use crate::backend::Backend;
    use crate::backend::buffers::{Buffers, LayerWeights, Scratch, WeightSet};
    use crate::backend::cpu::{CpuBackend, CpuBuffer, CpuFormat};
    use crate::family::mistral::MistralConfig;
    use crate::family::mistral::graph::{forward, prefill};
    use crate::kv_cache;
    use crate::kv_cache::scheme::KvQuant;

    fn cfg() -> MistralConfig {
        MistralConfig {
            block_count: 1,
            context_length: 16,
            embedding_length: 8,
            feed_forward_length: 8,
            head_count: 2,
            kv_head_count: 1,
            key_length: 4,
            value_length: 2,
            q_width: 8,
            k_width: 4,
            v_width: 2,
            attention_width: 4,
            rope_dimension: 4,
            rope_freq_base: 10_000.0,
            rms_epsilon: 0.00001,
            yarn_factor: 2.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_log_multiplier: 0.1,
            yarn_original_context: 8,
            attention_temperature_scale: 1.1,
            vocab_size: 8,
            bos_id: 1,
            eos_id: 2,
        }
    }

    fn f16(values: &[f32]) -> CpuBuffer {
        let buffer = CpuBuffer::zeroed(values.len() * 2, CpuFormat::F16);
        buffer.write_f16_from_f32(values);
        buffer
    }

    fn matrix(input: usize, output: usize, seed: usize) -> CpuBuffer {
        f16(&(0..input * output)
            .map(|i| ((i + seed) % 11) as f32 * 0.025 - 0.12)
            .collect::<Vec<_>>())
    }

    #[derive(Clone, Copy)]
    enum Profile {
        Dense,
        Mixed,
    }

    fn weight(input: usize, output: usize, seed: usize, profile: Profile) -> CpuBuffer {
        match profile {
            Profile::Dense => matrix(input, output, seed),
            Profile::Mixed if seed.is_multiple_of(2) => {
                quant(input, output, seed, CpuFormat::Q6_K, 210)
            }
            Profile::Mixed => quant(input, output, seed, CpuFormat::Q4_K, 144),
        }
    }

    fn quant(
        input: usize,
        output: usize,
        seed: usize,
        format: CpuFormat,
        block_bytes: usize,
    ) -> CpuBuffer {
        // All retained K-quants use one 256-value super-block.
        let block = 256;
        let mut bytes = vec![0u8; output * (input / block) * block_bytes];
        for chunk in bytes.chunks_exact_mut(block_bytes) {
            match format {
                CpuFormat::Q4_K => {
                    chunk[0..2].copy_from_slice(&f16_bits(0.02));
                    chunk[2..4].copy_from_slice(&f16_bits(0.005));
                    chunk[4..12].fill(1);
                    for (i, byte) in chunk[16..].iter_mut().enumerate() {
                        *byte = ((i + seed) as u8 & 0x0f) | ((((i + seed + 3) as u8) & 0x0f) << 4);
                    }
                }
                CpuFormat::Q6_K => {
                    for (i, byte) in chunk[..192].iter_mut().enumerate() {
                        *byte = (i + seed) as u8;
                    }
                    chunk[192..208].fill(1);
                    chunk[208..210].copy_from_slice(&f16_bits(0.01));
                }
                _ => unreachable!(),
            }
        }
        CpuBuffer::from_bytes(bytes, format)
    }

    fn f16_bits(value: f32) -> [u8; 2] {
        crate::backend::cpu::buffer::f32_to_f16(value).to_le_bytes()
    }

    fn backend(cfg: &MistralConfig, dedicated: bool, profile: Profile) -> CpuBackend {
        let norm = || f16(&vec![1.0; cfg.embedding_length]);
        let layer = LayerWeights {
            attn_norm: norm(),
            attn_q: weight(cfg.embedding_length, cfg.q_width, 1, profile),
            attn_k: weight(cfg.embedding_length, cfg.k_width, 2, profile),
            attn_v: weight(cfg.embedding_length, cfg.v_width, 3, profile),
            attn_output: weight(cfg.attention_width, cfg.embedding_length, 4, profile),
            ffn_norm: norm(),
            ffn_gate: weight(cfg.embedding_length, cfg.feed_forward_length, 5, profile),
            ffn_up: weight(cfg.embedding_length, cfg.feed_forward_length, 6, profile),
            ffn_down: weight(cfg.feed_forward_length, cfg.embedding_length, 7, profile),
        };
        let token_embd = weight(cfg.embedding_length, cfg.vocab_size, 8, profile);
        let output = dedicated.then(|| weight(cfg.embedding_length, cfg.vocab_size, 9, profile));
        let z16 = |n| CpuBuffer::zeroed(n * 2, CpuFormat::F16);
        CpuBackend::from_buffers(Buffers {
            weights: WeightSet {
                token_embd: Some(token_embd),
                output_norm: Some(norm()),
                output,
                layers: vec![layer],
            },
            scratch: Scratch {
                x: CpuBuffer::zeroed(cfg.embedding_length * 4, CpuFormat::F32),
                normed: z16(cfg.embedding_length),
                q: z16(cfg.q_width),
                k: z16(cfg.k_width),
                v: z16(cfg.v_width),
                attn: z16(cfg.attention_width),
                proj: z16(cfg.embedding_length),
                gate: z16(cfg.feed_forward_length),
                up: z16(cfg.feed_forward_length),
                act: z16(cfg.feed_forward_length),
                ffn_out: z16(cfg.embedding_length),
            },
            logits: CpuBuffer::zeroed(cfg.vocab_size * 4, CpuFormat::F32),
        })
    }

    fn logits(
        cfg: &MistralConfig,
        scheme: KvQuant,
        prompt: &[u32],
        batched: bool,
        dedicated: bool,
        profile: Profile,
    ) -> Vec<f32> {
        let backend = backend(cfg, dedicated, profile);
        let kv = kv_cache::alloc_shape(
            &backend,
            cfg.block_count,
            cfg.context_length,
            cfg.kv_head_count,
            cfg.key_length,
            cfg.value_length,
            scheme,
        )
        .unwrap();
        if batched {
            prefill::prefill_with(&backend, cfg, &kv, prompt, || Ok(())).unwrap();
        } else {
            for (pos, &token) in prompt.iter().enumerate() {
                forward::token(&backend, cfg, &kv, token, pos).unwrap();
            }
        }
        let result = backend
            .read_logits(&backend.buffers().logits, cfg.vocab_size)
            .unwrap();
        kv_cache::free(&backend, kv);
        result
    }

    #[test]
    pub(crate) fn batched_final_logits_match_sequential_for_both_kv_schemes() {
        let cfg = cfg();
        for (scheme, dedicated) in [
            (KvQuant::F16, false),
            (KvQuant::F16, true),
            (KvQuant::Int8, false),
            (KvQuant::Int8, true),
        ] {
            for prompt in [&[3u32][..], &[1, 3, 4, 5, 6][..]] {
                let sequential = logits(&cfg, scheme, prompt, false, dedicated, Profile::Dense);
                let batched = logits(&cfg, scheme, prompt, true, dedicated, Profile::Dense);
                let max = sequential
                    .iter()
                    .zip(&batched)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!(
                    "scheme={} rows={} max_error={max}",
                    scheme.name(),
                    prompt.len()
                );
                assert!(max <= 0.05, "prefill error {max}");
            }
        }
    }

    #[test]
    pub(crate) fn quantized_profiles_drive_decode_and_short_prefill() {
        let mut cfg = cfg();
        cfg.embedding_length = 256;
        cfg.feed_forward_length = 256;
        cfg.head_count = 2;
        cfg.kv_head_count = 1;
        cfg.key_length = 128;
        cfg.value_length = 128;
        cfg.q_width = 256;
        cfg.k_width = 128;
        cfg.v_width = 128;
        cfg.attention_width = 256;
        cfg.rope_dimension = 128;
        cfg.vocab_size = 32;
        for scheme in [KvQuant::F16, KvQuant::Int8] {
            for dedicated in [false, true] {
                let prompt = [1, 3, 4, 5, 6];
                let sequential = logits(&cfg, scheme, &prompt, false, dedicated, Profile::Mixed);
                let batched = logits(&cfg, scheme, &prompt, true, dedicated, Profile::Mixed);
                let max = sequential
                    .iter()
                    .zip(&batched)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(sequential.iter().all(|v| v.is_finite()));
                assert!(batched.iter().all(|v| v.is_finite()));
                assert!(max <= 0.05, "quantized prefill error {max}");
            }
        }
    }
}
