/*
 * graph_horizon_engine — family-neutral chat throughput
 * Measures the public Engine event stream and returns data-only timing
 * aggregates. Prompt counts come from the terminal GenerationStats, so this
 * harness never reaches into a family tokenizer or template.
 */

use std::time::Instant;

use color_eyre::eyre::{Result, bail, eyre};

use crate::api::engine::Engine;
use crate::api::event::Event;
use crate::api::message::{Message, Role};
use crate::api::request::{Request, SamplingParams};
use crate::harness::stats::{aggregate, rep_metrics};

pub struct Stat {
    pub mean: f64,
    pub stddev: Option<f64>,
}

pub struct BenchConfig {
    pub preset: String,
    pub prompt: String,
    pub gen_tokens: usize,
    pub reps: usize,
    pub warmup: usize,
}

pub struct ThroughputReport {
    pub model: String,
    pub preset: String,
    pub n_prompt: usize,
    pub gen_tokens: usize,
    pub context_tokens: usize,
    pub reps: usize,
    pub warmup: usize,
    pub ttft_seconds: Stat,
    pub prompt_tps: Stat,
    pub tg: Option<Stat>,
    pub tg_first_segment: Option<Stat>,
    pub tg_last_segment: Option<Stat>,
    pub decoded_tokens: usize,
}

pub(super) struct RepSample {
    pub(super) deltas: usize,
    pub(super) ttft_s: Option<f64>,
    pub(super) prompt_tps: Option<f64>,
    pub(super) tg: Option<f64>,
    pub(super) tg_first: Option<f64>,
    pub(super) tg_last: Option<f64>,
}

pub fn run(engine: &Engine, cfg: &BenchConfig) -> Result<ThroughputReport> {
    let reps = cfg.reps.max(1);
    for _ in 0..cfg.warmup {
        measure_rep(engine, cfg)?;
    }
    let mut samples = Vec::with_capacity(reps);
    let mut n_prompt = 0;
    for _ in 0..reps {
        let (sample, prompt) = measure_rep(engine, cfg)?;
        samples.push(sample);
        n_prompt = prompt;
    }
    if samples.iter().all(|sample| sample.deltas == 0) {
        bail!("throughput: no measurable rep");
    }
    let decoded_tokens = samples.last().map(|sample| sample.deltas).unwrap_or(0);
    Ok(ThroughputReport {
        model: "mistral3".into(),
        preset: cfg.preset.clone(),
        n_prompt,
        gen_tokens: cfg.gen_tokens,
        context_tokens: n_prompt.saturating_add(cfg.gen_tokens),
        reps,
        warmup: cfg.warmup,
        ttft_seconds: aggregate(&samples, |sample| sample.ttft_s).unwrap(),
        prompt_tps: aggregate(&samples, |sample| sample.prompt_tps).unwrap(),
        tg: aggregate(&samples, |sample| sample.tg),
        tg_first_segment: aggregate(&samples, |sample| sample.tg_first),
        tg_last_segment: aggregate(&samples, |sample| sample.tg_last),
        decoded_tokens,
    })
}

fn measure_rep(engine: &Engine, cfg: &BenchConfig) -> Result<(RepSample, usize)> {
    let request = Request {
        messages: vec![Message {
            role: Role::User,
            content: cfg.prompt.clone(),
        }],
        sampling: SamplingParams::greedy(),
        max_tokens: cfg.gen_tokens,
    };
    let start = Instant::now();
    let mut timestamps = Vec::new();
    let mut stats = None;
    let mut failure = None;
    engine.generate(request, &mut |event| match event {
        Event::TextDelta(_) => {
            timestamps.push(Instant::now());
            true
        }
        Event::Finished(value) => {
            stats = Some(value);
            true
        }
        Event::Error(message) => {
            failure = Some(message);
            false
        }
    });
    if let Some(message) = failure {
        return Err(eyre!(message));
    }
    let stats = stats.ok_or_else(|| eyre!("throughput: generation was cancelled"))?;
    let offsets = timestamps
        .iter()
        .map(|time| time.duration_since(start).as_secs_f64())
        .collect::<Vec<_>>();
    Ok((
        rep_metrics(&offsets, stats.prompt_tokens),
        stats.prompt_tokens,
    ))
}
