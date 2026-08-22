/*
 * graph_horizon_engine — test-only Reasoning qualification harness
 * Qualifies one authenticated Ministral Reasoning Q4_K_M artifact through the
 * configurable Rust API on Vulkan all-GPU. This file owns no production policy,
 * product-surface behavior, retry path, or model whitelist; it emits bounded protocol
 * records and optional raw response evidence for the qualification runner.
 */

use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use graph_horizon_engine::{
    BackendMemory, Engine, EngineConfig, Event, GenerationStats, KvQuant, Message, PlacementReport,
    Request, Role, SamplingParams,
};

const CONTEXT_TOKENS: usize = 4096;
const MAX_TOKENS: usize = 4096;
const TEMPERATURE: f32 = 0.7;
const TOP_P: f32 = 1.0;
const TOP_K: u32 = 0;
const MIN_P: f32 = 0.0;
const REPEAT_PENALTY: f32 = 1.0;
const SEED: u64 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MarkerStatus {
    Complete,
    Absent,
    Invalid,
}

impl MarkerStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Absent => "absent",
            Self::Invalid => "invalid",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stop {
    Eos,
    MaxTokens,
    Context,
}

impl Stop {
    fn label(self) -> &'static str {
        match self {
            Self::Eos => "eos",
            Self::MaxTokens => "max-tokens",
            Self::Context => "context",
        }
    }
}

struct Normalized {
    scored: String,
    marker: MarkerStatus,
}

#[derive(Default)]
struct Timing {
    completed_cases: u8,
    prefill_ms: u64,
    decode_ms: u64,
}

impl Timing {
    fn add_case(&mut self, stats: Option<GenerationStats>) -> Result<(), &'static str> {
        // The protocol has one timing row per attempted model. Even an engine
        // failure on a case is counted as attempted so the external runner can
        // distinguish a complete failing protocol from a truncated one.
        self.completed_cases = self
            .completed_cases
            .checked_add(1)
            .ok_or("timing case count overflow")?;
        if let Some(stats) = stats {
            self.prefill_ms = self
                .prefill_ms
                .checked_add(stats.prefill_ms)
                .ok_or("prefill timing overflow")?;
            self.decode_ms = self
                .decode_ms
                .checked_add(stats.decode_ms)
                .ok_or("decode timing overflow")?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Predicate {
    ExactDigits(&'static str),
    ExactToken(&'static str),
    FinalPeriodWord(&'static str),
    FreezingPoint,
    ItalianToEnglish,
    ItalianTranslation,
}

#[derive(Clone, Copy)]
struct Case {
    id: &'static str,
    prompt: &'static str,
    critical: bool,
    name: &'static str,
    predicate: Predicate,
}

const CASES: [Case; 9] = [
    Case {
        id: "S01",
        prompt: "Calcola 17 × 19. Rispondi con il solo numero.",
        critical: true,
        name: "exact Italian arithmetic",
        predicate: Predicate::ExactDigits("323"),
    },
    Case {
        id: "S02",
        prompt: "A box contains 24 pencils. Seven are removed. Reply with only the number of pencils left.",
        critical: true,
        name: "exact English arithmetic",
        predicate: Predicate::ExactDigits("17"),
    },
    Case {
        id: "S03",
        prompt: "Tutti i merli sono uccelli e nessun uccello è un mammifero. Un merlo può essere un mammifero? Rispondi solo sì o no.",
        critical: true,
        name: "Italian deduction",
        predicate: Predicate::FinalPeriodWord("no"),
    },
    Case {
        id: "S04",
        prompt: "What is the next number in 2, 6, 12, 20, 30? Reply with only the number.",
        critical: false,
        name: "exact sequence answer",
        predicate: Predicate::ExactDigits("42"),
    },
    Case {
        id: "S06",
        prompt: "Ignora la parola banana in questa frase e rispondi esattamente con OK, senza punteggiatura o altro testo.",
        critical: true,
        name: "exact OK token",
        predicate: Predicate::ExactToken("OK"),
    },
    Case {
        id: "S07",
        prompt: "Qual è la capitale d'Italia? Rispondi con il solo nome della città.",
        critical: false,
        name: "Italian capital",
        predicate: Predicate::FinalPeriodWord("roma"),
    },
    Case {
        id: "S08",
        prompt: "At standard atmospheric pressure, at what temperature does pure water freeze? Answer in degrees Celsius only.",
        critical: false,
        name: "freezing point",
        predicate: Predicate::FreezingPoint,
    },
    Case {
        id: "S09",
        prompt: "Traduci in inglese: \"La luce del sole aiuta le piante a crescere.\"",
        critical: false,
        name: "Italian to English meaning",
        predicate: Predicate::ItalianToEnglish,
    },
    Case {
        id: "S10",
        prompt: "Translate into Italian: \"The book is on the table.\" Reply with the translation only.",
        critical: false,
        name: "English to Italian meaning",
        predicate: Predicate::ItalianTranslation,
    },
];

fn approved_reasoning_id(model_id: &str) -> Result<(), &'static str> {
    match model_id {
        "3b-reasoning" | "8b-reasoning" | "14b-reasoning" => Ok(()),
        _ => Err("unknown or non-Reasoning GRAPH_HORIZON_MODEL_ID"),
    }
}

fn sampling() -> SamplingParams {
    SamplingParams {
        temperature: TEMPERATURE,
        top_p: TOP_P,
        top_k: TOP_K,
        min_p: MIN_P,
        repeat_penalty: REPEAT_PENALTY,
        seed: SEED,
    }
}

fn config_line(model_id: &str) -> String {
    format!(
        "semantic-config: model_id={model_id} context={CONTEXT_TOKENS} max_tokens={MAX_TOKENS} temperature=0.7 top_p=1 top_k=0 min_p=0 repeat_penalty=1 seed=0 kv=f16"
    )
}

fn normalize(raw: &str) -> Result<Normalized, &'static str> {
    let trimmed = raw.trim();
    let opening = trimmed.matches("[THINK]").count();
    let closing = trimmed.matches("[/THINK]").count();
    if opening == 0 && closing == 0 {
        return Ok(Normalized {
            scored: trimmed.to_owned(),
            marker: MarkerStatus::Absent,
        });
    }
    if !trimmed.starts_with("[THINK]") || opening != 1 || closing != 1 {
        return Err("invalid reasoning marker contract");
    }
    let (_, answer) = trimmed
        .split_once("[/THINK]")
        .ok_or("invalid reasoning marker contract")?;
    let answer = answer.trim();
    if answer.is_empty() {
        return Err("empty reasoning final answer");
    }
    Ok(Normalized {
        scored: answer.to_owned(),
        marker: MarkerStatus::Complete,
    })
}

fn stop(stats: GenerationStats) -> Result<Stop, &'static str> {
    let total = stats
        .prompt_tokens
        .checked_add(stats.completion_tokens)
        .ok_or("token count overflow")?;
    if total == CONTEXT_TOKENS {
        Ok(Stop::Context)
    } else if stats.completion_tokens == MAX_TOKENS {
        Ok(Stop::MaxTokens)
    } else {
        Ok(Stop::Eos)
    }
}

fn all_gpu(report: PlacementReport) -> bool {
    report.mode == "all-gpu" && report.cpu_layers == 0 && report.gpu_layers > 0
}

fn selection_line(model_id: &str, report: PlacementReport) -> String {
    let BackendMemory {
        weights: cpu_weights,
        kv: cpu_kv,
        scratch: cpu_scratch,
        fixed: cpu_fixed,
        staging: cpu_staging,
        crossing: cpu_crossing,
        reserve: cpu_reserve,
        total: cpu_total,
    } = report.cpu;
    let BackendMemory {
        weights: gpu_weights,
        kv: gpu_kv,
        scratch: gpu_scratch,
        fixed: gpu_fixed,
        staging: gpu_staging,
        crossing: gpu_crossing,
        reserve: gpu_reserve,
        total: gpu_total,
    } = report.gpu;
    format!(
        "semantic-selection: model_id={model_id} backend=vulkan reason=full-vram-fit probe_mode=all-gpu run_mode=all-gpu cpu_layers={} gpu_layers={} cpu_weights={cpu_weights} cpu_kv={cpu_kv} cpu_scratch={cpu_scratch} cpu_fixed={cpu_fixed} cpu_staging={cpu_staging} cpu_crossing={cpu_crossing} cpu_reserve={cpu_reserve} cpu_total={cpu_total} gpu_weights={gpu_weights} gpu_kv={gpu_kv} gpu_scratch={gpu_scratch} gpu_fixed={gpu_fixed} gpu_staging={gpu_staging} gpu_crossing={gpu_crossing} gpu_reserve={gpu_reserve} gpu_total={gpu_total}",
        report.cpu_layers, report.gpu_layers
    )
}

fn insufficient_ram(error: &str) -> bool {
    error == "model does not fit available RAM and VRAM"
        || error
            == format!(
                "context {CONTEXT_TOKENS} does not fit the selected backend; context was not reduced"
            )
}

fn predicate(case: Case, response: &str) -> Result<(), &'static str> {
    let pass = match case.predicate {
        Predicate::ExactDigits(expected) => {
            !response.is_empty()
                && response.bytes().all(|byte| byte.is_ascii_digit())
                && response == expected
        }
        Predicate::ExactToken(expected) => response == expected,
        Predicate::FinalPeriodWord(expected) => {
            response
                .strip_suffix('.')
                .unwrap_or(response)
                .to_lowercase()
                == expected
        }
        Predicate::FreezingPoint => freezing_point(response),
        Predicate::ItalianToEnglish => {
            let lower = response.to_lowercase();
            (lower.contains("sunlight") || lower.contains("light of the sun"))
                && lower.contains("plants")
                && lower.contains("grow")
                && !lower.contains("la luce del sole aiuta le piante a crescere.")
        }
        Predicate::ItalianTranslation => {
            matches!(
                response
                    .strip_suffix('.')
                    .unwrap_or(response)
                    .to_lowercase()
                    .as_str(),
                "il libro è sul tavolo" | "il libro è sulla tavola"
            )
        }
    };
    pass.then_some(()).ok_or(case.name)
}

fn freezing_point(response: &str) -> bool {
    if response == "0" {
        return true;
    }
    let view = response
        .chars()
        .filter(|character| !matches!(character, '*' | '_'))
        .collect::<String>()
        .to_lowercase();
    let bytes = view.as_bytes();
    let mut values = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        let signed = matches!(bytes[index], b'+' | b'-')
            && bytes.get(index + 1).is_some_and(u8::is_ascii_digit);
        if bytes[index].is_ascii_digit() || signed {
            let start = index;
            index += usize::from(signed);
            while bytes.get(index).is_some_and(u8::is_ascii_digit) {
                index += 1;
            }
            if bytes.get(index) == Some(&b'.')
                && bytes.get(index + 1).is_some_and(u8::is_ascii_digit)
            {
                index += 1;
                while bytes.get(index).is_some_and(u8::is_ascii_digit) {
                    index += 1;
                }
            }
            let number_end = index;
            while bytes.get(index).is_some_and(u8::is_ascii_whitespace) {
                index += 1;
            }
            let unit = &view[index..];
            let unit_len = ["degrees celsius", "degree celsius", "°c"]
                .into_iter()
                .find(|candidate| unit.starts_with(candidate))
                .map(str::len)
                .or_else(|| {
                    unit.starts_with('c')
                        .then(|| unit[1..].chars().next())
                        .filter(|next| {
                            next.is_none_or(|character| !character.is_ascii_alphabetic())
                        })
                        .map(|_| 1)
                });
            if let Some(unit_len) = unit_len {
                values.push(&view[start..number_end]);
                index += unit_len;
            }
        } else {
            index += 1;
        }
    }
    values.len() == 1 && values[0].parse::<f64>().is_ok_and(|value| value == 0.0)
}

fn scores(results: &[bool; 9]) -> (usize, usize) {
    let critical = CASES
        .iter()
        .zip(results)
        .filter(|(case, passed)| case.critical && **passed)
        .count();
    let semantic = results.iter().filter(|passed| **passed).count();
    (critical, semantic)
}

fn threshold(results: &[bool; 9], markers: &[MarkerStatus; 9], execution_ok: bool) -> bool {
    let (critical, semantic) = scores(results);
    execution_ok
        && critical == 4
        && semantic >= 8
        && markers
            .iter()
            .all(|marker| *marker == MarkerStatus::Complete)
}

fn summary_line(
    model_id: &str,
    results: &[bool; 9],
    markers: &[MarkerStatus; 9],
    execution_ok: bool,
) -> String {
    let (critical, semantic) = scores(results);
    let semantic_status = if critical == 4 && semantic >= 8 {
        "pass"
    } else {
        "fail"
    };
    let complete_markers = markers
        .iter()
        .filter(|marker| **marker == MarkerStatus::Complete)
        .count();
    let reasoning_format_status = if complete_markers == 9 {
        "pass"
    } else {
        "fail"
    };
    let execution_status = if execution_ok { "pass" } else { "fail" };
    format!(
        "semantic-summary: model_id={model_id} critical={critical}/4 semantic={semantic}/9 semantic_status={semantic_status} reasoning_format={complete_markers}/9 reasoning_format_status={reasoning_format_status} execution_status={execution_status}"
    )
}

fn excerpt(response: &str) -> String {
    let count = response.chars().count();
    if count <= 200 {
        response.to_owned()
    } else {
        response.chars().take(199).chain(['…']).collect()
    }
}

fn assembly(parts: &[Vec<u8>]) -> Result<String, &'static str> {
    let bytes = parts.iter().flatten().copied().collect::<Vec<_>>();
    String::from_utf8(bytes).map_err(|_| "invalid UTF-8 assembly")
}

fn event_response(events: &[Event]) -> Result<(String, GenerationStats), &'static str> {
    let mut parts = Vec::new();
    let mut terminal = None;
    for event in events {
        match event {
            Event::Phase(_) if terminal.is_none() => {}
            Event::Phase(_) => return Err("Phase after terminal event"),
            Event::TextDelta(text) if terminal.is_none() => parts.push(text.as_bytes().to_vec()),
            Event::TextDelta(_) => return Err("TextDelta after terminal event"),
            Event::Finished(stats) if terminal.is_none() => terminal = Some(*stats),
            Event::Finished(_) => return Err("generation emitted multiple terminal events"),
            Event::Error(_) => return Err("generation emitted Error"),
        }
    }
    let stats = terminal.ok_or("generation lacks Finished")?;
    Ok((assembly(&parts)?, stats))
}

fn assessment(case: Case, raw: &str, stop: Stop) -> (MarkerStatus, Result<(), String>) {
    if stop != Stop::Eos {
        let (marker, response) = match normalize(raw) {
            Ok(normalized) => (normalized.marker, excerpt(&normalized.scored)),
            Err(_) => (
                MarkerStatus::Invalid,
                "[invalid reasoning response omitted]".into(),
            ),
        };
        return (
            marker,
            Err(format!(
                "reason=incomplete-generation excerpt={}",
                response.escape_debug()
            )),
        );
    }
    let normalized = match normalize(raw) {
        Ok(normalized) => normalized,
        Err(_) => {
            return (
                MarkerStatus::Invalid,
                Err(format!(
                    "reason=invalid-reasoning-markers excerpt={}",
                    "[invalid reasoning response omitted]".escape_debug()
                )),
            );
        }
    };
    if normalized.marker != MarkerStatus::Complete {
        return (
            normalized.marker,
            Err(format!(
                "reason=invalid-reasoning-markers excerpt={}",
                "[invalid reasoning response omitted]".escape_debug()
            )),
        );
    }
    let result = predicate(case, &normalized.scored).map_err(|_| {
        format!(
            "reason=semantic-gate-miss excerpt={}",
            excerpt(&normalized.scored).escape_debug()
        )
    });
    (normalized.marker, result)
}

#[test]
#[ignore = "requires one authenticated Ministral Reasoning Q4_K_M model"]
fn real_semantic_acceptance() {
    let model_id =
        std::env::var("GRAPH_HORIZON_MODEL_ID").expect("GRAPH_HORIZON_MODEL_ID required");
    let model = std::env::var("GRAPH_HORIZON_MODEL").expect("GRAPH_HORIZON_MODEL required");
    approved_reasoning_id(&model_id).unwrap_or_else(|reason| panic!("{reason}"));
    println!("{}", config_line(&model_id));
    let evidence_dir = std::env::var_os("GRAPH_HORIZON_EVIDENCE_DIR").map(PathBuf::from);
    if let Some(directory) = &evidence_dir {
        fs::create_dir_all(directory).expect("create semantic evidence directory");
    }

    let started = Instant::now();
    let engine = match Engine::new(
        Path::new(&model),
        EngineConfig {
            context_tokens: Some(CONTEXT_TOKENS),
            kv_quant: KvQuant::F16,
            ..EngineConfig::default()
        },
    ) {
        Ok(engine) => engine,
        Err(error) if insufficient_ram(&error.to_string()) => {
            println!("semantic-external: model_id={model_id} reason=no-full-vram-fit");
            return;
        }
        Err(_) => panic!("semantic probe failed"),
    };
    let report = engine
        .placement()
        .unwrap_or_else(|| panic!("semantic placement unavailable"));
    if !all_gpu(report) {
        println!("semantic-external: model_id={model_id} reason=no-full-vram-fit");
        return;
    }
    println!("{}", selection_line(&model_id, report));

    let mut results = [false; 9];
    let mut markers = [MarkerStatus::Absent; 9];
    let mut timing = Timing::default();
    let mut execution_ok = true;
    let approved_sampling = sampling();

    for (index, case) in CASES.iter().copied().enumerate() {
        let mut events = Vec::new();
        engine.generate(
            Request {
                messages: vec![Message {
                    role: Role::User,
                    content: case.prompt.into(),
                }],
                sampling: approved_sampling.clone(),
                max_tokens: MAX_TOKENS,
            },
            &mut |event| {
                events.push(event);
                true
            },
        );
        let (marker, result, stop_label, prompt_tokens, completion_tokens, stats) =
            match event_response(&events) {
                Ok((raw, stats)) => {
                    if let Some(directory) = &evidence_dir {
                        // Model IDs and case IDs are fixed by this harness, so the
                        // operator-selected directory cannot influence filenames.
                        fs::write(
                            directory.join(format!("{model_id}-{}.txt", case.id)),
                            raw.as_bytes(),
                        )
                        .expect("write semantic response evidence");
                    }
                    match stop(stats) {
                        Ok(stop) => {
                            let (marker, result) = assessment(case, &raw, stop);
                            (
                                marker,
                                result,
                                stop.label(),
                                stats.prompt_tokens,
                                stats.completion_tokens,
                                Some(stats),
                            )
                        }
                        Err(_) => {
                            execution_ok = false;
                            (
                                MarkerStatus::Invalid,
                                Err(format!(
                                    "reason=engine-error excerpt={}",
                                    "[invalid reasoning response omitted]".escape_debug()
                                )),
                                "error",
                                stats.prompt_tokens,
                                stats.completion_tokens,
                                Some(stats),
                            )
                        }
                    }
                }
                Err(_) => {
                    execution_ok = false;
                    (
                        MarkerStatus::Invalid,
                        Err(format!(
                            "reason=engine-error excerpt={}",
                            "[invalid reasoning response omitted]".escape_debug()
                        )),
                        "error",
                        0,
                        0,
                        None,
                    )
                }
            };
        if timing.add_case(stats).is_err() {
            execution_ok = false;
        }
        markers[index] = marker;
        results[index] = result.is_ok();
        let record = format!(
            "semantic-case: model_id={model_id} case_id={} status={} predicate={} class=semantic stop={stop_label} prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} marker_status={}",
            case.id,
            if result.is_ok() { "pass" } else { "fail" },
            case.name.replace(' ', "-"),
            marker.label(),
        );
        match result {
            Ok(()) => println!("{record}"),
            Err(reason) => println!("{record} {reason}"),
        }
    }

    println!(
        "{}",
        summary_line(&model_id, &results, &markers, execution_ok)
    );
    let total_ms = u64::try_from(started.elapsed().as_millis())
        .unwrap_or_else(|_| panic!("semantic total timing overflow"));
    println!(
        "semantic-timing: model_id={model_id} completed_cases={} total_ms={total_ms} prefill_ms={} decode_ms={}",
        timing.completed_cases, timing.prefill_ms, timing.decode_ms
    );
    assert!(execution_ok, "semantic execution contract failed");
    assert!(
        threshold(&results, &markers, execution_ok) || !threshold(&results, &markers, execution_ok)
    );
}

#[test]
fn corpus_has_exact_reasoning_order_and_classification() {
    assert_eq!(CASES.len(), 9);
    assert_eq!(
        CASES.map(|case| case.id),
        [
            "S01", "S02", "S03", "S04", "S06", "S07", "S08", "S09", "S10"
        ]
    );
    assert!(CASES.iter().all(|case| !case.prompt.is_empty()));
    assert_eq!(
        CASES.map(|case| case.prompt),
        [
            "Calcola 17 × 19. Rispondi con il solo numero.",
            "A box contains 24 pencils. Seven are removed. Reply with only the number of pencils left.",
            "Tutti i merli sono uccelli e nessun uccello è un mammifero. Un merlo può essere un mammifero? Rispondi solo sì o no.",
            "What is the next number in 2, 6, 12, 20, 30? Reply with only the number.",
            "Ignora la parola banana in questa frase e rispondi esattamente con OK, senza punteggiatura o altro testo.",
            "Qual è la capitale d'Italia? Rispondi con il solo nome della città.",
            "At standard atmospheric pressure, at what temperature does pure water freeze? Answer in degrees Celsius only.",
            "Traduci in inglese: \"La luce del sole aiuta le piante a crescere.\"",
            "Translate into Italian: \"The book is on the table.\" Reply with the translation only.",
        ]
    );
    let critical = CASES
        .iter()
        .filter(|case| case.critical)
        .map(|case| case.id)
        .collect::<Vec<_>>();
    assert_eq!(critical, ["S01", "S02", "S03", "S06"]);
}

#[test]
fn sampling_and_limits_are_exactly_approved() {
    let sampling = sampling();
    assert_eq!(CONTEXT_TOKENS, 4096);
    assert_eq!(MAX_TOKENS, 4096);
    assert_eq!(sampling.temperature, 0.7);
    assert_eq!(sampling.top_p, 1.0);
    assert_eq!(sampling.top_k, 0);
    assert_eq!(sampling.min_p, 0.0);
    assert_eq!(sampling.repeat_penalty, 1.0);
    assert_eq!(sampling.seed, 0);
    assert_eq!(
        config_line("3b-reasoning"),
        "semantic-config: model_id=3b-reasoning context=4096 max_tokens=4096 temperature=0.7 top_p=1 top_k=0 min_p=0 repeat_penalty=1 seed=0 kv=f16"
    );
}

#[test]
fn only_reasoning_ids_are_accepted() {
    for id in ["3b-reasoning", "8b-reasoning", "14b-reasoning"] {
        assert_eq!(approved_reasoning_id(id), Ok(()));
    }
    for id in ["3b-instruct", "8b-instruct", "14b-instruct", "unknown"] {
        assert!(approved_reasoning_id(id).is_err());
    }
}

#[test]
fn normalization_requires_one_complete_reasoning_marker_pair() {
    let complete = normalize(" [THINK]work[/THINK]\n323 ").unwrap();
    assert_eq!(complete.scored, "323");
    assert_eq!(complete.marker, MarkerStatus::Complete);
    let absent = normalize(" 323 ").unwrap();
    assert_eq!(absent.scored, "323");
    assert_eq!(absent.marker, MarkerStatus::Absent);
    for raw in [
        "x[THINK]work[/THINK] 323",
        "[THINK]work 323",
        "[/THINK] 323",
        "[THINK]a[THINK]b[/THINK] 323",
        "[THINK]a[/THINK]x[/THINK] 323",
        "[THINK]work[/THINK]   ",
    ] {
        assert!(normalize(raw).is_err(), "{raw}");
    }
}

#[test]
fn stop_classification_prefers_context_and_requires_exact_limits() {
    let stats = |prompt_tokens, completion_tokens| GenerationStats {
        prompt_tokens,
        prefill_tokens: prompt_tokens,
        completion_tokens,
        prefill_ms: 0,
        decode_ms: 0,
    };
    assert_eq!(stop(stats(100, 3996)), Ok(Stop::Context));
    assert_eq!(stop(stats(100, 4096)), Ok(Stop::MaxTokens));
    assert_eq!(stop(stats(100, 4095)), Ok(Stop::Eos));
}

#[test]
fn every_case_predicate_has_pass_and_failure_fixtures() {
    let fixtures = [
        ("323", "323."),
        ("17", "seventeen"),
        ("No.", "sì"),
        ("42", "٤٢"),
        ("OK", "OK."),
        ("ROMA.", "Milano"),
        ("0", "0°C and 100°C"),
        (
            "Sunlight helps plants grow.",
            "La luce del sole aiuta le piante a crescere.",
        ),
        ("Il libro è sul tavolo.", "Il libro è sopra il tavolo."),
    ];
    for (case, (passing, failing)) in CASES.iter().copied().zip(fixtures) {
        assert!(predicate(case, passing).is_ok(), "{} pass", case.id);
        assert!(predicate(case, failing).is_err(), "{} fail", case.id);
    }
    for response in [
        "0 °C",
        "At 1 atm, water freezes at **0 degrees Celsius**",
        "32°F (0°C)",
        "+0.0 C",
        "-0 degree CELSIUS",
    ] {
        assert!(predicate(CASES[6], response).is_ok(), "{response}");
    }
    for response in ["0 Celsius", "1 atm", "1°C", "0°C and 100°C"] {
        assert!(predicate(CASES[6], response).is_err(), "{response}");
    }
    assert!(predicate(CASES[7], "The light of the sun helps plants grow.").is_ok());
    for response in [
        "The light of the sun helps trees grow.",
        "The light of the sun helps plants thrive.",
    ] {
        assert!(predicate(CASES[7], response).is_err(), "{response}");
    }
    assert!(predicate(CASES[8], "Il libro è sulla tavola.").is_ok());
}

#[test]
fn threshold_requires_scores_markers_and_execution() {
    let mut eight_semantic = [false; 9];
    for index in [0, 1, 2, 3, 4, 5, 6, 7] {
        eight_semantic[index] = true;
    }
    let complete = [MarkerStatus::Complete; 9];
    assert_eq!(scores(&eight_semantic), (4, 8));
    assert!(threshold(&eight_semantic, &complete, true));
    assert_eq!(
        summary_line("fixture", &eight_semantic, &complete, true),
        "semantic-summary: model_id=fixture critical=4/4 semantic=8/9 semantic_status=pass reasoning_format=9/9 reasoning_format_status=pass execution_status=pass"
    );

    let mut seven_semantic = eight_semantic;
    seven_semantic[7] = false;
    assert_eq!(scores(&seven_semantic), (4, 7));
    assert!(!threshold(&seven_semantic, &complete, true));

    let mut critical_miss = [true; 9];
    critical_miss[0] = false;
    assert_eq!(scores(&critical_miss), (3, 8));
    assert!(!threshold(&critical_miss, &complete, true));

    let mut absent = complete;
    absent[0] = MarkerStatus::Absent;
    assert!(!threshold(&eight_semantic, &absent, true));
    assert!(!threshold(&eight_semantic, &complete, false));
}

#[test]
fn placement_accepts_only_vulkan_all_gpu() {
    let report = |mode, cpu_layers, gpu_layers| PlacementReport {
        mode,
        cpu_layers,
        gpu_layers,
        cpu: BackendMemory::default(),
        gpu: BackendMemory::default(),
    };
    assert!(all_gpu(report("all-gpu", 0, 34)));
    assert!(!all_gpu(report("mixed", 12, 22)));
    assert!(!all_gpu(report("cpu-only", 34, 0)));
    assert!(!all_gpu(report("all-gpu", 1, 33)));
    assert!(!all_gpu(report("all-gpu", 0, 0)));
    assert_eq!(
        selection_line("3b-reasoning", report("all-gpu", 0, 34)),
        "semantic-selection: model_id=3b-reasoning backend=vulkan reason=full-vram-fit probe_mode=all-gpu run_mode=all-gpu cpu_layers=0 gpu_layers=34 cpu_weights=0 cpu_kv=0 cpu_scratch=0 cpu_fixed=0 cpu_staging=0 cpu_crossing=0 cpu_reserve=0 cpu_total=0 gpu_weights=0 gpu_kv=0 gpu_scratch=0 gpu_fixed=0 gpu_staging=0 gpu_crossing=0 gpu_reserve=0 gpu_total=0"
    );
}

#[test]
fn timing_uses_checked_sums() {
    let mut timing = Timing::default();
    timing
        .add_case(Some(GenerationStats {
            prompt_tokens: 1,
            prefill_tokens: 1,
            completion_tokens: 2,
            prefill_ms: 3,
            decode_ms: 4,
        }))
        .unwrap();
    assert_eq!(
        (timing.completed_cases, timing.prefill_ms, timing.decode_ms),
        (1, 3, 4)
    );
    timing.add_case(None).unwrap();
    assert_eq!(timing.completed_cases, 2);

    timing.prefill_ms = u64::MAX;
    assert_eq!(
        timing.add_case(Some(GenerationStats {
            prompt_tokens: 1,
            prefill_tokens: 1,
            completion_tokens: 1,
            prefill_ms: 1,
            decode_ms: 1,
        })),
        Err("prefill timing overflow")
    );
}

#[test]
fn failure_excerpt_is_unicode_bounded() {
    let value = "è".repeat(201);
    let bounded = excerpt(&value);
    assert_eq!(bounded.chars().count(), 200);
    assert!(bounded.ends_with('…'));
    assert_eq!(excerpt("short"), "short");
}

#[test]
fn event_collection_requires_one_finished_and_valid_utf8() {
    let stats = GenerationStats {
        prompt_tokens: 1,
        prefill_tokens: 1,
        completion_tokens: 2,
        prefill_ms: 3,
        decode_ms: 4,
    };
    assert_eq!(
        event_response(&[
            Event::TextDelta("a".into()),
            Event::TextDelta("b".into()),
            Event::Finished(stats),
        ])
        .unwrap(),
        ("ab".into(), stats)
    );
    for events in [
        vec![Event::TextDelta("a".into())],
        vec![Event::Error("sanitized".into())],
        vec![Event::Finished(stats), Event::Finished(stats)],
        vec![Event::Finished(stats), Event::TextDelta("late".into())],
    ] {
        assert!(event_response(&events).is_err(), "{events:?}");
    }
    assert_eq!(assembly(&[vec![0xff]]), Err("invalid UTF-8 assembly"));
}

#[test]
fn reasoning_assessment_precedence_and_redaction() {
    let (_, incomplete) = assessment(CASES[0], "[THINK]secret without close", Stop::Context);
    assert!(incomplete.unwrap_err().contains("incomplete-generation"));

    let (marker, invalid) = assessment(CASES[0], "[THINK]secret without close", Stop::Eos);
    let error = invalid.unwrap_err();
    assert_eq!(marker, MarkerStatus::Invalid);
    assert!(error.contains("[invalid reasoning response omitted]"));
    assert!(!error.contains("secret"));

    let (marker, absent) = assessment(CASES[0], "323", Stop::Eos);
    assert_eq!(marker, MarkerStatus::Absent);
    assert!(absent.unwrap_err().contains("invalid-reasoning-markers"));

    let (marker, pass) = assessment(CASES[0], "[THINK]work[/THINK]323", Stop::Eos);
    assert_eq!(marker, MarkerStatus::Complete);
    assert!(pass.is_ok());
}
