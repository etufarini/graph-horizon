/*
 * gh_zero_engine — M3 all-GPU/CPU-only semantic acceptance harness
 * Selects one final backend from the observed hybrid placement, then owns the
 * fixed corpus, profile-specific context, scoring, class-sensitive stop gating,
 * marker diagnostics, placement, and timing. A mixed probe never generates;
 * runtime output and production hybrid policy remain outside this test-only
 * file.
 */

use std::{path::Path, time::Instant};

use gh_zero_engine::{
    BackendMemory, Engine, EngineConfig, Event, GenerationStats, KvQuant, Message, PlacementReport,
    Request, Role, SamplingParams,
};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Profile {
    Instruct,
    Reasoning,
}

impl Profile {
    fn context_tokens(self) -> usize {
        match self {
            Self::Instruct => 4096,
            Self::Reasoning => 32768,
        }
    }

    fn max_tokens(self) -> usize {
        match self {
            Self::Instruct => 256,
            Self::Reasoning => 32768,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MarkerStatus {
    NotApplicable,
    Complete,
    Absent,
    Invalid,
}

impl MarkerStatus {
    fn label(self) -> &'static str {
        match self {
            Self::NotApplicable => "not-applicable",
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SemanticBackend {
    Vulkan,
    Cpu,
}

impl SemanticBackend {
    fn label(self) -> &'static str {
        match self {
            Self::Vulkan => "vulkan",
            Self::Cpu => "cpu",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SemanticSelection {
    backend: SemanticBackend,
    reason: &'static str,
    probe_mode: &'static str,
}

#[derive(Default)]
struct Timing {
    completed_cases: u8,
    prefill_ms: u64,
    decode_ms: u64,
}

impl Timing {
    fn add(&mut self, stats: GenerationStats) -> Result<(), &'static str> {
        let prefill_ms = self
            .prefill_ms
            .checked_add(stats.prefill_ms)
            .ok_or("prefill timing overflow")?;
        let decode_ms = self
            .decode_ms
            .checked_add(stats.decode_ms)
            .ok_or("decode timing overflow")?;
        self.prefill_ms = prefill_ms;
        self.decode_ms = decode_ms;
        self.completed_cases += 1;
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum CaseClass {
    CriticalSemantic,
    Semantic,
    Conformance,
}

impl CaseClass {
    fn label(self) -> &'static str {
        match self {
            Self::CriticalSemantic | Self::Semantic => "semantic",
            Self::Conformance => "conformance",
        }
    }
}

#[derive(Clone, Copy)]
enum Predicate {
    ExactDigits(&'static str),
    ExactToken(&'static str),
    FinalPeriodWord(&'static str),
    ThreeLines,
    FreezingPoint,
    ItalianToEnglish,
    ItalianTranslation,
    MarkdownColors,
    JsonObject,
}

#[derive(Clone, Copy)]
struct Case {
    id: &'static str,
    prompt: &'static str,
    class: CaseClass,
    name: &'static str,
    predicate: Predicate,
}

const CASES: [Case; 12] = [
    Case {
        id: "S01",
        prompt: "Calcola 17 × 19. Rispondi con il solo numero.",
        class: CaseClass::CriticalSemantic,
        name: "exact Italian arithmetic",
        predicate: Predicate::ExactDigits("323"),
    },
    Case {
        id: "S02",
        prompt: "A box contains 24 pencils. Seven are removed. Reply with only the number of pencils left.",
        class: CaseClass::CriticalSemantic,
        name: "exact English arithmetic",
        predicate: Predicate::ExactDigits("17"),
    },
    Case {
        id: "S03",
        prompt: "Tutti i merli sono uccelli e nessun uccello è un mammifero. Un merlo può essere un mammifero? Rispondi solo sì o no.",
        class: CaseClass::CriticalSemantic,
        name: "Italian deduction",
        predicate: Predicate::FinalPeriodWord("no"),
    },
    Case {
        id: "S04",
        prompt: "What is the next number in 2, 6, 12, 20, 30? Reply with only the number.",
        class: CaseClass::Semantic,
        name: "exact sequence answer",
        predicate: Predicate::ExactDigits("42"),
    },
    Case {
        id: "S05",
        prompt: "Write exactly three lines. First line: Alpha. Second line: Beta. Third line: Gamma. Add nothing else.",
        class: CaseClass::Conformance,
        name: "exact three lines",
        predicate: Predicate::ThreeLines,
    },
    Case {
        id: "S06",
        prompt: "Ignora la parola banana in questa frase e rispondi esattamente con OK, senza punteggiatura o altro testo.",
        class: CaseClass::CriticalSemantic,
        name: "exact OK token",
        predicate: Predicate::ExactToken("OK"),
    },
    Case {
        id: "S07",
        prompt: "Qual è la capitale d'Italia? Rispondi con il solo nome della città.",
        class: CaseClass::Semantic,
        name: "Italian capital",
        predicate: Predicate::FinalPeriodWord("roma"),
    },
    Case {
        id: "S08",
        prompt: "At standard atmospheric pressure, at what temperature does pure water freeze? Answer in degrees Celsius only.",
        class: CaseClass::Semantic,
        name: "freezing point",
        predicate: Predicate::FreezingPoint,
    },
    Case {
        id: "S09",
        prompt: "Traduci in inglese: \"La luce del sole aiuta le piante a crescere.\"",
        class: CaseClass::Semantic,
        name: "Italian to English meaning",
        predicate: Predicate::ItalianToEnglish,
    },
    Case {
        id: "S10",
        prompt: "Translate into Italian: \"The book is on the table.\" Reply with the translation only.",
        class: CaseClass::Semantic,
        name: "English to Italian meaning",
        predicate: Predicate::ItalianTranslation,
    },
    Case {
        id: "S11",
        prompt: "Return a Markdown unordered list containing exactly these colors in this order: red, green, blue. No introduction or conclusion.",
        class: CaseClass::Conformance,
        name: "Markdown color list",
        predicate: Predicate::MarkdownColors,
    },
    Case {
        id: "S12",
        prompt: "Return only a JSON object with exactly two fields: result set to the number 42, and unit set to the string \"items\".",
        class: CaseClass::Conformance,
        name: "exact JSON object",
        predicate: Predicate::JsonObject,
    },
];

fn normalize(profile: Profile, raw: &str) -> Result<Normalized, &'static str> {
    let trimmed = raw.trim();
    match profile {
        Profile::Instruct => {
            if trimmed.contains("[THINK]") || trimmed.contains("[/THINK]") {
                return Err("reasoning marker in Instruct response");
            }
            Ok(Normalized {
                scored: trimmed.to_owned(),
                marker: MarkerStatus::NotApplicable,
            })
        }
        Profile::Reasoning => {
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
    }
}

fn stop(
    context_tokens: usize,
    max_tokens: usize,
    stats: GenerationStats,
) -> Result<Stop, &'static str> {
    let total = stats
        .prompt_tokens
        .checked_add(stats.completion_tokens)
        .ok_or("token count overflow")?;
    if total == context_tokens {
        Ok(Stop::Context)
    } else if stats.completion_tokens == max_tokens {
        Ok(Stop::MaxTokens)
    } else {
        Ok(Stop::Eos)
    }
}

fn model_profile(model_id: &str) -> Result<Profile, &'static str> {
    match model_id {
        "3b-instruct" | "8b-instruct" | "14b-instruct" => Ok(Profile::Instruct),
        "3b-reasoning" | "8b-reasoning" | "14b-reasoning" => Ok(Profile::Reasoning),
        _ => Err("unknown GH_ZERO_MODEL_ID"),
    }
}

fn semantic_selection(report: PlacementReport) -> Result<SemanticSelection, &'static str> {
    match report.mode {
        "all-gpu" => Ok(SemanticSelection {
            backend: SemanticBackend::Vulkan,
            reason: "full-vram-fit",
            probe_mode: "all-gpu",
        }),
        "mixed" => Ok(SemanticSelection {
            backend: SemanticBackend::Cpu,
            reason: "no-full-vram-fit",
            probe_mode: "mixed",
        }),
        "cpu-only" => Ok(SemanticSelection {
            backend: SemanticBackend::Cpu,
            reason: "no-full-vram-fit",
            probe_mode: "cpu-only",
        }),
        _ => Err("unknown placement mode"),
    }
}

fn validate_final(
    selection: SemanticSelection,
    report: PlacementReport,
) -> Result<(), &'static str> {
    match selection.backend {
        SemanticBackend::Vulkan if report.mode == "all-gpu" && report.cpu_layers == 0 => Ok(()),
        SemanticBackend::Cpu if report.mode == "cpu-only" && report.gpu_layers == 0 => Ok(()),
        _ => Err("final placement contradicts selected backend"),
    }
}

fn selection_line(model_id: &str, selection: SemanticSelection, report: PlacementReport) -> String {
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
        "semantic-selection: model_id={model_id} backend={} reason={} probe_mode={} run_mode={} cpu_layers={} gpu_layers={} cpu_weights={cpu_weights} cpu_kv={cpu_kv} cpu_scratch={cpu_scratch} cpu_fixed={cpu_fixed} cpu_staging={cpu_staging} cpu_crossing={cpu_crossing} cpu_reserve={cpu_reserve} cpu_total={cpu_total} gpu_weights={gpu_weights} gpu_kv={gpu_kv} gpu_scratch={gpu_scratch} gpu_fixed={gpu_fixed} gpu_staging={gpu_staging} gpu_crossing={gpu_crossing} gpu_reserve={gpu_reserve} gpu_total={gpu_total}",
        selection.backend.label(),
        selection.reason,
        selection.probe_mode,
        report.mode,
        report.cpu_layers,
        report.gpu_layers,
    )
}

fn performance(
    model_id: &str,
    backend: SemanticBackend,
    total_ms: u64,
) -> (&'static str, &'static str, bool) {
    if model_id == "3b-instruct" && backend == SemanticBackend::Vulkan {
        let passed = total_ms < 1_506_690;
        ("1506690", if passed { "pass" } else { "fail" }, passed)
    } else {
        ("not-applicable", "not-applicable", true)
    }
}

fn insufficient_ram(error: &str, context_tokens: usize) -> bool {
    error == "model does not fit available RAM and VRAM"
        || error
            == format!(
                "context {context_tokens} does not fit the selected backend; context was not reduced"
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
        Predicate::ThreeLines => {
            let normalized = response.replace("\r\n", "\n");
            normalized.split('\n').collect::<Vec<_>>() == ["Alpha", "Beta", "Gamma"]
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
        Predicate::MarkdownColors => {
            let lines = response.split('\n').collect::<Vec<_>>();
            lines.len() == 3
                && lines.iter().all(|line| !line.is_empty())
                && lines
                    .iter()
                    .filter_map(|line| line.strip_prefix("- "))
                    .map(str::to_lowercase)
                    .collect::<Vec<_>>()
                    == ["red", "green", "blue"]
        }
        Predicate::JsonObject => exact_json(response),
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

fn exact_json(response: &str) -> bool {
    let Ok(Value::Object(object)) = serde_json::from_str::<Value>(response) else {
        return false;
    };
    // Values are fixed, so repeated literal key spellings can only be duplicate keys.
    let one_result = response.matches("\"result\"").count() == 1;
    let one_unit = response.matches("\"unit\"").count() == 1;
    object.len() == 2
        && one_result
        && one_unit
        && object.get("result") == Some(&Value::from(42))
        && object.get("unit") == Some(&Value::from("items"))
}

fn scores(results: &[bool; 12]) -> (usize, usize, usize) {
    let passed = CASES.iter().zip(results).filter(|(_, passed)| **passed);
    let critical = passed
        .clone()
        .filter(|(case, _)| matches!(case.class, CaseClass::CriticalSemantic))
        .count();
    let semantic = passed
        .clone()
        .filter(|(case, _)| !matches!(case.class, CaseClass::Conformance))
        .count();
    let conformance = passed
        .filter(|(case, _)| matches!(case.class, CaseClass::Conformance))
        .count();
    (critical, semantic, conformance)
}

fn threshold(results: &[bool; 12]) -> bool {
    let (critical, semantic, _) = scores(results);
    critical == 4 && semantic >= 8
}

fn summary_line(
    model_id: &str,
    profile: Profile,
    backend: SemanticBackend,
    results: &[bool; 12],
    markers: &[MarkerStatus; 12],
) -> String {
    let (critical, semantic, conformance) = scores(results);
    let status = if threshold(results) { "pass" } else { "fail" };
    let (reasoning_format, reasoning_format_status) = match profile {
        Profile::Instruct => ("not-applicable".into(), "not-applicable"),
        Profile::Reasoning => (
            format!(
                "{}/12",
                markers
                    .iter()
                    .filter(|marker| **marker == MarkerStatus::Complete)
                    .count()
            ),
            "diagnostic",
        ),
    };
    format!(
        "semantic-summary: model_id={model_id} backend={} critical={critical}/4 semantic={semantic}/9 semantic_status={status} conformance={conformance}/3 conformance_status=diagnostic reasoning_format={reasoning_format} reasoning_format_status={reasoning_format_status}",
        backend.label()
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

fn assessment(
    profile: Profile,
    case: Case,
    raw: &str,
    stop: Stop,
) -> (MarkerStatus, Result<(), String>) {
    let normalized = match normalize(profile, raw) {
        Ok(normalized) => normalized,
        Err(reason) => {
            let (marker, response) = match profile {
                Profile::Instruct => (MarkerStatus::NotApplicable, excerpt(raw)),
                Profile::Reasoning => (
                    MarkerStatus::Invalid,
                    "[invalid reasoning response omitted]".into(),
                ),
            };
            return (
                marker,
                Err(format!(
                    "reason={} excerpt={}",
                    reason.replace(' ', "-"),
                    response.escape_debug()
                )),
            );
        }
    };
    if stop != Stop::Eos {
        return (
            normalized.marker,
            Err(format!(
                "reason=incomplete-generation excerpt={}",
                excerpt(&normalized.scored).escape_debug()
            )),
        );
    }
    let result = predicate(case, &normalized.scored).map_err(|reason| {
        format!(
            "reason={} excerpt={}",
            reason.replace(' ', "-"),
            excerpt(&normalized.scored).escape_debug()
        )
    });
    (normalized.marker, result)
}

#[test]
#[ignore = "requires one authenticated Ministral Q4_K_M model"]
fn real_semantic_acceptance() {
    let model_id = std::env::var("GH_ZERO_MODEL_ID").expect("GH_ZERO_MODEL_ID required");
    let model = std::env::var("GH_ZERO_MODEL").expect("GH_ZERO_MODEL required");
    // Both environment values and the exact profile ID are validated before construction.
    let profile = model_profile(&model_id).unwrap_or_else(|reason| panic!("{reason}"));
    let context_tokens = profile.context_tokens();
    let started = Instant::now();
    let probe = match Engine::new(
        Path::new(&model),
        EngineConfig {
            context_tokens: Some(context_tokens),
            kv_quant: KvQuant::F16,
            ..EngineConfig::default()
        },
    ) {
        Ok(engine) => engine,
        Err(error) if insufficient_ram(&error.to_string(), context_tokens) => {
            println!("semantic-external: model_id={model_id} reason=insufficient RAM");
            return;
        }
        Err(_) => panic!("semantic probe failed"),
    };
    let probe_report = probe
        .placement()
        .unwrap_or_else(|| panic!("semantic placement unavailable"));
    let selection = semantic_selection(probe_report).unwrap_or_else(|reason| panic!("{reason}"));

    // A mixed probe owns no request: it is dropped before the CPU-only reopen.
    let engine = if selection.probe_mode == "mixed" {
        drop(probe);
        match Engine::new(
            Path::new(&model),
            EngineConfig {
                context_tokens: Some(context_tokens),
                vram_weights_percent: Some(0),
                kv_quant: KvQuant::F16,
                ..EngineConfig::default()
            },
        ) {
            Ok(engine) => engine,
            Err(error) if insufficient_ram(&error.to_string(), context_tokens) => {
                println!("semantic-external: model_id={model_id} reason=insufficient RAM");
                return;
            }
            Err(_) => panic!("semantic CPU reopen failed"),
        }
    } else {
        probe
    };
    let final_report = engine
        .placement()
        .unwrap_or_else(|| panic!("semantic placement unavailable"));
    validate_final(selection, final_report).unwrap_or_else(|reason| panic!("{reason}"));
    println!("{}", selection_line(&model_id, selection, final_report));

    let mut results = [false; 12];
    let mut markers = [MarkerStatus::NotApplicable; 12];
    let mut timing = Timing::default();
    let mut execution_ok = true;
    let max_tokens = profile.max_tokens();

    for (index, case) in CASES.iter().copied().enumerate() {
        let mut events = Vec::new();
        engine.generate(
            Request {
                messages: vec![Message {
                    role: Role::User,
                    content: case.prompt.into(),
                }],
                sampling: SamplingParams {
                    temperature: 0.0,
                    top_p: 1.0,
                    top_k: 1,
                    min_p: 0.0,
                    repeat_penalty: 1.0,
                    seed: 0,
                },
                max_tokens,
            },
            &mut |event| {
                events.push(event);
                true
            },
        );
        let (marker, result, stop_label, prompt_tokens, completion_tokens) =
            match event_response(&events) {
                Ok((raw, stats)) => match stop(context_tokens, max_tokens, stats) {
                    Ok(stop) => {
                        let (marker, mut result) = assessment(profile, case, &raw, stop);
                        // Conformance scores are diagnostic; engine integrity is not.
                        if stop != Stop::Eos && !matches!(case.class, CaseClass::Conformance) {
                            execution_ok = false;
                        }
                        if let Err(reason) = timing.add(stats) {
                            execution_ok = false;
                            result = Err(format!(
                                "reason={} excerpt={}",
                                reason.replace(' ', "-"),
                                "".escape_debug()
                            ));
                        }
                        (
                            marker,
                            result,
                            stop.label(),
                            stats.prompt_tokens,
                            stats.completion_tokens,
                        )
                    }
                    Err(reason) => {
                        execution_ok = false;
                        (
                            MarkerStatus::Invalid,
                            Err(format!(
                                "reason={} excerpt={}",
                                reason.replace(' ', "-"),
                                "[invalid reasoning response omitted]".escape_debug()
                            )),
                            "error",
                            stats.prompt_tokens,
                            stats.completion_tokens,
                        )
                    }
                },
                Err(reason) => {
                    execution_ok = false;
                    let (marker, response) = match profile {
                        Profile::Instruct => (MarkerStatus::NotApplicable, ""),
                        Profile::Reasoning => (
                            MarkerStatus::Invalid,
                            "[invalid reasoning response omitted]",
                        ),
                    };
                    (
                        marker,
                        Err(format!(
                            "reason={} excerpt={}",
                            reason.replace(' ', "-"),
                            response.escape_debug()
                        )),
                        "error",
                        0,
                        0,
                    )
                }
            };
        markers[index] = marker;
        results[index] = result.is_ok();
        let record = format!(
            "semantic-case: model_id={model_id} case_id={} status={} predicate={} class={} stop={stop_label} prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} marker_status={}",
            case.id,
            if result.is_ok() { "pass" } else { "fail" },
            case.name.replace(' ', "-"),
            case.class.label(),
            marker.label(),
        );
        match result {
            Ok(()) => println!("{record}"),
            Err(reason) => println!("{record} {reason}"),
        }
    }

    let passed = threshold(&results);
    println!(
        "{}",
        summary_line(&model_id, profile, selection.backend, &results, &markers)
    );
    let total_ms = u64::try_from(started.elapsed().as_millis())
        .unwrap_or_else(|_| panic!("semantic total timing overflow"));
    let (baseline, performance_status, performance_ok) =
        performance(&model_id, selection.backend, total_ms);
    println!(
        "semantic-timing: model_id={model_id} backend={} completed_cases={} total_ms={total_ms} prefill_ms={} decode_ms={} baseline_cpu_ms={baseline} performance_status={performance_status}",
        selection.backend.label(),
        timing.completed_cases,
        timing.prefill_ms,
        timing.decode_ms,
    );
    assert!(execution_ok, "semantic execution contract failed");
    assert!(passed, "semantic acceptance threshold missed");
    assert!(performance_ok, "semantic performance criterion missed");
}

#[test]
fn corpus_has_exact_order_and_classification() {
    assert_eq!(CASES.len(), 12);
    for (index, case) in CASES.iter().enumerate() {
        assert_eq!(case.id, format!("S{:02}", index + 1));
        assert!(!case.prompt.is_empty());
    }
    assert_eq!(
        CASES.map(|case| case.prompt),
        [
            "Calcola 17 × 19. Rispondi con il solo numero.",
            "A box contains 24 pencils. Seven are removed. Reply with only the number of pencils left.",
            "Tutti i merli sono uccelli e nessun uccello è un mammifero. Un merlo può essere un mammifero? Rispondi solo sì o no.",
            "What is the next number in 2, 6, 12, 20, 30? Reply with only the number.",
            "Write exactly three lines. First line: Alpha. Second line: Beta. Third line: Gamma. Add nothing else.",
            "Ignora la parola banana in questa frase e rispondi esattamente con OK, senza punteggiatura o altro testo.",
            "Qual è la capitale d'Italia? Rispondi con il solo nome della città.",
            "At standard atmospheric pressure, at what temperature does pure water freeze? Answer in degrees Celsius only.",
            "Traduci in inglese: \"La luce del sole aiuta le piante a crescere.\"",
            "Translate into Italian: \"The book is on the table.\" Reply with the translation only.",
            "Return a Markdown unordered list containing exactly these colors in this order: red, green, blue. No introduction or conclusion.",
            "Return only a JSON object with exactly two fields: result set to the number 42, and unit set to the string \"items\".",
        ]
    );
    let classes = CASES.map(|case| case.class.label());
    assert_eq!(
        classes,
        [
            "semantic",
            "semantic",
            "semantic",
            "semantic",
            "conformance",
            "semantic",
            "semantic",
            "semantic",
            "semantic",
            "semantic",
            "conformance",
            "conformance",
        ]
    );
    let critical = CASES
        .iter()
        .filter(|case| matches!(case.class, CaseClass::CriticalSemantic))
        .map(|case| case.id)
        .collect::<Vec<_>>();
    assert_eq!(critical, ["S01", "S02", "S03", "S06"]);
}

#[test]
fn normalization_enforces_profile_marker_contracts() {
    let instruct = normalize(Profile::Instruct, " 323 \n").unwrap();
    assert_eq!(instruct.scored, "323");
    assert_eq!(instruct.marker, MarkerStatus::NotApplicable);
    for raw in ["[THINK]x[/THINK] 323", "x [/THINK]"] {
        assert!(normalize(Profile::Instruct, raw).is_err(), "{raw}");
    }
    let complete = normalize(Profile::Reasoning, " [THINK]work[/THINK]\n323 ").unwrap();
    assert_eq!(complete.scored, "323");
    assert_eq!(complete.marker, MarkerStatus::Complete);
    let absent = normalize(Profile::Reasoning, " 323 ").unwrap();
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
        assert!(normalize(Profile::Reasoning, raw).is_err(), "{raw}");
    }
}

#[test]
fn model_ids_select_only_the_approved_profiles() {
    for id in ["3b-instruct", "8b-instruct", "14b-instruct"] {
        assert_eq!(model_profile(id), Ok(Profile::Instruct));
    }
    for id in ["3b-reasoning", "8b-reasoning", "14b-reasoning"] {
        assert_eq!(model_profile(id), Ok(Profile::Reasoning));
    }
    assert!(model_profile("unknown").is_err());
    assert_eq!(Profile::Instruct.context_tokens(), 4096);
    assert_eq!(Profile::Reasoning.context_tokens(), 32768);
    assert_eq!(Profile::Instruct.max_tokens(), 256);
    assert_eq!(Profile::Reasoning.max_tokens(), 32768);
}

#[test]
fn stop_classification_prefers_context_and_requires_exact_limits() {
    let stats = |prompt_tokens, completion_tokens| GenerationStats {
        prompt_tokens,
        completion_tokens,
        prefill_ms: 0,
        decode_ms: 0,
    };
    assert_eq!(stop(4096, 4096, stats(100, 3996)), Ok(Stop::Context));
    assert_eq!(stop(4096, 256, stats(100, 256)), Ok(Stop::MaxTokens));
    assert_eq!(stop(4096, 256, stats(100, 255)), Ok(Stop::Eos));
    assert_eq!(stop(32768, 32768, stats(1, 32767)), Ok(Stop::Context));
}

#[test]
fn every_case_predicate_has_pass_and_failure_fixtures() {
    let fixtures = [
        ("323", "323."),
        ("17", "seventeen"),
        ("No.", "sì"),
        ("42", "٤٢"),
        ("Alpha\r\nBeta\r\nGamma", "Alpha\nBeta\nGamma\n"),
        ("OK", "OK."),
        ("ROMA.", "Milano"),
        ("0", "0°C and 100°C"),
        (
            "Sunlight helps plants grow.",
            "La luce del sole aiuta le piante a crescere.",
        ),
        ("Il libro è sul tavolo.", "Il libro è sopra il tavolo."),
        ("- red\n- green\n- blue", "- red\n- blue\n- green"),
        (
            r#"{"result":42,"unit":"items"}"#,
            r#"{"result":42,"unit":"items","extra":true}"#,
        ),
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
        assert!(predicate(CASES[7], response).is_ok(), "{response}");
    }
    for response in ["0 Celsius", "1 atm", "1°C", "0°C and 100°C"] {
        assert!(predicate(CASES[7], response).is_err(), "{response}");
    }
    assert!(predicate(CASES[8], "The light of the sun helps plants grow.").is_ok());
    for response in [
        "The light of the sun helps trees grow.",
        "The light of the sun helps plants thrive.",
    ] {
        assert!(predicate(CASES[8], response).is_err(), "{response}");
    }
    assert!(predicate(CASES[9], "Il libro è sulla tavola.").is_ok());
    assert!(predicate(CASES[11], r#"{"result":0,"result":42,"unit":"items"}"#).is_err());
}

#[test]
fn threshold_separates_semantic_acceptance_from_conformance() {
    let mut eight_semantic = [false; 12];
    for index in [0, 1, 2, 3, 5, 6, 7, 8] {
        eight_semantic[index] = true;
    }
    assert_eq!(scores(&eight_semantic), (4, 8, 0));
    assert!(threshold(&eight_semantic));
    let markers = [MarkerStatus::NotApplicable; 12];
    assert_eq!(
        summary_line(
            "fixture",
            Profile::Instruct,
            SemanticBackend::Cpu,
            &eight_semantic,
            &markers,
        ),
        "semantic-summary: model_id=fixture backend=cpu critical=4/4 semantic=8/9 semantic_status=pass conformance=0/3 conformance_status=diagnostic reasoning_format=not-applicable reasoning_format_status=not-applicable"
    );
    let mut reasoning_markers = [MarkerStatus::Absent; 12];
    reasoning_markers[..8].fill(MarkerStatus::Complete);
    assert!(
        summary_line(
            "fixture",
            Profile::Reasoning,
            SemanticBackend::Cpu,
            &eight_semantic,
            &reasoning_markers,
        )
        .ends_with("reasoning_format=8/12 reasoning_format_status=diagnostic")
    );

    let mut seven_semantic = eight_semantic;
    seven_semantic[8] = false;
    assert_eq!(scores(&seven_semantic), (4, 7, 0));
    assert!(!threshold(&seven_semantic));

    let mut critical_miss = [true; 12];
    critical_miss[0] = false;
    assert_eq!(scores(&critical_miss), (3, 8, 3));
    assert!(!threshold(&critical_miss));

    let all = [true; 12];
    assert_eq!(scores(&all), (4, 9, 3));
    assert!(threshold(&all));
}

#[test]
fn semantic_backend_follows_the_observed_probe() {
    let report = |mode, cpu_layers, gpu_layers| PlacementReport {
        mode,
        cpu_layers,
        gpu_layers,
        cpu: BackendMemory::default(),
        gpu: BackendMemory::default(),
    };
    assert_eq!(
        semantic_selection(report("all-gpu", 0, 34)).unwrap(),
        SemanticSelection {
            backend: SemanticBackend::Vulkan,
            reason: "full-vram-fit",
            probe_mode: "all-gpu",
        }
    );
    let mixed = semantic_selection(report("mixed", 12, 22)).unwrap();
    assert_eq!(mixed.backend, SemanticBackend::Cpu);
    assert_eq!(mixed.probe_mode, "mixed");
    let cpu = semantic_selection(report("cpu-only", 34, 0)).unwrap();
    assert_eq!(cpu.backend, SemanticBackend::Cpu);
    assert_eq!(cpu.probe_mode, "cpu-only");
    assert!(semantic_selection(report("unknown", 0, 0)).is_err());

    assert!(
        validate_final(
            semantic_selection(report("all-gpu", 0, 34)).unwrap(),
            report("all-gpu", 0, 34),
        )
        .is_ok()
    );
    assert!(validate_final(mixed, report("cpu-only", 34, 0)).is_ok());
    assert!(validate_final(cpu, report("cpu-only", 34, 0)).is_ok());
    assert!(validate_final(mixed, report("mixed", 12, 22)).is_err());
    assert!(validate_final(cpu, report("cpu-only", 33, 1)).is_err());
    assert!(
        validate_final(
            semantic_selection(report("all-gpu", 0, 34)).unwrap(),
            report("all-gpu", 1, 33),
        )
        .is_err()
    );
}

#[test]
fn timing_uses_checked_sums_and_exact_performance_boundary() {
    let mut timing = Timing::default();
    timing
        .add(GenerationStats {
            prompt_tokens: 1,
            completion_tokens: 2,
            prefill_ms: 3,
            decode_ms: 4,
        })
        .unwrap();
    assert_eq!(
        (timing.completed_cases, timing.prefill_ms, timing.decode_ms),
        (1, 3, 4)
    );

    timing.prefill_ms = u64::MAX;
    assert_eq!(
        timing.add(GenerationStats {
            prompt_tokens: 1,
            completion_tokens: 1,
            prefill_ms: 1,
            decode_ms: 1,
        }),
        Err("prefill timing overflow")
    );
    assert_eq!(
        performance("3b-instruct", SemanticBackend::Vulkan, 1_506_689),
        ("1506690", "pass", true)
    );
    assert_eq!(
        performance("3b-instruct", SemanticBackend::Vulkan, 1_506_690),
        ("1506690", "fail", false)
    );
    assert_eq!(
        performance("3b-instruct", SemanticBackend::Cpu, 1),
        ("not-applicable", "not-applicable", true)
    );
    assert_eq!(
        performance("8b-instruct", SemanticBackend::Vulkan, 1),
        ("not-applicable", "not-applicable", true)
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
fn reasoning_normalization_failure_omits_raw_thinking() {
    let (marker, result) = assessment(
        Profile::Reasoning,
        CASES[0],
        "[THINK]secret without close",
        Stop::Eos,
    );
    let error = result.unwrap_err();
    assert_eq!(marker, MarkerStatus::Invalid);
    assert!(error.contains("[invalid reasoning response omitted]"));
    assert!(!error.contains("secret"));

    let (marker, result) = assessment(Profile::Reasoning, CASES[0], "323", Stop::Context);
    assert_eq!(marker, MarkerStatus::Absent);
    assert!(result.unwrap_err().contains("incomplete-generation"));
}
