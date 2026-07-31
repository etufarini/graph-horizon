/*
 * gh_zero_engine — deterministic semantic product acceptance
 * This integration harness owns one fixed objective corpus and its strict
 * response predicates. It is neither a runtime quality API nor a benchmark.
 */

use std::path::Path;

use gh_zero_engine::{
    Engine, EngineConfig, Event, GenerationStats, KvQuant, Message, Request, Role, SamplingParams,
};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Profile {
    Instruct,
    Reasoning,
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
    critical: bool,
    name: &'static str,
    predicate: Predicate,
}

const CASES: [Case; 12] = [
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
        critical: true,
        name: "exact sequence answer",
        predicate: Predicate::ExactDigits("42"),
    },
    Case {
        id: "S05",
        prompt: "Write exactly three lines. First line: Alpha. Second line: Beta. Third line: Gamma. Add nothing else.",
        critical: true,
        name: "exact three lines",
        predicate: Predicate::ThreeLines,
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
    Case {
        id: "S11",
        prompt: "Return a Markdown unordered list containing exactly these colors in this order: red, green, blue. No introduction or conclusion.",
        critical: false,
        name: "Markdown color list",
        predicate: Predicate::MarkdownColors,
    },
    Case {
        id: "S12",
        prompt: "Return only a JSON object with exactly two fields: result set to the number 42, and unit set to the string \"items\".",
        critical: false,
        name: "exact JSON object",
        predicate: Predicate::JsonObject,
    },
];

fn normalize(profile: Profile, raw: &str) -> Result<String, &'static str> {
    let trimmed = raw.trim();
    match profile {
        Profile::Instruct => {
            if trimmed.contains("[THINK]") || trimmed.contains("[/THINK]") {
                return Err("reasoning marker in Instruct response");
            }
            Ok(trimmed.to_owned())
        }
        Profile::Reasoning => {
            if !trimmed.starts_with("[THINK]")
                || trimmed.matches("[THINK]").count() != 1
                || trimmed.matches("[/THINK]").count() != 1
            {
                return Err("invalid reasoning marker contract");
            }
            let (_, answer) = trimmed
                .split_once("[/THINK]")
                .ok_or("invalid reasoning marker contract")?;
            let answer = answer.trim();
            if answer.is_empty() {
                return Err("empty reasoning final answer");
            }
            Ok(answer.to_owned())
        }
    }
}

fn model_profile(model_id: &str) -> Result<Profile, &'static str> {
    match model_id {
        "3b-instruct" | "8b-instruct" | "14b-instruct" => Ok(Profile::Instruct),
        "3b-reasoning" | "8b-reasoning" | "14b-reasoning" => Ok(Profile::Reasoning),
        _ => Err("unknown GH_ZERO_MODEL_ID"),
    }
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
            lower.contains("sunlight")
                && lower.contains("plants")
                && lower.contains("grow")
                && !lower.contains("la luce del sole aiuta le piante a crescere.")
        }
        Predicate::ItalianTranslation => {
            response
                .strip_suffix('.')
                .unwrap_or(response)
                .to_lowercase()
                == "il libro è sul tavolo"
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
    let lower = response.to_lowercase();
    let unit = response.contains("°C") || response.contains('C') || lower.contains("celsius");
    let bytes = response.as_bytes();
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
            values.push(&response[start..index]);
        } else {
            index += 1;
        }
    }
    unit && values.len() == 1 && values[0].parse::<f64>().is_ok_and(|value| value == 0.0)
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

fn threshold(results: &[bool; 12]) -> bool {
    CASES
        .iter()
        .zip(results)
        .filter(|(case, _)| case.critical)
        .all(|(_, passed)| *passed)
        && results.iter().filter(|passed| **passed).count() >= 11
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

fn event_response(events: &[Event]) -> Result<String, &'static str> {
    let mut parts = Vec::new();
    let mut terminals = 0;
    for event in events {
        match event {
            Event::TextDelta(text) if terminals == 0 => parts.push(text.as_bytes().to_vec()),
            Event::TextDelta(_) => return Err("TextDelta after terminal event"),
            Event::Finished(_) => terminals += 1,
            Event::Error(_) => return Err("generation emitted Error"),
        }
    }
    match terminals {
        1 => assembly(&parts),
        0 => Err("generation lacks Finished"),
        _ => Err("generation emitted multiple terminal events"),
    }
}

fn assessment(profile: Profile, case: Case, raw: &str) -> Result<(), String> {
    let scored = normalize(profile, raw).map_err(|reason| {
        let response = match profile {
            Profile::Instruct => excerpt(raw),
            Profile::Reasoning => "[invalid reasoning response omitted]".into(),
        };
        format!(
            "predicate=response-normalization reason={reason} excerpt={}",
            response.escape_debug()
        )
    })?;
    predicate(case, &scored).map_err(|reason| {
        format!(
            "predicate={} reason={reason} excerpt={}",
            case.name.replace(' ', "-"),
            excerpt(&scored).escape_debug()
        )
    })
}

#[test]
#[ignore = "requires one authenticated Ministral Q4_K_M model"]
fn real_semantic_acceptance() {
    let model_id = std::env::var("GH_ZERO_MODEL_ID").expect("GH_ZERO_MODEL_ID required");
    let model = std::env::var("GH_ZERO_MODEL").expect("GH_ZERO_MODEL required");
    // Both environment values and the exact profile ID are validated before construction.
    let profile = model_profile(&model_id).unwrap_or_else(|reason| panic!("{reason}"));
    let engine = Engine::new(
        Path::new(&model),
        EngineConfig {
            context_tokens: Some(4096),
            kv_quant: KvQuant::F16,
            ..EngineConfig::default()
        },
    )
    .unwrap_or_else(|_| panic!("authenticated model failed to load"));
    let mut results = [false; 12];

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
                max_tokens: 256,
            },
            &mut |event| {
                events.push(event);
                true
            },
        );
        let result = event_response(&events)
            .map_err(|reason| format!("predicate=engine-events reason={reason}"))
            .and_then(|raw| assessment(profile, case, &raw));
        results[index] = result.is_ok();
        match result {
            Ok(()) => println!(
                "semantic-case: model_id={model_id} case_id={} status=pass predicate={}",
                case.id,
                case.name.replace(' ', "-")
            ),
            Err(reason) => println!(
                "semantic-case: model_id={model_id} case_id={} status=fail {reason}",
                case.id
            ),
        }
    }

    let critical = CASES
        .iter()
        .zip(results)
        .filter(|(case, passed)| case.critical && *passed)
        .count();
    let total = results.iter().filter(|passed| **passed).count();
    let passed = threshold(&results);
    println!(
        "semantic-summary: model_id={model_id} critical={critical}/6 total={total}/12 status={}",
        if passed { "pass" } else { "fail" }
    );
    assert!(passed, "semantic acceptance threshold missed");
}

#[test]
fn corpus_is_exact_and_has_six_critical_cases() {
    assert_eq!(CASES.len(), 12);
    for (index, case) in CASES.iter().enumerate() {
        assert_eq!(case.id, format!("S{:02}", index + 1));
        assert!(!case.prompt.is_empty());
        assert_eq!(case.critical, index < 6);
    }
}

#[test]
fn normalization_enforces_profile_marker_contracts() {
    assert_eq!(normalize(Profile::Instruct, " 323 \n").unwrap(), "323");
    for raw in ["[THINK]x[/THINK] 323", "x [/THINK]"] {
        assert!(normalize(Profile::Instruct, raw).is_err(), "{raw}");
    }
    assert_eq!(
        normalize(Profile::Reasoning, " [THINK]work[/THINK]\n323 ").unwrap(),
        "323"
    );
    for raw in [
        "323",
        "x[THINK]work[/THINK] 323",
        "[THINK]work 323",
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
        ("0 °C", "0 °C or 32 °F"),
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
    assert!(predicate(CASES[11], r#"{"result":0,"result":42,"unit":"items"}"#).is_err());
}

#[test]
fn threshold_requires_all_critical_and_eleven_total() {
    assert!(threshold(&[true; 12]));
    let mut eleven = [true; 12];
    eleven[11] = false;
    assert!(threshold(&eleven));
    let mut ten = eleven;
    ten[10] = false;
    assert!(!threshold(&ten));
    let mut critical_miss = [true; 12];
    critical_miss[0] = false;
    assert!(!threshold(&critical_miss));
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
        "ab"
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
    let error =
        assessment(Profile::Reasoning, CASES[0], "[THINK]secret without close").unwrap_err();
    assert!(error.contains("response-normalization"));
    assert!(error.contains("[invalid reasoning response omitted]"));
    assert!(!error.contains("secret"));
}
