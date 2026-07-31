/*
 * gh_zero_engine — Reasoning parity acceptance boundary
 * This is the single test-only boundary between externally supplied oracle ID
 * vectors and backend-specific real-model tests. It requires exact prompt and
 * local-vector equality plus oracle top-two inclusion at every decode step.
 */

pub(super) const USER_CONTENT: &str = "Quanto fa 17 × 19?";
pub(super) const CONTEXT: usize = 4096;
pub(super) const TOKEN_COUNT: usize = 16;

pub(super) struct ReferenceVectors {
    pub(super) prompt: Vec<u32>,
    pub(super) completion: Vec<u32>,
}

pub(super) fn reference_vectors() -> ReferenceVectors {
    let prompt = read("GH_ZERO_REFERENCE_PROMPT_IDS", None);
    let completion = read("GH_ZERO_REFERENCE_COMPLETION_IDS", Some(TOKEN_COUNT));
    ReferenceVectors { prompt, completion }
}

pub(super) fn assert_exact(label: &str, actual: &[u32], expected: &[u32]) {
    assert!(
        actual == expected,
        "{label} mismatch\nexpected: {expected:?}\nactual: {actual:?}"
    );
}

pub(super) fn assert_oracle_top2(actual: &[Vec<u32>], expected: &[u32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "oracle top-two step count mismatch"
    );
    for (step, (&expected, candidates)) in expected.iter().zip(actual).enumerate() {
        assert!(
            candidates.contains(&expected),
            "oracle completion ID absent from top two at step {step}\n\
             expected: {expected}\ncandidates: {candidates:?}"
        );
    }
}

pub(super) fn csv(ids: &[u32]) -> String {
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
}

fn read(name: &str, expected_len: Option<usize>) -> Vec<u32> {
    let value = std::env::var(name).unwrap_or_else(|_| panic!("{name} required"));
    parse(name, &value, expected_len).unwrap_or_else(|message| panic!("{message}"))
}

fn parse(name: &str, value: &str, expected_len: Option<usize>) -> Result<Vec<u32>, String> {
    if value.trim().is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    let ids = value
        .split(',')
        .map(|part| {
            let part = part.trim();
            if part.is_empty() {
                return Err(format!("{name} contains an empty ID"));
            }
            part.parse::<u32>()
                .map_err(|_| format!("{name} contains an invalid ID"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if expected_len.is_some_and(|expected| ids.len() != expected) {
        return Err(format!(
            "{name} must contain exactly {} IDs",
            expected_len.unwrap()
        ));
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_strict_unsigned_decimal_ids() {
        assert_eq!(
            parse("IDS", "1, 2,4294967295", None).unwrap(),
            [1, 2, u32::MAX]
        );
    }

    #[test]
    fn rejects_empty_malformed_negative_and_overflow_ids() {
        for value in ["", "1,", "1,,2", "-1", "4294967296", "one"] {
            assert!(parse("IDS", value, None).is_err(), "{value:?}");
        }
    }

    #[test]
    fn completion_length_is_exactly_sixteen() {
        let sixteen = (0..TOKEN_COUNT)
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(
            parse("IDS", &sixteen, Some(TOKEN_COUNT)).unwrap().len(),
            TOKEN_COUNT
        );
        assert!(parse("IDS", "1,2", Some(TOKEN_COUNT)).is_err());
    }

    #[test]
    fn vector_mismatch_reports_expected_and_actual() {
        let panic = std::panic::catch_unwind(|| assert_exact("completion", &[1], &[2]));
        let message = panic.unwrap_err();
        let message = message
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| message.downcast_ref::<&str>().copied())
            .unwrap();
        assert!(message.contains("expected: [2]"));
        assert!(message.contains("actual: [1]"));
    }

    #[test]
    fn oracle_id_must_be_present_in_every_top_two() {
        assert_oracle_top2(&[vec![2, 1], vec![3, 4]], &[1, 3]);
        let panic =
            std::panic::catch_unwind(|| assert_oracle_top2(&[vec![1, 2], vec![3, 4]], &[1, 5]));
        let message = panic.unwrap_err();
        let message = message
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| message.downcast_ref::<&str>().copied())
            .unwrap();
        assert!(message.contains("step 1"));
        assert!(message.contains("expected: 5"));
    }
}
