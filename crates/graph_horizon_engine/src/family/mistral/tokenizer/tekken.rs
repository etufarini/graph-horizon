/*
 * graph_horizon_engine — Tekken pre-tokenizer segmentation
 * Small dependency-free segmentation state machine for the M1 Tekken boundary.
 * It keeps `clean_spaces=false` behavior by preserving whitespace exactly and
 * treats every segment as ordinary bytes; control-token interpretation is left
 * exclusively to the chat renderer.
*/

pub(super) fn segments(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let start = i;
        let ch = chars[i].1;
        let lead = is_prefix(ch) && i + 1 < chars.len() && is_word_unit(chars[i + 1].1);
        let letter_at = i + usize::from(lead);
        if let Some(end) = word_end(&chars, letter_at) {
            i = end;
            out.push(slice(text, &chars, start, end));
            continue;
        }
        if ch.is_numeric() {
            out.push(slice(text, &chars, i, i + 1));
            i += 1;
            continue;
        }
        let symbol_start = i + usize::from(ch == ' ');
        if symbol_start < chars.len() && is_symbol(chars[symbol_start].1) {
            i = symbol_start + 1;
            while i < chars.len() && is_symbol(chars[i].1) {
                i += 1;
            }
            while i < chars.len() && matches!(chars[i].1, '\r' | '\n' | '/') {
                i += 1;
            }
            out.push(slice(text, &chars, start, i));
            continue;
        }
        if ch.is_whitespace() {
            let mut end = i + 1;
            let mut last_newline = matches!(ch, '\r' | '\n').then_some(end);
            while end < chars.len() && chars[end].1.is_whitespace() {
                end += 1;
                if matches!(chars[end - 1].1, '\r' | '\n') {
                    last_newline = Some(end);
                }
            }
            // `\s*[\r\n]+` wins before the generic whitespace alternatives,
            // leaving whitespace after the final newline for the next segment.
            i = last_newline.unwrap_or_else(|| {
                if end < chars.len() && is_word_unit(chars[end].1) {
                    end - 1
                } else {
                    end
                }
            });
        } else {
            i += 1;
        }
        out.push(slice(text, &chars, start, i));
    }
    out
}

fn slice<'a>(text: &'a str, chars: &[(usize, char)], start: usize, end: usize) -> &'a str {
    let byte_start = chars[start].0;
    let byte_end = chars
        .get(end)
        .map(|(i, _)| *i)
        .unwrap_or_else(|| text.len());
    &text[byte_start..byte_end]
}

fn is_prefix(ch: char) -> bool {
    !matches!(ch, '\r' | '\n') && !ch.is_alphabetic() && !ch.is_numeric()
}

fn is_symbol(ch: char) -> bool {
    !ch.is_whitespace() && !ch.is_alphabetic() && !ch.is_numeric()
}

fn word_end(chars: &[(usize, char)], start: usize) -> Option<usize> {
    if start >= chars.len() || !is_word_unit(chars[start].1) {
        return None;
    }
    let mut i = start;
    while i < chars.len() && upper_side(chars[i].1) {
        i += 1;
    }
    let lower_start = i;
    while i < chars.len() && lower_side(chars[i].1) {
        i += 1;
    }
    if i > lower_start {
        return Some(i);
    }

    i = start;
    while i < chars.len() && upper_side(chars[i].1) {
        i += 1;
    }
    if i > start {
        while i < chars.len() && lower_side(chars[i].1) {
            i += 1;
        }
        return Some(i);
    }
    None
}

fn upper_side(ch: char) -> bool {
    is_word_unit(ch) && !ch.is_ascii_lowercase()
}

fn lower_side(ch: char) -> bool {
    is_word_unit(ch) && !ch.is_ascii_uppercase()
}

fn is_word_unit(ch: char) -> bool {
    ch.is_alphabetic() || is_combining_mark(ch)
}

fn is_combining_mark(ch: char) -> bool {
    matches!(
        ch,
        '\u{0300}'..='\u{036f}'
            | '\u{0483}'..='\u{0489}'
            | '\u{0591}'..='\u{05bd}'
            | '\u{05bf}'
            | '\u{05c1}'..='\u{05c2}'
            | '\u{05c4}'..='\u{05c5}'
            | '\u{05c7}'
            | '\u{0610}'..='\u{061a}'
            | '\u{064b}'..='\u{065f}'
            | '\u{0670}'
            | '\u{06d6}'..='\u{06dc}'
            | '\u{06df}'..='\u{06e4}'
            | '\u{06e7}'..='\u{06e8}'
            | '\u{06ea}'..='\u{06ed}'
            | '\u{1ab0}'..='\u{1aff}'
            | '\u{1dc0}'..='\u{1dff}'
            | '\u{20d0}'..='\u{20ff}'
            | '\u{fe20}'..='\u{fe2f}'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pinned_unicode_and_whitespace_classes() {
        let vectors = [
            ("ascii case", "abcDEF ABCdef", vec!["abc", "DEF", " ABCdef"]),
            ("titlecase", "ǅuro", vec!["ǅuro"]),
            ("modifier letter", "ʰello", vec!["ʰello"]),
            ("other letter", "中文", vec!["中文"]),
            (
                "combining mark",
                "e\u{301}\u{327}x",
                vec!["e\u{301}\u{327}x"],
            ),
            ("single digits", "42", vec!["4", "2"]),
            ("punctuation", "x !?/\ny", vec!["x", " !?/\n", "y"]),
            (
                "newline run",
                "a \t\n\r  b",
                vec!["a", " \t\n\r", " ", " b"],
            ),
            ("interior whitespace", "a\t\tb", vec!["a", "\t", "\tb"]),
            ("trailing whitespace", "a  ", vec!["a", "  "]),
        ];
        for (name, input, expected) in vectors {
            assert_eq!(segments(input), expected, "{name}");
        }
        assert!(segments("").is_empty());
    }
}
