/*
 * graph_horizon_engine — Ministral bounded BPE merge
 * Applies ranked byte-level BPE merges for one Tekken segment. The merge table
 * is already validated from untrusted GGUF metadata, no regex or external
 * dependency is used, and byte fallback remains available for unknown symbols.
*/

use std::collections::HashMap;

use color_eyre::eyre::{Result, bail, eyre};

use crate::gguf::loader::GgufValue;

pub(super) fn byte_maps() -> ([char; 256], HashMap<char, u8>) {
    let printable = |b: u32| {
        (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b)
    };
    let mut encoder = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..=255u32 {
        encoder[b as usize] = if printable(b) {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
    }
    let decoder = encoder
        .iter()
        .enumerate()
        .map(|(b, &c)| (c, b as u8))
        .collect();
    (encoder, decoder)
}

pub(super) fn load_merges(values: &[GgufValue]) -> Result<HashMap<String, u32>> {
    let mut ranks = HashMap::with_capacity(values.len());
    for (rank, value) in values.iter().enumerate() {
        let merge = value
            .as_str()
            .ok_or_else(|| eyre!("E09 invalid Tekken tokenizer"))?;
        let Some((left, right)) = merge.split_once(' ') else {
            bail!("E09 invalid Tekken tokenizer");
        };
        if left.is_empty() || right.is_empty() || right.contains(' ') {
            bail!("E09 invalid Tekken tokenizer");
        }
        if ranks.insert(merge.to_string(), rank as u32).is_some() {
            bail!("E09 invalid Tekken tokenizer");
        }
    }
    Ok(ranks)
}

pub(super) fn encode_piece(ranks: &HashMap<String, u32>, word: &str) -> Vec<String> {
    let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
    if symbols.len() < 2 {
        return symbols;
    }
    loop {
        let mut best = None;
        for i in 0..symbols.len() - 1 {
            let key = format!("{} {}", symbols[i], symbols[i + 1]);
            if let Some(&rank) = ranks.get(&key)
                && best.is_none_or(|(_, best_rank)| rank < best_rank)
            {
                best = Some((i, rank));
            }
        }
        match best {
            Some((i, _)) => {
                let merged = format!("{}{}", symbols[i], symbols[i + 1]);
                symbols[i] = merged;
                symbols.remove(i + 1);
            }
            None => break,
        }
    }
    symbols
}
