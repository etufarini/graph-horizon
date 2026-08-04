/*
 * gh_zero_engine — Ministral phase-benchmark adapter
 * Owns the exact short/long token fixtures, validates them against the loaded
 * vocabulary before session allocation, computes their stable SHA-256 identity,
 * and delegates timing to the neutral runtime. It owns no statistics or I/O.
 */

use color_eyre::eyre::{Result, bail};

use super::RuntimeModel;
use super::graph::MistralGraph;
use crate::backend::selection;
use crate::runtime::phases::{self, PhaseFixture, PhaseSample};

const SHORT: [u32; 16] = [
    1, 17, 18, 3, 6605, 4842, 3456, 1032, 1049, 1055, 10039, 1032, 1049, 1057, 1063, 4,
];
const LONG_TOKEN: u32 = 1032;
const LONG_LEN: usize = 512;

pub(crate) fn measure(
    model: &RuntimeModel,
    fixture: PhaseFixture,
) -> Result<(PhaseSample, String)> {
    let tokens = tokens(fixture);
    validate(&tokens, model.config.vocab_size)?;
    let digest = digest(&tokens);
    let session = selection::session::<MistralGraph>(
        &model.backend,
        &model.config,
        model.shape(),
        model.context,
        model.scheme,
    )?;
    Ok((
        phases::measure(&session, &tokens, model.config.vocab_size)?,
        digest,
    ))
}

fn tokens(fixture: PhaseFixture) -> Vec<u32> {
    match fixture {
        PhaseFixture::Short => SHORT.to_vec(),
        PhaseFixture::Long => vec![LONG_TOKEN; LONG_LEN],
    }
}

fn validate(tokens: &[u32], vocab: usize) -> Result<()> {
    if tokens.is_empty() || tokens.iter().any(|&token| token as usize >= vocab) {
        bail!("invalid fixture");
    }
    Ok(())
}

fn digest(tokens: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    sha256(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn sha256(input: &[u8]) -> [u8; 32] {
    const INITIAL: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    let mut hash = INITIAL;
    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (word, bytes) in w.iter_mut().zip(chunk.chunks_exact(4)) {
            *word = u32::from_be_bytes(bytes.try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = hash;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (state, value) in hash.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *state = state.wrapping_add(value);
        }
    }
    let mut output = [0u8; 32];
    for (bytes, value) in output.chunks_exact_mut(4).zip(hash) {
        bytes.copy_from_slice(&value.to_be_bytes());
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harness_phases_fixtures_are_exact_and_digest_their_little_endian_ids() {
        assert_eq!(SHORT.len(), 16);
        assert_eq!(LONG_LEN, 512);
        assert_eq!(tokens(PhaseFixture::Short), SHORT);
        assert_eq!(tokens(PhaseFixture::Long), vec![LONG_TOKEN; LONG_LEN]);
        assert_eq!(
            digest(&SHORT),
            "17bec2373e86851982bc193bec838f25ad2443d247404609dcceee53937a2991"
        );
        assert_eq!(
            digest(&tokens(PhaseFixture::Long)),
            "d45d4aeef88e8dbbd085d928fe121b282965a9a363b280cc07449e4b561f5bac"
        );
        assert_eq!(
            digest(&[]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            digest(&[1]),
            "67abdd721024f0ff4e0b3f4c2fc13bc5bad42d0b7851d456d88d203d15aaa450"
        );
    }

    #[test]
    fn harness_phases_fixtures_must_fit_the_loaded_vocabulary() {
        assert_eq!(validate(&tokens(PhaseFixture::Short), 10_040).unwrap(), ());
        assert_eq!(
            validate(&[3], 3).unwrap_err().to_string(),
            "invalid fixture"
        );
        assert_eq!(
            validate(&[], 10).unwrap_err().to_string(),
            "invalid fixture"
        );
    }
}
