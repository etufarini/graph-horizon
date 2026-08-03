/*
 * gh_zero_engine — read-only Ministral inspector
 * Applies the Q4-only profile gate, validates metadata/tokenizer shape, and
 * prints capability facts without constructing a backend. It never
 * authenticates provenance and normalizes file/parse errors for callers.
 */

use std::collections::BTreeMap;
use std::path::PathBuf;

use color_eyre::eyre::{Result, bail, eyre};
use gh_zero_engine::{GgufFile, GgufValue, MistralConfig, TekkenTokenizer};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(
        args.next()
            .ok_or_else(|| eyre!("usage: inspect <model.gguf> [--list]"))?,
    );
    let list = args.next().as_deref() == Some("--list");
    if args.next().is_some() {
        bail!("usage: inspect <model.gguf> [--list]");
    }

    std::fs::File::open(&path).map_err(|_| eyre!("model file is missing or unreadable"))?;
    let file = GgufFile::open(&path).map_err(|_| eyre!("invalid GGUF file"))?;
    let architecture = file
        .metadata()
        .get("general.architecture")
        .and_then(GgufValue::as_str)
        .ok_or_else(|| eyre!("E06 missing or invalid GGUF metadata 'general.architecture'"))?;
    if architecture != "mistral3" {
        bail!("E03 unsupported architecture '{architecture}'; supported architecture: mistral3");
    }
    profile(file.metadata())?;
    let tokenizer = TekkenTokenizer::from_metadata(file.metadata())
        .map_err(|_| eyre!("invalid Tekken tokenizer"))?;
    let config = MistralConfig::from_metadata(
        file.metadata(),
        tokenizer.vocab_size(),
        tokenizer.bos_id(),
        tokenizer.eos_id(),
    )?;
    let output = if file
        .tensors()
        .iter()
        .any(|tensor| tensor.name == "output.weight")
    {
        "dedicated"
    } else {
        "tied-to-embedding"
    };
    let mut histogram = BTreeMap::new();
    for tensor in file.tensors() {
        *histogram.entry(tensor.ggml_type.name()).or_insert(0usize) += 1;
    }

    println!("architecture: {architecture}");
    println!("weight_profile: Q4_K_M");
    println!("verification: compatible/unverified");
    println!(
        "dimensions: blocks={} hidden={} q={} k={} v={} ffn={} context={}",
        config.block_count,
        config.embedding_length,
        config.q_width,
        config.k_width,
        config.v_width,
        config.feed_forward_length,
        config.context_length
    );
    println!(
        "tokenizer: model={} pre={} vocab={} bos={} eos={}",
        text(file.metadata(), "tokenizer.ggml.model")?,
        text(file.metadata(), "tokenizer.ggml.pre")?,
        tokenizer.vocab_size(),
        tokenizer.bos_id(),
        tokenizer.eos_id()
    );
    println!("output: {output}");
    println!("tensor_histogram:");
    for (kind, count) in histogram {
        println!("  {kind}: {count}");
    }
    if list {
        println!("tensors:");
        for tensor in file.tensors() {
            println!(
                "  {} {} {:?}",
                tensor.name,
                tensor.ggml_type.name(),
                tensor.dims
            );
        }
    }
    Ok(())
}

fn text<'a>(
    metadata: &'a std::collections::HashMap<String, GgufValue>,
    key: &str,
) -> Result<&'a str> {
    metadata
        .get(key)
        .and_then(GgufValue::as_str)
        .ok_or_else(|| eyre!("missing or invalid GGUF metadata '{key}'"))
}

fn profile(metadata: &std::collections::HashMap<String, GgufValue>) -> Result<()> {
    let value = match metadata.get("general.file_type") {
        Some(GgufValue::U8(value)) => *value as u64,
        Some(GgufValue::U16(value)) => *value as u64,
        Some(GgufValue::U32(value)) => *value as u64,
        Some(GgufValue::U64(value)) => *value,
        _ => bail!("E06 missing or invalid GGUF metadata 'general.file_type'"),
    };
    match value {
        15 => Ok(()),
        value => {
            let name = match value {
                0 => "F32".into(),
                1 => "F16".into(),
                7 => "Q8_0".into(),
                14 => "Q6_K".into(),
                other => format!("unknown({other})"),
            };
            bail!("E04 unsupported GGUF weight profile '{name}'; supported profile: Q4_K_M")
        }
    }
}
