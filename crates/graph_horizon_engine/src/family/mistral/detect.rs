/*
 * graph_horizon_engine — Ministral architecture detector
 * Gates untrusted GGUF architecture, forbidden capabilities, and the sole
 * Q4_K_M profile without allocating resources or returning a dispatch value.
 */

use std::collections::HashMap;

use color_eyre::eyre::{Result, bail, eyre};

use crate::gguf::loader::GgufValue;
use crate::gguf::tensor_index::TensorInfo;

pub(crate) fn detect(md: &HashMap<String, GgufValue>, tensors: &[TensorInfo]) -> Result<()> {
    let arch = md
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .ok_or_else(|| eyre!("E06 missing or invalid GGUF metadata 'general.architecture'"))?;
    if arch != "mistral3" {
        bail!("E03 unsupported architecture '{arch}'; supported architecture: mistral3");
    }
    reject_forbidden(md, tensors)?;
    profile(md)
}

fn profile(md: &HashMap<String, GgufValue>) -> Result<()> {
    match md
        .get("general.file_type")
        .and_then(unsigned_value)
        .ok_or_else(|| eyre!("E06 missing or invalid GGUF metadata 'general.file_type'"))?
    {
        15 => Ok(()),
        other => bail!(
            "E04 unsupported GGUF weight profile '{}'; supported profile: Q4_K_M",
            file_type_name(other)
        ),
    }
}

fn unsigned_value(value: &GgufValue) -> Option<u64> {
    match *value {
        GgufValue::U8(v) => Some(v as u64),
        GgufValue::U16(v) => Some(v as u64),
        GgufValue::U32(v) => Some(v as u64),
        GgufValue::U64(v) => Some(v),
        _ => None,
    }
}

fn file_type_name(value: u64) -> String {
    match value {
        0 => "F32".into(),
        1 => "F16".into(),
        7 => "Q8_0".into(),
        14 => "Q6_K".into(),
        16 => "IQ2_XXS".into(),
        other => format!("unknown({other})"),
    }
}

fn reject_forbidden(md: &HashMap<String, GgufValue>, tensors: &[TensorInfo]) -> Result<()> {
    let names = md
        .keys()
        .map(String::as_str)
        .chain(tensors.iter().map(|t| t.name.as_str()));
    for name in names {
        if name.contains("expert") || name.contains("router") {
            bail!("E05 mixture-of-experts models are not supported");
        }
        if name.contains("ssm") || name.contains("recurrent") || name.contains("conv") {
            bail!("E05 state-space models are not supported");
        }
        if name.contains("vision")
            || name.contains("mmproj")
            || name.contains("image")
            || name.contains("multimodal")
        {
            bail!("E05 multimodal model tensors are not supported");
        }
    }
    Ok(())
}

#[cfg(test)]
#[test]
#[ignore]
fn real_contract() {
    use std::collections::BTreeMap;

    use crate::family::mistral::MistralContract;
    use crate::family::mistral::tensors::OutputTensor;
    use crate::family::mistral::version::{Q_WIDTH, REFERENCE_ROWS};
    use crate::gguf::loader::GgufFile;

    let path = std::env::var("GRAPH_HORIZON_MODEL").expect("GRAPH_HORIZON_MODEL required");
    let file = GgufFile::open(std::path::Path::new(&path)).expect("open GGUF");
    let contract = MistralContract::from_gguf(&file).expect("Ministral contract");
    let mut histogram = BTreeMap::new();
    for tensor in file.tensors() {
        *histogram.entry(tensor.ggml_type.name()).or_insert(0usize) += 1;
    }
    let output = match contract.tensors.output {
        OutputTensor::Tied => "tied",
        OutputTensor::Dedicated(_) => "dedicated",
    };
    println!(
        "profile=Q4_K_M hidden={} q={} k={} v={} histogram={:?} output={output} vocab={} bos={} eos={}",
        contract.config.embedding_length,
        contract.config.q_width,
        contract.config.k_width,
        contract.config.v_width,
        histogram,
        contract.tokenizer.vocab_size(),
        contract.tokenizer.bos_id(),
        contract.tokenizer.eos_id()
    );
    assert_eq!(contract.config.embedding_length, REFERENCE_ROWS[0].1);
    assert_eq!(contract.config.q_width, Q_WIDTH);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tensor_index::GgmlType;

    fn md(arch: &str, file_type: u32) -> HashMap<String, GgufValue> {
        HashMap::from([
            (
                "general.architecture".into(),
                GgufValue::String(arch.into()),
            ),
            ("general.file_type".into(), GgufValue::U32(file_type)),
        ])
    }

    #[test]
    fn error_matrix_e04_accepts_only_q4_k_m() {
        assert!(detect(&md("mistral3", 15), &[]).is_ok());
        for (file_type, name) in [
            (7, "Q8_0"),
            (0, "F32"),
            (1, "F16"),
            (14, "Q6_K"),
            (99, "unknown(99)"),
        ] {
            let err = detect(&md("mistral3", file_type), &[])
                .unwrap_err()
                .to_string();
            assert_eq!(
                err,
                format!("E04 unsupported GGUF weight profile '{name}'; supported profile: Q4_K_M")
            );
        }
    }

    #[test]
    fn file_type_requires_an_unsigned_integer_and_accepts_all_unsigned_widths() {
        for value in [
            GgufValue::U8(15),
            GgufValue::U16(15),
            GgufValue::U32(15),
            GgufValue::U64(15),
        ] {
            let mut metadata = md("mistral3", 15);
            metadata.insert("general.file_type".into(), value);
            assert!(detect(&metadata, &[]).is_ok());
        }

        let mut metadata = md("mistral3", 15);
        metadata.remove("general.file_type");
        assert_eq!(
            detect(&metadata, &[]).unwrap_err().to_string(),
            "E06 missing or invalid GGUF metadata 'general.file_type'"
        );
        metadata.insert("general.file_type".into(), GgufValue::String("15".into()));
        assert_eq!(
            detect(&metadata, &[]).unwrap_err().to_string(),
            "E06 missing or invalid GGUF metadata 'general.file_type'"
        );
    }

    #[test]
    fn error_matrix_e03_e05_rejects_architecture_and_forbidden_capabilities() {
        let err = detect(&md("llama", 7), &[]).unwrap_err().to_string();
        assert!(err.contains("E03 unsupported architecture 'llama'"));
        let tensors = [TensorInfo {
            name: "blk.0.ffn_gate_experts.weight".into(),
            dims: vec![1],
            ggml_type: GgmlType::F32,
            offset: 0,
        }];
        let err = detect(&md("mistral3", 7), &tensors)
            .unwrap_err()
            .to_string();
        assert!(err.contains("E05 mixture-of-experts"));

        let tensors = [TensorInfo {
            name: "blk.0.ssm_conv1d.weight".into(),
            dims: vec![1],
            ggml_type: GgmlType::F32,
            offset: 0,
        }];
        let err = detect(&md("mistral3", 7), &tensors)
            .unwrap_err()
            .to_string();
        assert!(err.contains("E05 state-space models"));

        let tensors = [TensorInfo {
            name: "mmproj.weight".into(),
            dims: vec![1],
            ggml_type: GgmlType::F32,
            offset: 0,
        }];
        let err = detect(&md("mistral3", 7), &tensors)
            .unwrap_err()
            .to_string();
        assert!(err.contains("E05 multimodal model tensors"));
    }
}
