/*
 * gh_zero_engine — Ministral architecture detector
 * Gates untrusted GGUF architecture/capabilities/profile and returns E03-E05 without allocation or path disclosure.
 */

use std::collections::HashMap;

use color_eyre::eyre::{Result, bail, eyre};

use crate::gguf::loader::GgufValue;
use crate::gguf::tensor_index::TensorInfo;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum WeightProfile {
    Q8_0,
    Q4_K_M,
}

pub(crate) fn detect(
    md: &HashMap<String, GgufValue>,
    tensors: &[TensorInfo],
) -> Result<WeightProfile> {
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

fn profile(md: &HashMap<String, GgufValue>) -> Result<WeightProfile> {
    match md
        .get("general.file_type")
        .and_then(unsigned_value)
        .ok_or_else(|| eyre!("E06 missing or invalid GGUF metadata 'general.file_type'"))?
    {
        7 => Ok(WeightProfile::Q8_0),
        15 => Ok(WeightProfile::Q4_K_M),
        other => bail!(
            "E04 unsupported GGUF weight profile '{}'; supported profiles: Q8_0, Q4_K_M",
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

    let path = std::env::var("GH_ZERO_MODEL").expect("GH_ZERO_MODEL required");
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
        "profile={:?} hidden={} q={} k={} v={} histogram={:?} output={output} vocab={} bos={} eos={}",
        contract.profile,
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
    fn error_matrix_e04_accepts_only_two_public_profiles() {
        assert_eq!(
            detect(&md("mistral3", 7), &[]).unwrap(),
            WeightProfile::Q8_0
        );
        assert_eq!(
            detect(&md("mistral3", 15), &[]).unwrap(),
            WeightProfile::Q4_K_M
        );
        let err = detect(&md("mistral3", 14), &[]).unwrap_err().to_string();
        assert!(err.contains("E04 unsupported GGUF weight profile 'Q6_K'"));
        let err = detect(&md("mistral3", 99), &[]).unwrap_err().to_string();
        assert!(err.contains("E04 unsupported GGUF weight profile 'unknown(99)'"));
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
