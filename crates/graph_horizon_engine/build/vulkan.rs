/*
 * graph_horizon_engine — offline Vulkan shader compilation and variants.
 */

use std::path::Path;

pub(crate) fn build() {
    let shader_dir = Path::new("src/backend/vulkan/shaders");
    println!("cargo:rerun-if-changed={}", shader_dir.display());
    let mut shaders = Vec::new();
    collect_comp(shader_dir, &mut shaders);

    let compiler = shaderc::Compiler::new().expect("shaderc: cannot create compiler");
    let out_dir = std::env::var("OUT_DIR").expect("build: OUT_DIR not set");
    for path in shaders {
        println!("cargo:rerun-if-changed={}", path.display());
        let name = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .expect("build: invalid shader file name");
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("build: cannot read shader '{}'", path.display()));
        compile_shader(&compiler, &path, &source, name, Path::new(&out_dir));
    }
}

fn compile_shader(
    compiler: &shaderc::Compiler,
    path: &Path,
    source: &str,
    name: &str,
    out_dir: &Path,
) {
    let variants: &[(&str, &str, &str)] = match name {
        "attention_decode" => &[
            ("attention_decode_wide", "ATTENTION_LOCAL_SIZE", "512"),
            ("attention_decode_1024", "ATTENTION_LOCAL_SIZE", "1024"),
        ],
        "attention_prefill" => &[("attention_prefill_wide", "ATTENTION_LOCAL_SIZE", "512")],
        "matmul_q4_k_coopmat_f16out" => &[(
            "matmul_q4_k_coopmat_metadata_f16out",
            "Q4_METADATA_BROADCAST",
            "1",
        )],
        _ => &[],
    };
    let variants = variants
        .iter()
        .map(|&(variant, macro_name, value)| (variant, Some((macro_name, value))));
    for (variant, macro_definition) in std::iter::once((name, None)).chain(variants) {
        let mut options = shaderc::CompileOptions::new().expect("shaderc: cannot create options");
        let vulkan13 = shaderc::EnvVersion::Vulkan1_3 as u32;
        options.set_target_env(shaderc::TargetEnv::Vulkan, vulkan13);
        if let Some((macro_name, value)) = macro_definition {
            options.add_macro_definition(macro_name, Some(value));
        }
        let artifact = compiler
            .compile_into_spirv(
                source,
                shaderc::ShaderKind::Compute,
                &path.to_string_lossy(),
                "main",
                Some(&options),
            )
            .unwrap_or_else(|error| {
                panic!("build: shader '{variant}' failed to compile:\n{error}")
            });
        let output = out_dir.join(format!("{variant}.spv"));
        std::fs::write(&output, artifact.as_binary_u8())
            .unwrap_or_else(|_| panic!("build: cannot write '{}'", output.display()));
    }
}

fn collect_comp(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let path = entry.expect("build: cannot read shader dir entry").path();
        if path.is_dir() {
            collect_comp(&path, out);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("comp") {
            out.push(path);
        }
    }
}
