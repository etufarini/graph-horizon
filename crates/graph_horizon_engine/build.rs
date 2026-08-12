/*
 * graph_horizon_engine — offline shader build: compiles trusted Vulkan/Metal sources and variants into build outputs; it performs no runtime compilation and reads no model data.
 */

fn main() {
    #[cfg(any(feature = "vulkan", feature = "vulkan-hybrid"))]
    build_vulkan();
    #[cfg(any(feature = "metal", feature = "metal-hybrid"))]
    build_metal();
}

#[cfg(any(feature = "vulkan", feature = "vulkan-hybrid"))]
fn build_vulkan() {
    use std::path::Path;

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
        compile_vulkan_shader(&compiler, &path, &source, name, Path::new(&out_dir));
    }
}

#[cfg(any(feature = "vulkan", feature = "vulkan-hybrid"))]
fn compile_vulkan_shader(
    compiler: &shaderc::Compiler,
    path: &std::path::Path,
    source: &str,
    name: &str,
    out_dir: &std::path::Path,
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

#[cfg(any(feature = "vulkan", feature = "vulkan-hybrid"))]
fn collect_comp(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
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

#[cfg(any(feature = "metal", feature = "metal-hybrid"))]
fn build_metal() {
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64")
    {
        panic!("metal profiles require macOS on Apple Silicon");
    }

    let metal = find_tool("metal");
    let metallib = find_tool("metallib");
    let source_dir = Path::new("src/backend/metal/shaders");
    println!("cargo:rerun-if-changed={}", source_dir.display());
    let names = [
        "embedding.metal",
        "matmul.metal",
        "rmsnorm.metal",
        "rope.metal",
        "silu_mul.metal",
        "residual_add.metal",
        "kv_write.metal",
        "attention.metal",
        "argmax.metal",
        "topk.metal",
    ];
    let mut stems = BTreeSet::new();
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("build: OUT_DIR not set"));
    let mut objects = Vec::with_capacity(names.len());

    for name in names {
        let source = source_dir.join(name);
        println!("cargo:rerun-if-changed={}", source.display());
        let stem = source
            .file_stem()
            .and_then(|value| value.to_str())
            .expect("build: invalid Metal shader name")
            .to_owned();
        if !stems.insert(stem.clone()) {
            panic!("Metal shader compilation failed: {name}");
        }
        let object = out_dir.join(format!("{stem}.air"));
        let output = Command::new(&metal)
            .args(["-std=metal4.0", "-c"])
            .arg(&source)
            .arg("-o")
            .arg(&object)
            .output()
            .unwrap_or_else(|_| toolchain_unavailable());
        if !output.status.success() {
            panic!(
                "Metal shader compilation failed: {name}\n{}",
                bounded(&output.stderr)
            );
        }
        objects.push(object);
    }

    let library = out_dir.join("graph_horizon_engine.metallib");
    let output = Command::new(&metallib)
        .args(&objects)
        .arg("-o")
        .arg(&library)
        .output()
        .unwrap_or_else(|_| toolchain_unavailable());
    if !output.status.success() {
        panic!("Metal library link failed\n{}", bounded(&output.stderr));
    }
}

#[cfg(any(feature = "metal", feature = "metal-hybrid"))]
fn find_tool(name: &str) -> std::path::PathBuf {
    let output = std::process::Command::new("xcrun")
        .args(["--find", name])
        .output()
        .unwrap_or_else(|_| toolchain_unavailable());
    let path = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !output.status.success() || path.is_empty() {
        toolchain_unavailable();
    }
    path.into()
}

#[cfg(any(feature = "metal", feature = "metal-hybrid"))]
fn toolchain_unavailable() -> ! {
    panic!(
        "Metal toolchain unavailable: install the Metal Toolchain and ensure xcrun can find metal and metallib"
    )
}

#[cfg(any(feature = "metal", feature = "metal-hybrid"))]
fn bounded(bytes: &[u8]) -> String {
    const LIMIT: usize = 8 * 1024;
    String::from_utf8_lossy(&bytes[..bytes.len().min(LIMIT)]).into_owned()
}
