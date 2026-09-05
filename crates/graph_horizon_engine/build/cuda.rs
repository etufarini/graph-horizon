/*
 * graph_horizon_engine — trusted CUDA sources compiled offline to embedded PTX.
 */

use std::path::{Path, PathBuf};
use std::process::Command;

pub(crate) fn build() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux")
        || std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("x86_64")
    {
        panic!("cuda profile requires Linux x86_64");
    }

    let source_dir = Path::new("src/backend/cuda/shaders");
    let mut sources = Vec::new();
    collect_sources(source_dir, &mut sources);
    sources.sort();
    for source in sources {
        println!("cargo:rerun-if-changed={}", source.display());
    }

    // Fixed images preserve75 admission while allowing80-only instructions where supported.
    let output_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("build: OUT_DIR not set"));
    for (architecture, filename) in [
        ("--gpu-architecture=compute_75", "cuda_kernels.ptx"),
        ("--gpu-architecture=compute_80", "cuda_kernels_80.ptx"),
    ] {
        let output_path = output_dir.join(filename);
        let output = Command::new("nvcc")
            .args(["--ptx", "--std=c++17", "-O3", architecture, "--output-file"])
            .arg(&output_path)
            .arg(source_dir.join("kernels.cu"))
            .output()
            .unwrap_or_else(|_| compiler_unavailable());
        if !output.status.success() {
            panic!(
                "CUDA kernel compilation failed\n{}",
                bounded(&output.stderr)
            );
        }
        if std::fs::metadata(output_path).map_or(true, |metadata| metadata.len() == 0) {
            panic!("CUDA kernel compilation failed");
        }
    }
}

fn collect_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|_| {
        panic!("CUDA kernel compilation failed");
    });
    for entry in entries {
        let path = entry
            .unwrap_or_else(|_| panic!("CUDA kernel compilation failed"))
            .path();
        if path.is_dir() {
            collect_sources(&path, out);
        } else if matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("cu" | "cuh")
        ) {
            out.push(path);
        }
    }
}

fn compiler_unavailable() -> ! {
    panic!("CUDA compiler unavailable: install a supported CUDA Toolkit with nvcc")
}

fn bounded(bytes: &[u8]) -> String {
    const LIMIT: usize = 8 * 1024;
    let text = String::from_utf8_lossy(&bytes[..bytes.len().min(LIMIT)]);
    if text.len() <= LIMIT {
        return text.into_owned();
    }
    let mut end = LIMIT;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_owned()
}
