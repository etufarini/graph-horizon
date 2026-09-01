/*
 * graph_horizon_engine — offline Metal source compilation and library linking.
 */

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(crate) fn build() {
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

fn find_tool(name: &str) -> PathBuf {
    let output = Command::new("xcrun")
        .args(["--find", name])
        .output()
        .unwrap_or_else(|_| toolchain_unavailable());
    let path = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !output.status.success() || path.is_empty() {
        toolchain_unavailable();
    }
    path.into()
}

fn toolchain_unavailable() -> ! {
    panic!(
        "Metal toolchain unavailable: install the Metal Toolchain and ensure xcrun can find metal and metallib"
    )
}

fn bounded(bytes: &[u8]) -> String {
    const LIMIT: usize = 8 * 1024;
    String::from_utf8_lossy(&bytes[..bytes.len().min(LIMIT)]).into_owned()
}
