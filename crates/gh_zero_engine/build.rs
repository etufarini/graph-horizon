/*
 * gh_zero_engine — build script
 * Compiles every GLSL compute shader (each `.comp` found RECURSIVELY under
 * `src/backend/vulkan/shaders/`, including the per-family subfolders) to a
 * SPIR-V module in OUT_DIR at build time, using libshaderc (bundled by the
 * `shaderc` build-dependency: no system tool required). The output stays FLAT,
 * keyed by stem (`OUT_DIR/{stem}.spv`), regardless of the source folder depth,
 * so the crate's `include_bytes!` paths never change. No shader is ever compiled
 * at runtime. This runs only for profiles containing Vulkan; CPU exits quietly.
 *
 * A missing source directory or a GLSL compilation error fails the build with a
 * clear message instead of producing a silently broken artifact.
*/

// No-op for the `cpu` backend (no build step). Build scripts are
// compiled with the package's active feature cfgs, so when Vulkan is absent
// `main` is empty and nothing references the optional shaderc dependency.
#[cfg(not(any(feature = "vulcan", feature = "vulcan-hybrid")))]
fn main() {}

#[cfg(any(feature = "vulcan", feature = "vulcan-hybrid"))]
fn main() {
    use std::path::Path;

    let shader_dir = Path::new("src/backend/vulkan/shaders");
    println!("cargo:rerun-if-changed={}", shader_dir.display());

    // No shaders yet (e.g. early build): nothing to compile, succeed quietly.
    // The walk collects every `.comp` under the subtree; a missing root dir
    // yields an empty list, preserving the prior quiet-success behaviour.
    let mut shaders = Vec::new();
    collect_comp(shader_dir, &mut shaders);

    let compiler = shaderc::Compiler::new().expect("shaderc: cannot create compiler");
    let out_dir = std::env::var("OUT_DIR").expect("build: OUT_DIR not set");

    // Target Vulkan 1.3 (SPIR-V 1.6) so kernels may use, besides subgroup
    // arithmetic (GL_KHR_shader_subgroup_*, SPIR-V >= 1.3), cooperative matrix
    // (SPV_KHR_cooperative_matrix, used by the prefill tensor-core path) and
    // integer dot product (SPV_KHR_integer_dot_product, used by mmvq decode). The
    // runtime device is created with
    // VK_API_VERSION_1_3 to consume these modules. Raising the target only lifts
    // the SPIR-V version header for the pre-existing (non-coopmat/non-dp4a)
    // shaders; their generated code — and thus their runtime output — is unchanged.
    let mut options = shaderc::CompileOptions::new().expect("shaderc: cannot create options");
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_3 as u32,
    );

    for path in shaders {
        println!("cargo:rerun-if-changed={}", path.display());

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("build: invalid shader file name");
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("build: cannot read shader '{}'", path.display()));

        // Compute stage; GLSL → SPIR-V. A compilation error aborts the build.
        let artifact = compiler
            .compile_into_spirv(
                &source,
                shaderc::ShaderKind::Compute,
                &path.to_string_lossy(),
                "main",
                Some(&options),
            )
            .unwrap_or_else(|e| panic!("build: shader '{name}' failed to compile:\n{e}"));

        let out = Path::new(&out_dir).join(format!("{name}.spv"));
        std::fs::write(&out, artifact.as_binary_u8())
            .unwrap_or_else(|_| panic!("build: cannot write '{}'", out.display()));
    }
}

// Depth-first collection of every `.comp` under `dir`. A non-existent or
// unreadable directory contributes nothing (preserves the quiet-success path);
// empty subfolders are simply skipped. Output is flat per stem, so callers must
// keep shader stems unique across the whole subtree.
#[cfg(any(feature = "vulcan", feature = "vulcan-hybrid"))]
fn collect_comp(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries {
        let path = entry.expect("build: cannot read shader dir entry").path();
        if path.is_dir() {
            collect_comp(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("comp") {
            out.push(path);
        }
    }
}
