# Contributing

Follow [AGENTS.md](AGENTS.md): keep changes small, state invariants explicitly,
and add no dependency or abstraction without a concrete need.

Before making a change:

1. identify the invariant to preserve;
2. choose the smallest viable change;
3. state the main risk.

Always select the CPU feature explicitly when working on the CPU path:

```sh
cargo check --no-default-features --features cpu
cargo test --no-default-features --features cpu
```

Also verify the default build and the Vulkan/hybrid features when code touches
backends, placement, or shaders. Real-model tests are external: the local,
synthetic suite does not depend on GGUF files being present.

A new family follows the CPU-first order described under
[supported models](README.md#supported-models). Do not expand public profiles,
reduce the context, or introduce fallbacks without first updating the
specification.

Do not commit secrets, personal paths in public messages, or model artifacts.
