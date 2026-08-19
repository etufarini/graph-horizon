<!--
These notes describe only the user-facing v0.1.0 contract, installation,
capabilities, and limitations. Qualification mechanics belong in VALIDATION.md.
-->

# Graph Horizon — Ministral 3 v0.1.0

Graph Horizon v0.1.0 is a local text-to-text runtime for Ministral 3 2512 with
an interactive console, OpenAI-compatible streaming chat, and a Web UI. Backend
choice is fixed when the binary is built.

The release is qualified for five exact Q4_K_M artifacts: 3B, 8B, and 14B
Instruct plus 3B and 14B Reasoning. This applies only to the SHA-256 identified
bytes and Linux x86_64 / NVIDIA RTX 3060 / Vulkan-hybrid 100% all-GPU tuple in
[VALIDATION.md](VALIDATION.md).

8B Reasoning is not supported. Its teacher-forced parity passed, but three
fixed-seed fresh-process runs produced materially different response bytes and
lengths and failed the predeclared reproducibility gate.

Install after the repository and release assets are public:

```sh
curl --fail --location --silent --show-error https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.0/install.sh | bash -s -- --backend vulkan-hybrid
```

The bootstrap downloads the immutable source asset and `.sha256`, verifies it
before extraction, then builds. It never downloads a model. A tagged checkout
can instead run `./support/install.sh --backend vulkan-hybrid`.

Capabilities include local CLI inference, serialized SSE streaming, sequential
requests without restart, a browser UI, F16/int8 KV selection, explicit context
sizing, and CPU/device/hybrid planning.

Known limitations: serialized serving; compile-time backend selection; no
runtime fallback; no tools, multimodal input, or separate reasoning channel;
Q8 and MoE are outside scope. CPU, standalone/mixed Vulkan, Metal, AMD Vulkan,
and other devices are experimental even where they build or have historical
evidence.

Ministral 3 v0.1.0 is qualified for exactly the model artifacts, backend, and
environment listed in the v0.1.0 validation matrix. Qualification applies to
the immutable source revision and recorded model checksums. Every other
configuration is experimental, unavailable, or unsupported as documented.
