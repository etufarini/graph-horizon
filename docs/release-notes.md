<!--
These notes describe the released user-facing v0.1.0 contract, installation,
capabilities, and limitations. Qualification mechanics belong in
docs/validation.md.
-->

# Graph Horizon v0.1.0

Graph Horizon v0.1.0 is a local text-to-text runtime with an interactive console
and a Web UI. Backend choice is
fixed when the binary is built. Ministral 3 2512 is the currently integrated
model family, not part of the product name.

The final tag-derived campaign authenticated and semantically qualified all six
exact Q4_K_M artifacts: 3B, 8B, and 14B Instruct plus 3B, 8B, and 14B Reasoning.
Its real-model matrix completed with 37 pass, 37 external verification, and no
failure. Exact model identities, environment, and evidence boundaries are in
[validation.md](validation.md).

The tag and release assets are published. While the repository remains private,
use an authenticated tagged checkout and the local installer; after it becomes
public, install anonymously with:

```sh
curl --fail --location --silent --show-error https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.0/install.sh | bash -s -- --backend vulkan-hybrid
```

The bootstrap downloads the immutable source asset and `.sha256`, verifies it
before extraction, then builds. It never downloads a model. A tagged checkout
can instead run `./support/install.sh --backend vulkan-hybrid`.

Capabilities include local CLI inference, private Web streaming, sequential
requests without restart, a browser UI, F16/int8 KV selection, explicit context
sizing, and CPU/device/hybrid planning.

Known limitations: one generation at a time; compile-time backend selection; no
runtime fallback; no tools, multimodal input, or separate reasoning channel;
only Q4_K_M GGUF models are accepted, while Q8 and MoE are outside scope.

CPU is the reference path, standalone Vulkan is production within its declared
Linux/NVIDIA-or-AMD scope, and Vulkan-hybrid, Metal, and Metal-hybrid are
qualified only within their documented tuples. The exact scopes are maintained
in [backend.md](backend.md#support-status).

The release applies only to the immutable source revision, model
checksums, backend, and environment recorded by its completed validation
matrix. Every other configuration remains experimental, unavailable, or
unsupported as documented.
