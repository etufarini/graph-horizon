<!--
These draft notes describe the intended user-facing v0.1.0 contract,
installation, capabilities, and limitations. Qualification mechanics belong in
docs/validation.md.
-->

# Graph Horizon v0.1.0 draft

Graph Horizon v0.1.0 will be a local text-to-text runtime with an interactive
console, OpenAI-compatible streaming chat, and a Web UI. Backend choice is
fixed when the binary is built. Ministral 3 2512 is the currently integrated
model family, not part of the product name.

The preliminary candidate qualified five exact Q4_K_M artifacts: 3B, 8B, and
14B Instruct plus 3B and 14B Reasoning. Final claims must be refreshed against
the source commit selected for release and recorded in
[validation.md](validation.md).

In that preliminary campaign, 8B Reasoning was `NOT SUPPORTED`: teacher-forced
parity passed, but three fixed-seed fresh-process runs produced materially
different response bytes and lengths and failed the predeclared reproducibility
gate. The repaired runtime later passed the six-artifact technical matrix and
qualified all six semantic rows. The old byte-reproducibility release gate has
not been rerun on a final commit, so this draft does not yet include or exclude
8B Reasoning from the eventual release contract.

After the repository, tag, and release assets are public, install with:

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
only Q4_K_M GGUF models are accepted, while Q8 and MoE are outside scope.

These preliminary notes do not assign the current backend labels. In the source
being prepared now, CPU is the reference path, standalone Vulkan is production
within its declared Linux/NVIDIA-or-AMD scope, and Vulkan-hybrid, Metal, and
Metal-hybrid are qualified only within their documented tuples. The exact
scopes and any remaining fresh-campaign requirements are maintained in
[backend.md](backend.md#support-status), not in this historical draft.

The final release will apply only to the immutable source revision, model
checksums, backend, and environment recorded by its completed validation
matrix. Every other configuration remains experimental, unavailable, or
unsupported as documented.
