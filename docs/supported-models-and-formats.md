<!--
This reference owns the user-facing model-family, artifact-format, and
quantization contract. Numeric implementation details remain in the engine
crate documentation; qualification evidence remains in the validation record.
-->

# Supported Models And Formats

Graph Horizon supports the Ministral 3 Instruct and Reasoning 2512 models in
the 3B, 8B, and 14B sizes.

Download the `Q4_K_M` GGUF file from one of these repositories:

| Size | Instruct | Reasoning |
|---|---|---|
| 3B | [Ministral 3 3B Instruct](https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF) | [Ministral 3 3B Reasoning](https://huggingface.co/unsloth/Ministral-3-3B-Reasoning-2512-GGUF) |
| 8B | [Ministral 3 8B Instruct](https://huggingface.co/unsloth/Ministral-3-8B-Instruct-2512-GGUF) | [Ministral 3 8B Reasoning](https://huggingface.co/unsloth/Ministral-3-8B-Reasoning-2512-GGUF) |
| 14B | [Ministral 3 14B Instruct](https://huggingface.co/unsloth/Ministral-3-14B-Instruct-2512-GGUF) | [Ministral 3 14B Reasoning](https://huggingface.co/unsloth/Ministral-3-14B-Reasoning-2512-GGUF) |

Only the public `Q4_K_M` profile is accepted: Q4_K/Q6_K matrices and
one-dimensional F32 auxiliaries. Other GGUF quantization profiles, including
Q8, are rejected before backend allocation. Internal numeric formats used by a
backend do not expand this public artifact contract.

Recognition is capability-based. A filename, directory, total byte size, or
display name does not authorize a model. Loading validates the GGUF
architecture, metadata, tokenizer, tensor names, shapes, and types.

The exact engine-facing contract is in the
[engine crate README](../crates/graph_horizon_engine/README.md). Reviewed
artifact identities and qualification results are in the
[validation evidence](project-status/validation-evidence.md).
