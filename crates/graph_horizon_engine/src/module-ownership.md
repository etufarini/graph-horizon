<!--
This map assigns one ownership boundary to each engine source domain. It does
not define runtime behavior or record validation results.
-->

# Engine Module Ownership

- `api/`: public types, configuration, requests, roles, and events.
- `family/mod.rs`: closed architecture dispatch and family-neutral engine delegation.
- `family/mistral/detect.rs`: architecture and Q4_K_M gate before backend allocation.
- `family/mistral/config.rs`: metadata and derived dimensions.
- `family/mistral/version.rs`: Ministral 3 Instruct/Reasoning 2512 values and the fixed Reasoning system prompt; no dispatch.
- `family/mistral/tensors.rs`: accepted Q4_K_M names, shapes, and dtypes.
- `family/mistral/tokenizer/profile.rs`: private chat-policy selection from the three exact Reasoning names without authenticating GGUF.
- `family/mistral/tokenizer/reasoning.rs`: encoding of the internal Reasoning-system-prompt markers only.
- `family/mistral/tokenizer/`: Tekken BPE and pre-tokenization embedded in GGUF.
- `family/mistral/template.rs`: fixed chat sequence and implicit/explicit system rendering without Jinja execution.
- `family/mistral/parity.rs`: bounded oracle-vector parsing and model-neutral top-two criterion for the harness only.
- `tests/semantic.rs`: test-only Reasoning qualifier owning corpus, sampling, scoring, stop telemetry, and markers.
- `family/mistral/graph/`: dense operation order shared by backends; no placement decision.
- `runtime/`: homogeneous or partitioned lifecycle; partitioned execution owns one CPU prefix, one crossing, and one device suffix.
- `backend/`: tensor contract, weight source, static selection, and CPU/Vulkan/Metal implementations; `backend/hybrid/` owns placement and both resource sets.
- `gguf/`: bounded parsing, metadata, and tensor index, including Q8_0 diagnosis and rejection; no backend allocation.
- `kv_cache/`: f16/int8 scheme, layout, and request lifecycle.
- `sampling.rs`: deterministic or sampled token selection and private RNG.
- `harness/`: model-neutral measurements, parity report, and external checks; not runtime code.

```text
GGUF -> family::Model -> family/WeightSource -> runtime <- backend -> decoder -> events
```

Invariants:

- the complete artifact contract is validated before backend allocation;
- a hybrid plan is immutable after loading;
- each layer's KV cache follows that layer;
- mixed execution crosses the residual once;
- cancellation and errors release resources without exposing internal details.
