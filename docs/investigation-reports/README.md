<!--
This index owns navigation for historical performance and cleanup campaigns.
The reports preserve evidence at recorded revisions and do not define current
runtime support.
-->

# Investigation Reports

These reports contain authenticated inputs, measurements, candidate decisions,
qualification gates, and integration records from completed investigations.
Current conclusions are summarized in
[current performance status](../project-status/current-performance-status.md)
and [validation evidence](../project-status/validation-evidence.md).

| Report | Scope |
|---|---|
| [CUDA Amdahl optimization](cuda-amdahl-optimization.md) | Matmul and attention attribution, retained CUDA kernels, rejected candidates, and quantitative stop |
| [CPU Amdahl optimization](cpu-amdahl-optimization.md) | Short/medium/long CPU attribution, rejected AVX-512 row blocking, and quantitative stop |
| [NVIDIA Vulkan long-context prefill](vulkan-nvidia-long-context-prefill.md) | Prefill attribution, routing correction, and quantitative stop |
| [NVIDIA Vulkan long-context decode](vulkan-nvidia-long-context-decode.md) | Decode attribution, routing, residency, and reranking |
| [AMD Vulkan long-context prefill](vulkan-amd-long-context-prefill.md) | Split-GQA route and qualification |
| [AMD Vulkan long-context decode](vulkan-amd-long-context-decode.md) | Decode measurements, routing, and retained INT8 specialization |
| [Metal long-context prefill](metal-long-context-prefill.md) | Attention optimization and final checkpoint |
| [Metal long-context decode](metal-long-context-decode.md) | Candidate ledger and practical floors |
| [Metal and llama.cpp optimization](metal-llamacpp-optimization.md) | Competitive matrix and gap attribution |
| [Metal and Vulkan optimization parity](metal-vulkan-optimization-parity.md) | Capability mapping and retained parity candidates |
| [Backend and model-family cleanup](final-backend-model-family-cleanup.md) | Necessity inventory, cleanup decisions, and qualification snapshot |
| [AMD cleanup-regression repair](amd-deep-clean-regression-repair.md) | Correctness diagnosis, repair, and requalification |
