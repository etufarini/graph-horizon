# GPU Backend Benchmark

**Machine:** Linux x86_64 CUDA validation host

**OS:** Linux 7.0.0-30-generic

**CPU:** AMD Ryzen 7 3800X 8-Core Processor

**GPU / SoC:** NVIDIA CUDA device

**GPU memory:** 6 GB VRAM

**GPU driver:** 595.84

**Available runtime backends:** CUDA, Vulkan

## Configuration

| Parameter | Value |
| --- | --- |
| Model | `Ministral 3 3B Instruct 2512` |
| Model SHA-256 | `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` |
| Runtime model identifier(s) | `ministral-3B-Instruct-2512` |
| Quantization / precision | `Q4_K_M` |
| Runtime(s) | `Graph Horizon 0.1.4+e6c4572-benchmark` |
| Runtime configuration | `greedy; forced max tokens; f16 KV; all-gpu; release` |
| Context size | 4608 tokens |
| Batch size | 1 |
| Sampling | temperature 0, seed 1 |
| Warm-ups | 1 per backend/workload |
| Measured runs | 3 per backend/workload |
| Runtime options hash | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Short workload | 128 prompt + 128 generated tokens; adjustment: none |
| Medium workload | 1024 prompt + 256 generated tokens; adjustment: none |
| Long workload | 4096 prompt + 512 generated tokens; adjustment: none |

## Performance

| Workload | Backend | Prefill tok/s | Decode tok/s | TTFT ms | Runs |
| --- | --- | ---: | ---: | ---: | ---: |
| Short | CUDA | 0.84 | 0.61 | 153,105.57 | 3 |
| Short | Vulkan | 249.51 | 43.51 | 522.98 | 3 |
| Medium | CUDA | 0.83 | 0.55 | 1,231,837.21 | 3 |
| Medium | Vulkan | 323.54 | 42.60 | 3,175.77 | 3 |
| Long | Vulkan | 291.47 | 36.39 | 14,063.86 | 3 |

Medians are calculated from the individual measurements in `benchmark-results.json`.

## Backend Comparison

The CUDA/Vulkan comparison is **invalid or incomplete**; no percentage winner is reported.
- CUDA and Vulkan were not both completed.

## Availability and Failures

- CUDA: incomplete — one or more workloads failed.
- Long / CUDA: runtime adapter timed out.

## Reproduction Notes

Generated at `2026-09-02T20:09:59.865051+00:00`. CUDA version: `13.2`; Vulkan version: `1.4.341`; Metal runtime: `n/a`.
The raw JSON preserves every run, actual token count, prompt text and hash, timing, runtime identity, and comparison check.
The PNG contains separate scales for prefill throughput, decode throughput, and TTFT.
