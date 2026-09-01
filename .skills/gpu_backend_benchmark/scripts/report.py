#!/usr/bin/env python3
"""Generate the public Markdown report from structured benchmark results."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from chart import write_chart


LABELS = {"cuda": "CUDA", "vulkan": "Vulkan", "metal": "Metal"}
WORKLOADS = {"short": "Short", "medium": "Medium", "long": "Long"}


def safe(value: Any) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|").replace("`", "'")


def value(number: float) -> str:
    return f"{number:,.2f}"


def performance_table(data: dict[str, Any]) -> list[str]:
    lines = [
        "| Workload | Backend | Prefill tok/s | Decode tok/s | TTFT ms | Runs |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in data["summaries"]:
        lines.append(
            f"| {WORKLOADS.get(row['workload'], row['workload'])} | "
            f"{LABELS.get(row['backend'], row['backend'])} | {value(row['prefill_tps'])} | "
            f"{value(row['decode_tps'])} | {value(row['ttft_ms'])} | {row['samples']} |"
        )
    return lines


def indexed_summaries(data: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["workload"], row["backend"]): row for row in data["summaries"]}


def advantage(cuda: float, vulkan: float, lower_is_better: bool = False) -> float:
    return ((vulkan - cuda) / vulkan) * 100 if lower_is_better else ((cuda - vulkan) / vulkan) * 100


def metric_lead(cuda: float, vulkan: float, lower_is_better: bool) -> tuple[str, float]:
    if cuda == vulkan:
        return "Neither backend", 0.0
    cuda_wins = cuda < vulkan if lower_is_better else cuda > vulkan
    winner, loser = (cuda, vulkan) if cuda_wins else (vulkan, cuda)
    delta = (loser - winner) / loser * 100 if lower_is_better else (winner - loser) / loser * 100
    return ("CUDA" if cuda_wins else "Vulkan"), delta


def comparison_section(data: dict[str, Any]) -> list[str]:
    check = data["comparison"]
    if not check.get("applicable", True):
        return ["CUDA/Vulkan comparison is not applicable on macOS; the Metal results stand alone."]
    if not check["valid"]:
        lines = ["The CUDA/Vulkan comparison is **invalid or incomplete**; no percentage winner is reported."]
        lines.extend(f"- {reason}." for reason in check["reasons"])
        return lines
    rows = indexed_summaries(data)
    lines = [
        "Percentages use Vulkan as the table baseline: positive means CUDA performed better; negative means it performed worse.",
        "",
        "| Workload | CUDA prefill advantage | CUDA decode advantage | CUDA TTFT advantage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for workload in ("short", "medium", "long"):
        cuda, vulkan = rows[(workload, "cuda")], rows[(workload, "vulkan")]
        lines.append(
            f"| {WORKLOADS[workload]} | {advantage(cuda['prefill_tps'], vulkan['prefill_tps']):+.1f}% | "
            f"{advantage(cuda['decode_tps'], vulkan['decode_tps']):+.1f}% | "
            f"{advantage(cuda['ttft_ms'], vulkan['ttft_ms'], True):+.1f}% |"
        )
    conclusions = (
        ("Prefill-heavy", "long", "prefill_tps", False),
        ("Decode-heavy", "long", "decode_tps", False),
        ("Latency-sensitive", "short", "ttft_ms", True),
    )
    lines.extend(["", "Concise workload conclusions:"])
    for label, workload, metric, lower in conclusions:
        cuda, vulkan = rows[(workload, "cuda")], rows[(workload, "vulkan")]
        winner, delta = metric_lead(cuda[metric], vulkan[metric], lower)
        verb = "is tied" if delta == 0 else f"leads by {delta:.1f}%"
        lines.append(f"- {label}: {winner} {verb} on the representative {workload} metric.")
    lines.append("- These are metric-specific results, not an absolute overall winner.")
    return lines


def configuration(data: dict[str, Any]) -> list[str]:
    cfg, model = data["configuration"], data["model"]
    context = f"{cfg['context_size']} tokens"
    if cfg.get("requested_context_size", cfg["context_size"]) != cfg["context_size"]:
        context += f" (capped from {cfg['requested_context_size']} by the common model limit)"
    lines = [
        "| Parameter | Value |", "| --- | --- |",
        f"| Model | `{safe(model['name'])}` |", f"| Model SHA-256 | `{safe(model['sha256'])}` |",
        f"| Runtime model identifier(s) | `{safe(', '.join(model['runtime_identifiers']))}` |",
        f"| Quantization / precision | `{safe(model['quantization'])}` |",
        f"| Runtime(s) | `{safe(', '.join(cfg['runtimes']))}` |",
        f"| Runtime configuration | `{safe(', '.join(cfg['runtime_configs']))}` |",
        f"| Context size | {context} |", f"| Batch size | {cfg['batch_size']} |",
        f"| Sampling | temperature {cfg['temperature']}, seed {cfg['seed']} |",
        f"| Warm-ups | {cfg['warmups']} per backend/workload |",
        f"| Measured runs | {cfg['repetitions']} per backend/workload |",
        f"| Runtime options hash | `{cfg['adapter_args_sha256']}` |",
    ]
    for workload in data["workloads"]:
        adjustment = workload.get("adjustment") or "none"
        lines.append(
            f"| {WORKLOADS[workload['name']]} workload | {workload['actual_prompt_tokens']} prompt + "
            f"{workload['generated_tokens']} generated tokens; adjustment: {adjustment} |"
        )
    return lines


def failures(data: dict[str, Any]) -> list[str]:
    entries = []
    for backend, status in data.get("backend_status", {}).items():
        if status["status"] != "completed":
            entries.append(f"- {LABELS.get(backend, backend)}: {status['status']} — {status['reason']}.")
    entries.extend(
        f"- {WORKLOADS.get(item['workload'], item['workload'])} / {LABELS.get(item['backend'], item['backend'])}: {item['reason']}."
        for item in data.get("failures", [])
    )
    return entries or ["No skipped backends or incomplete workloads."]


def markdown(data: dict[str, Any]) -> str:
    machine = data["machine"]
    lines = [
        "# GPU Backend Benchmark", "",
        f"**Machine:** {safe(machine['machine'])}  ", f"**OS:** {safe(machine['os'])}  ",
        f"**CPU:** {safe(machine['cpu'])}  ", f"**GPU / SoC:** {safe(machine['gpu'])}  ",
        f"**GPU memory:** {safe(machine['memory'])}  ", f"**GPU driver:** {safe(machine['gpu_driver'])}  ",
        f"**Available runtime backends:** {', '.join(LABELS.get(name, name) for name in data['runtime_backends']) or 'none'}",
        "", "## Configuration", "", *configuration(data), "", "## Performance", "",
        *performance_table(data), "", "Medians are calculated from the individual measurements in `benchmark-results.json`.",
        "", "## Backend Comparison", "", *comparison_section(data), "", "## Availability and Failures", "",
        *failures(data), "", "## Reproduction Notes", "",
        f"Generated at `{data['generated_at']}`. CUDA version: `{machine.get('cuda_version') or 'n/a'}`; "
        f"Vulkan version: `{machine.get('vulkan_version') or 'n/a'}`; Metal runtime: `{machine.get('metal_runtime') or 'n/a'}`.",
        "The raw JSON preserves every run, actual token count, prompt text and hash, timing, runtime identity, and comparison check.",
        "The PNG contains separate scales for prefill throughput, decode throughput, and TTFT.", "",
    ]
    return "\n".join(lines)


def generate(data: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "benchmark-results.md").write_text(markdown(data), encoding="utf-8")
    write_chart(data, output_dir / "benchmark-results.png")


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            raise ValueError
        source, destination = Path(sys.argv[1]), Path(sys.argv[2])
        generate(json.loads(source.read_text(encoding="utf-8")), destination)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        print("report: invalid results or output", file=sys.stderr)
        raise SystemExit(2) from None
