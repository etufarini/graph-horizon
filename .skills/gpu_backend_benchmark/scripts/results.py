#!/usr/bin/env python3
"""Aggregate raw GPU benchmark runs and prove comparison compatibility."""

from __future__ import annotations

import hashlib
import math
import re
import statistics
from typing import Any


METRICS = ("prefill_tps", "decode_tps", "ttft_ms")


def enrich_run(run: dict[str, Any]) -> dict[str, Any]:
    result = dict(run)
    result["prefill_tps"] = run["prompt_tokens"] / run["prefill_seconds"]
    result["decode_tps"] = run["completion_tokens"] / run["decode_seconds"]
    for field in METRICS:
        if not math.isfinite(result[field]) or result[field] <= 0:
            raise ValueError(f"invalid {field}")
    return result


def summaries(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault((run["workload"], run["backend"]), []).append(run)
    output = []
    for (workload, backend), samples in groups.items():
        output.append({
            "workload": workload,
            "backend": backend,
            "samples": len(samples),
            **{field: statistics.median(sample[field] for sample in samples) for field in METRICS},
        })
    order = {"short": 0, "medium": 1, "long": 2}
    return sorted(output, key=lambda row: (order.get(row["workload"], 99), row["backend"]))


def normalized_device_id(value: str) -> str:
    return re.sub(r"[^0-9a-z]", "", value.lower().removeprefix("gpu-"))


def comparison(runs: list[dict[str, Any]], repetitions: int, backends: list[str]) -> dict[str, Any]:
    if sorted(backends) != ["cuda", "vulkan"]:
        return {"valid": False, "applicable": True, "reasons": ["CUDA and Vulkan were not both completed"]}
    reasons = []
    keys = (
        "model_sha256", "quantization", "prompt_sha256", "prompt_tokens",
        "completion_tokens", "context_size", "batch_size", "seed", "temperature",
        "runtime_name", "runtime_version", "runtime_config", "model_identifier",
    )
    for workload in ("short", "medium", "long"):
        rows = [run for run in runs if run["workload"] == workload]
        by_backend = {name: [run for run in rows if run["backend"] == name] for name in backends}
        if any(len(values) != repetitions for values in by_backend.values()):
            reasons.append(f"{workload} does not contain {repetitions} runs per backend")
            continue
        for key in keys:
            values = {str(run[key]) for run in rows}
            if len(values) != 1:
                reasons.append(f"{workload} differs in {key}")
        device_ids = {normalized_device_id(run["device_id"]) for run in rows}
        gpu_names = {run["gpu_name"].strip().lower() for run in rows}
        if len(device_ids) != 1 or len(gpu_names) != 1:
            reasons.append(f"{workload} used different physical GPUs")
    return {"valid": not reasons, "applicable": True, "reasons": sorted(set(reasons))}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
