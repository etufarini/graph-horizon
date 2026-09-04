#!/usr/bin/env python3
"""Orchestrate reproducible same-machine GPU backend benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hardware import detect
from report import generate
from results import comparison, enrich_run, normalized_device_id, sha256_text, summaries
from runtime import AdapterError, invoke, resolve_adapter, validate_count, validate_probe, validate_run


TARGETS = (("short", 128, 128), ("medium", 1024, 256), ("long", 4096, 512))
OUTPUTS = ("benchmark-results.json", "benchmark-results.md", "benchmark-results.png")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CUDA/Vulkan or Metal through a runtime adapter")
    parser.add_argument("--adapter", required=True, help="adapter executable or command on PATH")
    parser.add_argument("--adapter-arg", action="append", default=[], help="fixed adapter argument; repeat as needed")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--context-size", type=int, default=4608)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    values = parser.parse_args()
    if values.context_size < 4 or values.batch_size < 1 or values.repetitions < 3 or values.warmups < 1 or values.seed < 0 or values.timeout < 1:
        parser.error("context >= 4, batch >= 1, repetitions >= 3, warmups >= 1, seed >= 0, and timeout >= 1 are required")
    return values


def model_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_workloads(context: int) -> list[dict[str, Any]]:
    plans = []
    for name, prompt, generated in TARGETS:
        adjustment = None
        if prompt + generated > context:
            generated = min(generated, max(2, context // 4))
            prompt = context - generated
            adjustment = f"reduced to fit the {context}-token context"
        plans.append({"name": name, "target_prompt_tokens": dict((row[0], row[1]) for row in TARGETS)[name],
                      "target_generated_tokens": dict((row[0], row[2]) for row in TARGETS)[name],
                      "desired_prompt_tokens": prompt, "generated_tokens": generated, "adjustment": adjustment})
    return plans


def adapter_arguments(model: Path, backend: str) -> list[str]:
    return ["--model", str(model.resolve()), "--backend", backend]


def calibrate_prompt(adapter: str, fixed: list[str], model: Path, backend: str, target: int, folder: Path, timeout: int) -> tuple[str, int]:
    repetitions, candidates = max(1, target), []
    prompt_path = folder / "prompt.txt"
    for _ in range(8):
        prompt = ("benchmark " * repetitions).strip() + "."
        prompt_path.write_text(prompt, encoding="utf-8")
        payload = invoke(adapter, fixed, "count", [*adapter_arguments(model, backend), "--prompt-file", str(prompt_path)], timeout)
        tokens = validate_count(payload)
        candidates.append((prompt, tokens))
        if tokens == target:
            break
        next_repetitions = max(1, round(repetitions * target / tokens))
        repetitions = repetitions + (1 if tokens < target else -1) if next_repetitions == repetitions else next_repetitions
    viable = [item for item in candidates if item[1] <= target]
    if not viable:
        raise AdapterError("context_too_large", "the runtime prompt template exceeds the planned context")
    return min(viable, key=lambda item: target - item[1])


def run_arguments(model: Path, backend: str, prompt: Path, plan: dict[str, Any], cfg: argparse.Namespace) -> list[str]:
    return [*adapter_arguments(model, backend), "--context", str(cfg.context_size), "--batch-size", str(cfg.batch_size),
            "--prompt-file", str(prompt), "--max-tokens", str(plan["generated_tokens"]),
            "--seed", str(cfg.seed), "--temperature", "0", "--json"]


def collect(payload: dict[str, Any], backend: str, plan: dict[str, Any], cfg: argparse.Namespace, common: dict[str, Any], repetition: int) -> dict[str, Any]:
    validate_run(payload, backend)
    expected = {"prompt_tokens": plan["actual_prompt_tokens"], "completion_tokens": plan["generated_tokens"],
                "context_size": cfg.context_size, "batch_size": cfg.batch_size, "seed": cfg.seed,
                "temperature": 0, "quantization": cfg.quantization}
    if any(payload[key] != expected[key] for key in expected):
        raise AdapterError("incomplete_run", "runtime returned token counts or settings different from the requested tuple")
    if backend != "metal" and normalized_device_id(payload["device_id"]) != normalized_device_id(common["hardware_gpu_id"]):
        raise AdapterError("incomplete_run", "runtime used a different GPU than hardware detection")
    return enrich_run({**payload, **common, "prompt_sha256": plan["prompt_sha256"],
                       "workload": plan["name"], "backend": backend, "repetition": repetition})


def redact_hardware_identity(data: dict[str, Any]) -> dict[str, Any]:
    public = json.loads(json.dumps(data))
    metal = "metal" in public["runtime_backends"]
    public["machine"]["machine"] = "Apple silicon validation host" if metal else "GPU validation host"
    public["machine"]["gpu"] = "Apple silicon GPU" if metal else "NVIDIA GPU"
    public["machine"].pop("gpu_id", None)
    for run in public["runs"]:
        for field in ("device_id", "gpu_name", "hardware_gpu_id"):
            run.pop(field, None)
    return public


def main() -> int:
    cfg = arguments()
    model = cfg.model.expanduser()
    if not model.is_file():
        raise AdapterError("missing_model", "model is missing or unreadable")
    adapter = resolve_adapter(cfg.adapter)
    output = cfg.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in OUTPUTS):
        raise AdapterError("output_exists", "output files already exist; choose a fresh output directory")
    machine, probes, status = detect(), {}, {}
    for backend, hardware in machine["backends"].items():
        if not hardware["available"]:
            status[backend] = {"status": "unavailable", "reason": hardware["reason"]}
            continue
        try:
            probe = validate_probe(invoke(adapter, cfg.adapter_arg, "probe", adapter_arguments(model, backend), cfg.timeout), backend)
            if probe["model"]["quantization"].lower() != cfg.quantization.lower():
                raise AdapterError("model_load", "adapter quantization does not match --quantization")
            if backend != "metal" and normalized_device_id(probe["device"]["id"]) != normalized_device_id(machine["gpu_id"]):
                raise AdapterError("unsupported_backend", "adapter selected a different GPU than hardware detection")
            probes[backend], status[backend] = probe, {"status": "ready", "reason": ""}
        except AdapterError as error:
            status[backend] = {"status": "unavailable", "reason": str(error)}
    if not probes:
        reasons = "; ".join(f"{name}: {item['reason']}" for name, item in status.items())
        raise AdapterError("unsupported_backend", f"no supported runtime backend is available ({reasons})")
    requested_context_size = cfg.context_size
    cfg.context_size = min(cfg.context_size, *(probe["model"]["context_limit"] for probe in probes.values()))
    plans, runs, failures = plan_workloads(cfg.context_size), [], []
    common = {"model_sha256": model_hash(model), "hardware_gpu_id": machine["gpu_id"], "batch_size": cfg.batch_size,
              "context_size": cfg.context_size, "seed": cfg.seed, "temperature": 0}
    with tempfile.TemporaryDirectory(prefix="gpu-benchmark-") as temporary:
        folder, calibration_backend = Path(temporary), next(iter(probes))
        for plan in plans:
            prompt, tokens = calibrate_prompt(adapter, cfg.adapter_arg, model, calibration_backend, plan["desired_prompt_tokens"], folder, cfg.timeout)
            plan.update({"prompt": prompt, "actual_prompt_tokens": tokens, "prompt_sha256": sha256_text(prompt)})
            if tokens != plan["desired_prompt_tokens"]:
                plan["adjustment"] = (plan["adjustment"] + "; " if plan["adjustment"] else "") + f"tokenizer produced {tokens} prompt tokens"
            prompt_path = folder / f"{plan['name']}.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            for backend in probes:
                try:
                    args = run_arguments(model, backend, prompt_path, plan, cfg)
                    for _ in range(cfg.warmups):
                        collect(invoke(adapter, cfg.adapter_arg, "run", args, cfg.timeout), backend, plan, cfg, common, 0)
                    for repetition in range(1, cfg.repetitions + 1):
                        runs.append(collect(invoke(adapter, cfg.adapter_arg, "run", args, cfg.timeout), backend, plan, cfg, common, repetition))
                except AdapterError as error:
                    failures.append({"backend": backend, "workload": plan["name"], "reason": str(error)})
    aggregate = summaries(runs)
    complete = [backend for backend in probes if all(any(row["backend"] == backend and row["workload"] == name and row["samples"] == cfg.repetitions for row in aggregate) for name, _, _ in TARGETS)]
    for backend in probes:
        status[backend] = {"status": "completed" if backend in complete else "incomplete", "reason": "" if backend in complete else "one or more workloads failed"}
    if not aggregate:
        raise AdapterError("incomplete_run", "no complete benchmark measurements were collected")
    check = comparison(runs, cfg.repetitions, complete)
    check["applicable"] = "metal" not in machine["backends"]
    data = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "machine": machine,
            "model": {"name": cfg.model_name or model.name, "sha256": common["model_sha256"], "quantization": cfg.quantization,
                      "runtime_identifiers": sorted({run["model_identifier"] for run in runs})},
            "configuration": {"context_size": cfg.context_size, "requested_context_size": requested_context_size,
                              "batch_size": cfg.batch_size, "seed": cfg.seed, "temperature": 0,
                              "repetitions": cfg.repetitions, "warmups": cfg.warmups,
                              "runtimes": sorted({f"{run['runtime_name']} {run['runtime_version']}" for run in runs}),
                              "runtime_configs": sorted({run["runtime_config"] for run in runs}),
                              "adapter_args_sha256": sha256_text("\0".join(cfg.adapter_arg))},
            "workloads": plans, "runs": runs, "summaries": aggregate, "comparison": check,
            "runtime_backends": sorted(probes), "completed_backends": complete,
            "backend_status": status, "failures": failures}
    data = redact_hardware_identity(data)
    (output / OUTPUTS[0]).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    generate(data, output)
    if not complete:
        print("gpu-benchmark: benchmark outputs are incomplete", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AdapterError as error:
        print(f"gpu-benchmark: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    except OSError:
        print("gpu-benchmark: local file operation failed", file=sys.stderr)
        raise SystemExit(2) from None
    except (KeyError, TypeError, ValueError):
        print("gpu-benchmark: invalid benchmark measurement", file=sys.stderr)
        raise SystemExit(2) from None
