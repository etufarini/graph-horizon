#!/usr/bin/env python3
"""Bounded process interface for a GPU benchmark runtime adapter."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


class AdapterError(Exception):
    """A safe, actionable adapter failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


ERROR_ACTIONS = {
    "unsupported_backend": "build or install the runtime with this backend enabled",
    "model_load": "verify that the model format is supported and the file is intact",
    "out_of_memory": "use a smaller model or free GPU memory, then rerun",
    "context_too_large": "set a smaller context or report the model context limit in probe output",
    "runtime_failure": "run the adapter directly for backend-specific diagnostics",
}


def resolve_adapter(command: str) -> str:
    candidate = Path(command).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        if not candidate.is_file():
            raise AdapterError("missing_runtime", "runtime adapter is missing")
        if os.name != "nt" and not os.access(candidate, os.X_OK):
            raise AdapterError("missing_runtime", "runtime adapter is not executable")
        return str(candidate.resolve())
    resolved = shutil.which(command)
    if not resolved:
        raise AdapterError("missing_runtime", "runtime adapter was not found on PATH")
    return resolved


def invoke(
    adapter: str,
    adapter_args: list[str],
    mode: str,
    arguments: list[str],
    timeout: int,
) -> dict[str, Any]:
    command = [adapter, *adapter_args, mode, *arguments]
    try:
        process = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise AdapterError("runtime_failure", "runtime adapter timed out") from error
    except OSError as error:
        raise AdapterError("runtime_failure", "runtime adapter could not start") from error

    if len(process.stdout.encode("utf-8")) > 65_536:
        raise AdapterError("malformed_output", "runtime adapter output exceeded 64 KiB")
    try:
        payload = json.loads(process.stdout)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise AdapterError("malformed_output", "runtime adapter did not return one JSON object") from error
    if not isinstance(payload, dict):
        raise AdapterError("malformed_output", "runtime adapter output must be a JSON object")
    if process.returncode != 0:
        failure = payload.get("error", {})
        code = failure.get("code", "runtime_failure") if isinstance(failure, dict) else "runtime_failure"
        action = ERROR_ACTIONS.get(code, ERROR_ACTIONS["runtime_failure"])
        raise AdapterError(code, action)
    if "error" in payload:
        raise AdapterError("malformed_output", "successful adapter output contains an error")
    return payload


def require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise AdapterError("malformed_output", f"adapter field {field} must be non-empty text")
    return value.strip()


def require_number(value: Any, field: str, *, integer: bool = False) -> float | int:
    valid_type = isinstance(value, int) if integer else isinstance(value, (int, float))
    if isinstance(value, bool) or not valid_type or value <= 0 or not math.isfinite(float(value)):
        raise AdapterError("malformed_output", f"adapter field {field} must be positive")
    return int(value) if integer else float(value)


def validate_probe(payload: dict[str, Any], backend: str) -> dict[str, Any]:
    if payload.get("available") is not True or payload.get("backend") != backend:
        raise AdapterError("unsupported_backend", ERROR_ACTIONS["unsupported_backend"])
    runtime = payload.get("runtime")
    device = payload.get("device")
    model = payload.get("model")
    if not all(isinstance(item, dict) for item in (runtime, device, model)):
        raise AdapterError("malformed_output", "probe must include runtime, device, and model objects")
    require_text(runtime.get("name"), "runtime.name")
    require_text(runtime.get("version"), "runtime.version")
    require_text(device.get("id"), "device.id")
    require_text(device.get("name"), "device.name")
    require_text(model.get("identifier"), "model.identifier")
    require_text(model.get("quantization"), "model.quantization")
    require_number(model.get("context_limit"), "model.context_limit", integer=True)
    return payload


def validate_count(payload: dict[str, Any]) -> int:
    return int(require_number(payload.get("prompt_tokens"), "prompt_tokens", integer=True))


def validate_run(payload: dict[str, Any], backend: str) -> dict[str, Any]:
    required_text = ("backend", "device_id", "gpu_name", "runtime_name", "runtime_version", "runtime_config", "model_identifier", "quantization")
    for field in required_text:
        require_text(payload.get(field), field)
    if payload["backend"] != backend:
        raise AdapterError("malformed_output", "adapter returned the wrong backend")
    for field in ("prompt_tokens", "completion_tokens", "context_size", "batch_size"):
        require_number(payload.get(field), field, integer=True)
    seed, temperature = payload.get("seed"), payload.get("temperature")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise AdapterError("malformed_output", "adapter field seed must be a non-negative integer")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature < 0:
        raise AdapterError("malformed_output", "adapter field temperature must be non-negative")
    for field in ("prefill_seconds", "decode_seconds", "ttft_ms"):
        require_number(payload.get(field), field)
    return payload
