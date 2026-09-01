#!/usr/bin/env python3
"""Detect benchmark-relevant hardware and supported GPU backends."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from typing import Any


def command(argv: list[str], timeout: int = 10) -> str:
    try:
        result = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def cpu_name(system: str) -> str:
    if system == "Darwin":
        return command(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    if system == "Windows":
        output = command([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
        ])
        return output or platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as source:
            for line in source:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def nvidia_devices() -> list[dict[str, Any]]:
    output = command([
        "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    devices = []
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 5 or not fields[0].isdigit():
            continue
        memory = f"{round(int(fields[3]) / 1024)} GB VRAM" if fields[3].isdigit() else "unknown"
        devices.append({
            "index": int(fields[0]), "name": fields[1], "id": fields[2],
            "memory": memory, "driver": fields[4],
        })
    return devices


def vulkan_devices() -> tuple[str | None, list[dict[str, str]]]:
    output = command(["vulkaninfo", "--summary"])
    version_match = re.search(r"Vulkan Instance Version:\s*([^\s]+)", output)
    devices = []
    for block in re.split(r"\nGPU\d+:\n", output)[1:]:
        values = {}
        for key in ("deviceType", "deviceName", "deviceUUID", "driverInfo"):
            match = re.search(rf"^\s*{key}\s*=\s*(.+)$", block, re.MULTILINE)
            if match:
                values[key] = match.group(1).strip()
        if values.get("deviceType") != "PHYSICAL_DEVICE_TYPE_CPU" and values.get("deviceName"):
            devices.append({
                "name": values["deviceName"], "id": values.get("deviceUUID", "unknown"),
                "driver": values.get("driverInfo", "unknown"),
            })
    return (version_match.group(1) if version_match else None, devices)


def normalized_id(value: str) -> str:
    return re.sub(r"[^0-9a-f]", "", value.lower().removeprefix("gpu-"))


def cuda_version() -> str | None:
    output = command(["nvidia-smi"])
    match = re.search(r"CUDA Version:\s*([^\s|]+)", output)
    if match:
        return match.group(1)
    output = command(["nvcc", "--version"])
    match = re.search(r"release\s+([^,]+)", output)
    return match.group(1) if match else None


def linux_or_windows(system: str) -> dict[str, Any]:
    nvidia = nvidia_devices()
    vulkan_version, vulkan = vulkan_devices()
    selected = nvidia[0] if len(nvidia) == 1 else None
    matching_vulkan = [] if not selected else [
        item for item in vulkan if normalized_id(item["id"]) == normalized_id(selected["id"])
    ]
    backends = {
        "cuda": {
            "available": selected is not None,
            "reason": None if selected else ("multiple NVIDIA GPUs detected; expose the same single GPU to detection and runtime" if nvidia else "nvidia-smi found no NVIDIA GPU"),
        },
        "vulkan": {
            "available": len(matching_vulkan) == 1,
            "reason": None if len(matching_vulkan) == 1 else "Vulkan did not expose the same single NVIDIA GPU",
        },
    }
    return {
        "gpu": selected["name"] if selected else (nvidia[0]["name"] if nvidia else "unknown"),
        "gpu_id": selected["id"] if selected else "unknown",
        "memory": selected["memory"] if selected else "unknown",
        "gpu_driver": selected["driver"] if selected else "unknown",
        "cuda_version": cuda_version(),
        "vulkan_version": vulkan_version,
        "backends": backends,
    }


def mac_hardware() -> dict[str, Any]:
    raw = command(["system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"], 30)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    hardware = (data.get("SPHardwareDataType") or [{}])[0]
    displays = data.get("SPDisplaysDataType") or []
    chip = hardware.get("chip_type") or hardware.get("cpu_type") or "Apple SoC"
    model = hardware.get("machine_name") or hardware.get("machine_model") or "Mac"
    chip_suffix = chip.removeprefix("Apple ").strip()
    if chip_suffix and chip_suffix.lower() not in model.lower():
        model = f"{model} {chip_suffix}"
    memory = hardware.get("physical_memory") or "unknown"
    metal = any("spdisplays_metal" in item or "Metal" in json.dumps(item) for item in displays)
    return {
        "model": model, "gpu": chip, "gpu_id": hardware.get("platform_UUID", model),
        "memory": f"{memory} unified memory" if memory != "unknown" else memory,
        "gpu_driver": "macOS", "cuda_version": None, "vulkan_version": None,
        "metal_runtime": platform.mac_ver()[0] or "unknown",
        "backends": {"metal": {"available": metal, "reason": None if metal else "Metal support was not reported"}},
    }


def detect() -> dict[str, Any]:
    system = platform.system()
    specific = mac_hardware() if system == "Darwin" else linux_or_windows(system)
    machine = specific.pop("model", specific["gpu"])
    os_name = f"macOS {platform.mac_ver()[0]}" if system == "Darwin" and platform.mac_ver()[0] else f"{system} {platform.release()}"
    return {
        "machine": machine, "os": os_name,
        "cpu": cpu_name(system), **specific,
    }


if __name__ == "__main__":
    print(json.dumps(detect(), indent=2, sort_keys=True))
