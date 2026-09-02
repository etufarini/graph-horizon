#!/usr/bin/env python3
"""Model-free validation for the GPU backend benchmark skill."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark
from chart import BAR_GAP, BAR_WIDTH, CARD_WIDTH, bar_x
from hardware import detect
from report import markdown
from results import comparison, summaries


ADAPTER = r'''import json, sys
mode, args = sys.argv[1], sys.argv[2:]
def value(flag):
    return args[args.index(flag) + 1]
backend = value("--backend")
if mode == "probe":
    print(json.dumps({"available": True, "backend": backend,
        "runtime": {"name": "Fixture Runtime", "version": "1.0"},
        "device": {"id": "GPU-11111111-2222-3333-4444-555555555555" if backend != "metal" else "apple-1", "name": "Fixture GPU"},
        "model": {"identifier": "fixture-model", "quantization": "Q4_K_M", "context_limit": 8192}}))
elif mode == "count":
    prompt = open(value("--prompt-file"), encoding="utf-8").read()
    print(json.dumps({"prompt_tokens": prompt.count("benchmark")}))
elif mode == "run":
    prompt = open(value("--prompt-file"), encoding="utf-8").read()
    prompt_tokens, generated = prompt.count("benchmark"), int(value("--max-tokens"))
    factor = 2 if backend == "cuda" else 1
    print(json.dumps({"backend": backend,
        "device_id": "GPU-11111111-2222-3333-4444-555555555555" if backend != "metal" else "apple-1",
        "gpu_name": "Fixture GPU", "runtime_name": "Fixture Runtime", "runtime_version": "1.0",
        "runtime_config": "greedy; fixed device; no early EOS",
        "model_identifier": "fixture-model", "quantization": "Q4_K_M",
        "prompt_tokens": prompt_tokens, "completion_tokens": generated,
        "context_size": int(value("--context")), "batch_size": int(value("--batch-size")),
        "seed": int(value("--seed")), "temperature": float(value("--temperature")),
        "prefill_seconds": prompt_tokens / (100 * factor), "decode_seconds": generated / (40 * factor),
        "ttft_ms": 50 if backend == "cuda" else 75}))
else:
    print(json.dumps({"error": {"code": "runtime_failure", "message": "bad mode"}})); sys.exit(1)
'''


def machine(backends: dict) -> dict:
    return {
        "machine": "Fixture Workstation", "os": "Linux fixture", "cpu": "Fixture CPU",
        "gpu": "Fixture GPU", "gpu_id": "GPU-11111111-2222-3333-4444-555555555555",
        "memory": "12 GB VRAM", "gpu_driver": "1.0", "cuda_version": "13.0",
        "vulkan_version": "1.4", "backends": backends,
    }


class SkillValidation(unittest.TestCase):
    def run_benchmark(self, detected: dict) -> tuple[int, Path, tempfile.TemporaryDirectory]:
        state = tempfile.TemporaryDirectory(prefix="gpu-skill-validation-")
        root = Path(state.name)
        adapter, model, output = root / "adapter.py", root / "model.gguf", root / "output"
        adapter.write_text(ADAPTER, encoding="utf-8")
        model.write_bytes(b"representative model fixture")
        argv = ["benchmark.py", "--adapter", sys.executable, "--adapter-arg", str(adapter),
                "--model", str(model), "--model-name", "Fixture Model", "--quantization", "Q4_K_M",
                "--output-dir", str(output), "--timeout", "10"]
        with patch.object(sys, "argv", argv), patch.object(benchmark, "detect", return_value=detected):
            code = benchmark.main()
        return code, output, state

    def test_cuda_vulkan_end_to_end(self) -> None:
        available = {name: {"available": True, "reason": None} for name in ("cuda", "vulkan")}
        code, output, state = self.run_benchmark(machine(available))
        self.addCleanup(state.cleanup)
        self.assertEqual(code, 0)
        data = json.loads((output / "benchmark-results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["runs"]), 18)
        self.assertTrue(data["comparison"]["valid"])
        self.assertEqual({row["samples"] for row in data["summaries"]}, {3})
        report = (output / "benchmark-results.md").read_text(encoding="utf-8")
        self.assertIn("+100.0%", report)
        self.assertIn("+33.3%", report)
        png = (output / "benchmark-results.png").read_bytes()
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (1200, 940))

    def test_missing_vulkan_is_reported(self) -> None:
        detected = machine({"cuda": {"available": True, "reason": None},
                            "vulkan": {"available": False, "reason": "Vulkan unavailable"}})
        code, output, state = self.run_benchmark(detected)
        self.addCleanup(state.cleanup)
        self.assertEqual(code, 0)
        report = (output / "benchmark-results.md").read_text(encoding="utf-8")
        self.assertIn("Vulkan: unavailable", report)
        self.assertIn("invalid or incomplete", report)

    def test_metal_report_uses_mac_identity(self) -> None:
        detected = machine({"metal": {"available": True, "reason": None}})
        detected.update({"machine": "MacBook Pro M4 Pro", "os": "Darwin 25.0", "gpu": "Apple M4 Pro",
                         "gpu_id": "apple-1", "memory": "48 GB unified memory", "metal_runtime": "26.0"})
        code, output, state = self.run_benchmark(detected)
        self.addCleanup(state.cleanup)
        self.assertEqual(code, 0)
        report = (output / "benchmark-results.md").read_text(encoding="utf-8")
        self.assertIn("MacBook Pro M4 Pro", report)
        self.assertIn("48 GB unified memory", report)
        self.assertIn("not applicable on macOS", report)

    def test_median_and_mismatch_checks(self) -> None:
        rows = [{"workload": "short", "backend": "cuda", "prefill_tps": number,
                 "decode_tps": number, "ttft_ms": number} for number in (1.0, 100.0, 3.0)]
        self.assertEqual(summaries(rows)[0]["prefill_tps"], 3.0)
        base = {"workload": "short", "model_sha256": "a", "quantization": "q", "prompt_sha256": "p",
                "prompt_tokens": 1, "completion_tokens": 1, "context_size": 8, "batch_size": 1,
                "seed": 1, "temperature": 0, "runtime_name": "r", "runtime_version": "1", "runtime_config": "fixed",
                "model_identifier": "m", "device_id": "gpu-1", "gpu_name": "gpu"}
        runs = [{**base, "backend": backend} for backend in ("cuda", "vulkan") for _ in range(3)]
        runs += [{**row, "workload": workload} for workload in ("medium", "long") for row in runs[:6]]
        runs[-1]["context_size"] = 9
        self.assertFalse(comparison(runs, 3, ["cuda", "vulkan"])["valid"])

    def test_context_reduction_is_explicit(self) -> None:
        plans = benchmark.plan_workloads(1000)
        self.assertEqual((plans[0]["desired_prompt_tokens"], plans[0]["generated_tokens"]), (128, 128))
        self.assertEqual((plans[1]["desired_prompt_tokens"], plans[1]["generated_tokens"]), (750, 250))
        self.assertIn("reduced to fit", plans[2]["adjustment"])

    def test_chart_backend_columns_are_separate_and_inside_the_card(self) -> None:
        first, second = (bar_x(60, index, 2) for index in range(2))
        self.assertEqual(second - first - BAR_WIDTH, BAR_GAP)
        self.assertGreaterEqual(first, 60)
        self.assertLessEqual(second + BAR_WIDTH, 60 + CARD_WIDTH)

    def test_skill_links_and_live_detection(self) -> None:
        root = Path(__file__).resolve().parent.parent
        self.assertTrue((root / "references" / "runtime-adapter.md").is_file())
        self.assertTrue((root / "assets" / "font5x7.json").is_file())
        found = detect()
        self.assertTrue({"machine", "os", "cpu", "gpu", "memory", "backends"} <= found.keys())
        json.dumps(found)


if __name__ == "__main__":
    unittest.main(verbosity=2)
