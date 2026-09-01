#!/usr/bin/env python3
"""Render the three benchmark metrics as a dependency-free PNG chart."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any


WIDTH, HEIGHT = 1200, 940
WHITE = (255, 255, 255)
INK = (28, 32, 40)
GRID = (214, 220, 229)
COLORS = {"cuda": (76, 171, 68), "vulkan": (218, 74, 74), "metal": (67, 120, 210)}


class Canvas:
    def __init__(self, font_path: Path):
        self.pixels = bytearray(WHITE * (WIDTH * HEIGHT))
        raw = json.loads(font_path.read_text(encoding="utf-8"))
        self.font = {key: value.split("/") for key, value in raw.items()}

    def rectangle(self, x: int, y: int, width: int, height: int, color: tuple[int, int, int]) -> None:
        left, right = max(0, x), min(WIDTH, x + width)
        top, bottom = max(0, y), min(HEIGHT, y + height)
        for row in range(top, bottom):
            offset = (row * WIDTH + left) * 3
            self.pixels[offset : offset + (right - left) * 3] = bytes(color) * (right - left)

    def text(self, x: int, y: int, value: str, scale: int = 2, color: tuple[int, int, int] = INK) -> None:
        cursor = x
        for character in value.upper():
            glyph = self.font.get(character, self.font["?"])
            for row, bits in enumerate(glyph):
                for column, bit in enumerate(bits):
                    if bit == "1":
                        self.rectangle(cursor + column * scale, y + row * scale, scale, scale, color)
            cursor += 6 * scale

    def centered_text(self, center: int, y: int, value: str, scale: int = 2) -> None:
        self.text(center - len(value) * 3 * scale, y, value, scale)

    def save(self, path: Path) -> None:
        rows = b"".join(b"\x00" + self.pixels[y * WIDTH * 3 : (y + 1) * WIDTH * 3] for y in range(HEIGHT))
        signature = b"\x89PNG\r\n\x1a\n"
        def chunk(kind: bytes, data: bytes) -> bytes:
            body = kind + data
            return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        png = signature + chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b"")
        path.write_bytes(png)


def format_value(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f}K"
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def render_section(canvas: Canvas, top: int, title: str, unit: str, metric: str, rows: list[dict[str, Any]], backends: list[str]) -> None:
    canvas.text(55, top, f"{title} ({unit})", 3)
    left, baseline, chart_height = 90, top + 230, 155
    canvas.rectangle(left, baseline, 1060, 2, INK)
    values = [row[metric] for row in rows]
    ceiling = max(values, default=1.0) * 1.18
    for step in range(1, 4):
        y = baseline - chart_height * step // 4
        canvas.rectangle(left, y, 1060, 1, GRID)
    groups = {name: index for index, name in enumerate(("short", "medium", "long"))}
    group_width = 330
    bar_width = 72 if len(backends) == 1 else 58
    for row in rows:
        group = groups.get(row["workload"])
        if group is None or row["backend"] not in backends:
            continue
        backend_index = backends.index(row["backend"])
        total_width = len(backends) * (bar_width + 14) - 14
        x = left + group * group_width + 165 - total_width // 2 + backend_index * (bar_width + 14)
        height = max(1, round(row[metric] / ceiling * chart_height))
        canvas.rectangle(x, baseline - height, bar_width, height, COLORS.get(row["backend"], INK))
        canvas.centered_text(x + bar_width // 2, baseline - height - 20, format_value(row[metric]), 2)
    for group, label in enumerate(("SHORT", "MEDIUM", "LONG")):
        canvas.centered_text(left + group * group_width + 165, baseline + 15, label, 2)


def write_chart(data: dict[str, Any], output: Path) -> None:
    font = Path(__file__).resolve().parent.parent / "assets" / "font5x7.json"
    canvas = Canvas(font)
    canvas.text(42, 28, "GPU BACKEND BENCHMARK", 4)
    canvas.text(44, 70, f"{data['machine']['gpu']} - MEDIANS OF {data['configuration']['repetitions']} RUNS", 2)
    backends = sorted({row["backend"] for row in data["summaries"]})
    x = 760
    for backend in backends:
        canvas.rectangle(x, 35, 24, 14, COLORS.get(backend, INK))
        canvas.text(x + 34, 34, backend, 2)
        x += 135
    rows = data["summaries"]
    render_section(canvas, 115, "PREFILL THROUGHPUT", "TOKENS/S", "prefill_tps", rows, backends)
    render_section(canvas, 380, "DECODE THROUGHPUT", "TOKENS/S", "decode_tps", rows, backends)
    render_section(canvas, 645, "TIME TO FIRST TOKEN", "MS - LOWER IS BETTER", "ttft_ms", rows, backends)
    canvas.save(output)
