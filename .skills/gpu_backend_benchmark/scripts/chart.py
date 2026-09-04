#!/usr/bin/env python3
"""Render the three benchmark metrics as a dependency-free PNG chart."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any


WIDTH, HEIGHT = 1200, 940
BACKGROUND = (244, 247, 251)
PANEL = (255, 255, 255)
HEADER = (20, 29, 43)
HEADER_MUTED = (185, 198, 216)
INK = (24, 33, 47)
MUTED = (102, 116, 135)
GRID = (229, 235, 243)
BORDER = (216, 224, 234)
COLORS = {"cuda": (105, 174, 45), "vulkan": (221, 78, 82), "metal": (70, 116, 207)}
CONTENT_LEFT, CONTENT_WIDTH = 60, 1080
CARD_GAP = 18
CARD_WIDTH = (CONTENT_WIDTH - 2 * CARD_GAP) // 3
BAR_WIDTH, BAR_GAP = 58, 44


class Canvas:
    def __init__(self, font_path: Path):
        self.pixels = bytearray(BACKGROUND * (WIDTH * HEIGHT))
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

    def centered_text(self, center: int, y: int, value: str, scale: int = 2, color: tuple[int, int, int] = INK) -> None:
        self.text(center - len(value) * 3 * scale, y, value, scale, color)

    def right_text(self, right: int, y: int, value: str, scale: int = 2, color: tuple[int, int, int] = INK) -> None:
        self.text(right - len(value) * 6 * scale, y, value, scale, color)

    def outlined_rectangle(self, x: int, y: int, width: int, height: int) -> None:
        self.rectangle(x, y, width, height, BORDER)
        self.rectangle(x + 1, y + 1, width - 2, height - 2, PANEL)

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
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1000:
        return f"{value / 1000:.1f}K"
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def bar_x(card_left: int, backend_index: int, backend_count: int) -> int:
    """Center fixed backend slots with an explicit non-overlapping gap."""
    total_width = backend_count * BAR_WIDTH + (backend_count - 1) * BAR_GAP
    return card_left + (CARD_WIDTH - total_width) // 2 + backend_index * (BAR_WIDTH + BAR_GAP)


def render_section(canvas: Canvas, top: int, title: str, hint: str, metric: str, rows: list[dict[str, Any]], backends: list[str]) -> None:
    canvas.text(CONTENT_LEFT, top, title, 3)
    canvas.right_text(CONTENT_LEFT + CONTENT_WIDTH, top + 4, hint, 2, MUTED)
    panel_top, panel_height = top + 35, 220
    baseline, chart_height = panel_top + 182, 145
    values = [row[metric] for row in rows]
    ceiling = max(values, default=1.0) * 1.15
    indexed = {(row["workload"], row["backend"]): row for row in rows}
    for group, workload in enumerate(("short", "medium", "long")):
        card_left = CONTENT_LEFT + group * (CARD_WIDTH + CARD_GAP)
        center = card_left + CARD_WIDTH // 2
        canvas.outlined_rectangle(card_left, panel_top, CARD_WIDTH, panel_height)
        canvas.centered_text(center, panel_top + 12, workload, 2)
        for step in (1, 2, 3):
            y = baseline - chart_height * step // 4
            canvas.rectangle(card_left + 18, y, CARD_WIDTH - 36, 1, GRID)
        canvas.rectangle(card_left + 18, baseline, CARD_WIDTH - 36, 2, BORDER)
        for backend_index, backend in enumerate(backends):
            x = bar_x(card_left, backend_index, len(backends))
            row = indexed.get((workload, backend))
            if row is None:
                canvas.centered_text(x + BAR_WIDTH // 2, baseline - 22, "N/A", 2, MUTED)
            else:
                height = max(4, round(row[metric] / ceiling * chart_height))
                canvas.rectangle(x, baseline - height, BAR_WIDTH, height, COLORS.get(backend, INK))
                label_y = max(panel_top + 42, baseline - height - 19)
                canvas.centered_text(x + BAR_WIDTH // 2, label_y, format_value(row[metric]), 2)
            canvas.centered_text(x + BAR_WIDTH // 2, baseline + 10, backend, 2, MUTED)


def write_chart(data: dict[str, Any], output: Path) -> None:
    font = Path(__file__).resolve().parent.parent / "assets" / "font5x7.json"
    canvas = Canvas(font)
    canvas.rectangle(0, 0, WIDTH, 94, HEADER)
    canvas.text(42, 22, "GPU BACKEND BENCHMARK", 4, PANEL)
    canvas.text(44, 64, f"{data['machine']['gpu']} / MEDIAN OF {data['configuration']['repetitions']} RUNS", 2, HEADER_MUTED)
    backends = sorted({row["backend"] for row in data["summaries"]})
    x = 775
    for backend in backends:
        canvas.rectangle(x, 34, 22, 14, COLORS.get(backend, INK))
        canvas.text(x + 32, 33, backend, 2, PANEL)
        x += 145
    rows = data["summaries"]
    render_section(canvas, 110, "PREFILL THROUGHPUT", "HIGHER / TOKENS PER SECOND", "prefill_tps", rows, backends)
    render_section(canvas, 380, "DECODE THROUGHPUT", "HIGHER / TOKENS PER SECOND", "decode_tps", rows, backends)
    render_section(canvas, 650, "TIME TO FIRST TOKEN", "LOWER / MILLISECONDS", "ttft_ms", rows, backends)
    canvas.save(output)
