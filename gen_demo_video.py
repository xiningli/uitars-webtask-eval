#!/usr/bin/env python3
"""Compose the WebTask Arena x UI-TARS demo reel from recorded episodes.

Finds the latest runs-arena/ episode per task, builds title/instruction/verdict
cards with PIL, re-times the annotated frame sequences, and concatenates
everything into runs-arena/demo.mp4 (1280x800, h264, 24 fps).

    python3 gen_demo_video.py            # uses newest run per task
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 800
BG = (18, 20, 26)
FG = (235, 238, 245)
DIM = (150, 158, 170)
GREEN = (52, 199, 89)
RED = (255, 90, 80)
ACCENT = (86, 140, 255)

TASK_ORDER = ["checkout-form", "email-triage", "settings-panel"]

DIAGNOSIS = {
    "checkout-form": "Clean run: precise field clicks, correct typing,\n"
                     "three wizard steps, submit.",
    "email-triage": "State-tracking failure: archives worked, but the model\n"
                    "never perceived rows disappearing — it re-clicked fixed\n"
                    "coordinates while the list reflowed underneath.",
    "settings-panel": "Environment finding: headless chromium renders native\n"
                      "<select> dropdowns outside the page — the timezone\n"
                      "options are invisible to any screenshot agent.",
}


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for candidate in (f"/usr/share/fonts/truetype/dejavu/{name}", name):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _card(lines: list[tuple[str, int, bool, tuple]], out: Path, top: int = None):
    """lines: (text, size, bold, color); vertically centered unless top given."""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    blocks = []
    for text, size, bold, color in lines:
        font = _font(size, bold)
        for sub in text.split("\n"):
            bbox = draw.textbbox((0, 0), sub, font=font)
            blocks.append((sub, font, color, bbox[3] - bbox[1] + int(size * 0.6)))
    total = sum(h for *_ , h in blocks)
    y = top if top is not None else (H - total) // 2
    for sub, font, color, h in blocks:
        bbox = draw.textbbox((0, 0), sub, font=font)
        draw.text(((W - (bbox[2] - bbox[0])) // 2, y), sub, fill=color, font=font)
        y += h
    img.save(out)


def _seg_from_card(card: Path, seconds: float, out: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-loop", "1", "-t", str(seconds), "-i", str(card),
         "-r", "24", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True)


def _seg_from_frames(frames: Path, out: Path, in_fps: int = 8):
    # frames were saved 8x per step -> in_fps=8 plays one agent step per second
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(in_fps),
         "-i", str(frames / "frame_%05d.png"),
         "-r", "24", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-vf", f"scale={W}:{H}", str(out)],
        check=True, capture_output=True)


def main() -> int:
    base = Path(__file__).parent / "runs-arena"
    episodes = {}
    for task in TASK_ORDER:
        runs = sorted(base.glob(f"*_{task}_s*"))
        if not runs:
            sys.exit(f"no recorded run for {task} — run run_arena.py first")
        episodes[task] = runs[-1]

    tmp = Path(tempfile.mkdtemp(prefix="demo-video-"))
    segs: list[Path] = []

    def add_card(name, lines, seconds, top=None):
        card = tmp / f"{name}.png"
        _card(lines, card, top=top)
        seg = tmp / f"{name}.mp4"
        _seg_from_card(card, seconds, seg)
        segs.append(seg)

    add_card("title", [
        ("WebTask Arena", 64, True, FG),
        ("deterministic, server-graded web tasks for GUI agents", 26, False, DIM),
        ("", 20, False, DIM),
        ("agent under test: UI-TARS-1.5-7B (4-bit, local GPU)", 28, False, FG),
        ("screenshots in — pixel-coordinate actions out — no DOM access", 24, False, DIM),
    ], 4.5)

    results = []
    for i, task in enumerate(TASK_ORDER, 1):
        run = episodes[task]
        result = json.loads((run / "result.json").read_text())
        results.append(result)
        add_card(f"intro{i}", [
            (f"Task {i}/3 — {task}", 44, True, ACCENT),
            ("", 16, False, DIM),
            ("\n".join(_wrap(result["instruction"], 58)), 27, False, FG),
            ("", 16, False, DIM),
            ("seed 0 · graded by named subgoals, server-side", 22, False, DIM),
        ], 4.5)
        seg = tmp / f"ep{i}.mp4"
        _seg_from_frames(run / "frames", seg)
        segs.append(seg)

        ok = result["success"]
        failed = [k for k, v in result["subgoals"].items() if not v]
        add_card(f"verdict{i}", [
            ("PASS" if ok else "FAIL", 64, True, GREEN if ok else RED),
            ("", 14, False, DIM),
            (("all subgoals met" if ok else "failed: " + ", ".join(failed)), 26, False, FG),
            ("", 14, False, DIM),
            (DIAGNOSIS[task], 24, False, DIM),
        ], 5.0)

    passed = sum(r["success"] for r in results)
    add_card("outro", [
        ("Scoreboard — seed 0, clean rendering", 40, True, FG),
        ("", 18, False, DIM),
        *[(f"{'PASS' if r['success'] else 'FAIL'}   {r['task']}", 30, True,
           GREEN if r["success"] else RED) for r in results],
        ("", 18, False, DIM),
        (f"{passed}/3 — every failure is reproducible: same seed,\n"
         "same episode, per-subgoal diagnosis", 25, False, DIM),
        ("", 18, False, DIM),
        ("github.com/xiningli/webtask-arena", 26, False, ACCENT),
    ], 7.0)

    concat = tmp / "concat.txt"
    concat.write_text("".join(f"file '{s}'\n" for s in segs))
    out = base / "demo.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
         "-c", "copy", str(out)],
        check=True, capture_output=True)
    print(f"wrote {out}")
    return 0


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


if __name__ == "__main__":
    sys.exit(main())
