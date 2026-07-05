#!/usr/bin/env python3
"""Run a VLA-driven browser session inside a Docker chromium container.

Usage:
    python run.py <dockerfile> <folder>

The folder must contain an `index.md` whose contents are the natural-language
task for the agent (e.g. "Go to bing.com, click Image Creator, ..."). The
Dockerfile must produce a chromium image that exposes CDP on port 9222.

Output: <folder>/runs/<timestamp>/{frames/, run.mp4, transcript.txt}
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from PIL import Image  # noqa: E402

import executor  # noqa: E402  (after env setup)
from vla_agent import UITarsAgent  # noqa: E402

# Markdown image syntax: ![alt](path "title")
_MD_IMG = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

VIEWPORT = (1280, 800)
CDP_PORT = 9223  # proxied by socat in the container -> chromium's localhost:9222
WEB_PORT = 3000
CONTAINER_NAME = "vla-chromium"
IMAGE_TAG = "vla-chromium:local"
MAX_STEPS = 25


def _wait_for_cdp(url: str, timeout: float = 90.0) -> None:
    """Poll /json/version until reachable."""
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/json/version", timeout=2) as r:
                json.loads(r.read())
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5)
    raise RuntimeError(f"CDP at {url} not ready within {timeout}s: {last_err}")


def _docker(*args, check=True, capture=False):
    return subprocess.run(["docker", *args], check=check,
                          capture_output=capture, text=True)


def _build_image(dockerfile: Path):
    print(f"[run] building image from {dockerfile}")
    _docker("build", "-t", IMAGE_TAG, "-f", str(dockerfile), str(dockerfile.parent))


def _start_container():
    _docker("rm", "-f", CONTAINER_NAME, check=False, capture=True)
    print(f"[run] starting container {CONTAINER_NAME}")
    _docker(
        "run", "-d", "--rm",
        "--name", CONTAINER_NAME,
        "--shm-size=2g",
        "-p", f"{WEB_PORT}:3000",
        "-p", f"{CDP_PORT}:9223",
        IMAGE_TAG,
    )


def _stop_container():
    _docker("rm", "-f", CONTAINER_NAME, check=False, capture=True)


def _stitch_video(frames_dir: Path, out_path: Path, fps: int = 4):
    if not any(frames_dir.iterdir()):
        print("[run] no frames recorded, skipping video")
        return
    if not shutil.which("ffmpeg"):
        print("[run] ffmpeg not found; frames are saved but no mp4 produced")
        return
    print(f"[run] encoding {out_path}")
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps),
         "-i", str(frames_dir / "frame_%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
         str(out_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dockerfile", type=Path)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--keep-container", action="store_true",
                    help="Don't kill the container after the run (for inspection).")
    args = ap.parse_args()

    dockerfile = args.dockerfile.resolve()
    folder = args.folder.resolve()
    index_md = folder / "index.md"
    if not dockerfile.is_file():
        sys.exit(f"Dockerfile not found: {dockerfile}")
    if not index_md.is_file():
        sys.exit(f"index.md not found in: {folder}")

    raw_instruction = index_md.read_text().strip()

    # Resolve markdown image references (relative to the folder) and strip them
    # from the prompt text.
    reference_images: list[tuple[str, Image.Image]] = []
    for alt, path in _MD_IMG.findall(raw_instruction):
        img_path = (folder / path).resolve()
        if not img_path.is_file():
            print(f"[run] warning: referenced image not found: {img_path}")
            continue
        reference_images.append((alt or img_path.name, Image.open(img_path).convert("RGB")))
    instruction = _MD_IMG.sub(lambda m: m.group(1) or m.group(2), raw_instruction).strip()

    print(f"[run] task:\n{instruction}\n")
    if reference_images:
        print(f"[run] reference images: {[n for n, _ in reference_images]}")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = folder / "runs" / stamp
    frames_dir = run_dir / "frames"
    run_dir.mkdir(parents=True, exist_ok=True)
    transcript = run_dir / "transcript.txt"

    # Build/start docker first so environment failures surface before the
    # (slow) model load.
    _build_image(dockerfile)
    _start_container()

    print("[run] loading VLA model (UI-TARS-1.5-7B)…")
    agent = UITarsAgent()
    pw = browser = None
    try:
        cdp = f"http://localhost:{CDP_PORT}"
        _wait_for_cdp(cdp)
        print(f"[run] CDP ready at {cdp}")

        pw, browser, ctx, page = executor.connect(cdp, viewport=VIEWPORT)

        # The VLA only sees page content (no browser chrome / URL bar), so we
        # bootstrap by navigating to the first URL mentioned in the task.
        m = re.search(r"https?://[^\s)\]\"'>]+", instruction)
        start_url = m.group(0).rstrip(".,;:!?'\"") if m else "about:blank"
        print(f"[run] pre-navigating to {start_url}")
        page.goto(start_url, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(1500)

        ex = executor.Executor(ctx, page, frames_dir, viewport=VIEWPORT)

        with transcript.open("w") as log:
            log.write(f"TASK: {instruction}\n\n")
            recent: list[str] = []  # for loop detection
            for step in range(args.max_steps):
                shot = ex.screenshot()
                act = agent.step(shot, instruction,
                                 reference_images=reference_images if step == 0 else None)
                line = (f"[step {step:02d}] {act.type} "
                        f"point={act.point} end={act.end_point} "
                        f"key={act.key!r} dir={act.direction!r} "
                        f"content={act.content!r}")
                print(line)
                if act.thought:
                    print(f"           thought: {act.thought}")
                log.write(line + "\n")
                if act.thought:
                    log.write(f"  thought: {act.thought}\n")
                log.write(f"  raw: {act.raw}\n\n")
                log.flush()

                ex.save_frame(shot, caption=f"#{step}: {act.type}({act.point or ''})",
                              marker=act.point, end_marker=act.end_point)

                # Stop if the agent has produced the same action 4 times in a row
                # (UI-TARS sometimes loops when the page state isn't changing).
                recent.append(act.raw)
                if len(recent) >= 4 and len(set(recent[-4:])) == 1:
                    print(f"[run] aborting: agent looped on {act.raw!r}")
                    break

                if ex.execute(act):
                    print(f"[run] agent signaled stop at step {step}")
                    break
            else:
                print(f"[run] hit max_steps={args.max_steps}")

            # one final frame after the last action
            ex.save_frame(ex.screenshot(), caption="final state")

    finally:
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        if pw is not None:
            try:
                pw.stop()
            except Exception:
                pass
        if not args.keep_container:
            _stop_container()

    _stitch_video(frames_dir, run_dir / "run.mp4")
    print(f"[run] done -> {run_dir}")


if __name__ == "__main__":
    main()
