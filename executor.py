"""Playwright-over-CDP executor that runs parsed VLA actions and records frames."""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import BrowserContext, Page, sync_playwright

from vla_agent import Action


_KEY_ALIASES = {
    "ctrl": "Control",
    "control": "Control",
    "shift": "Shift",
    "alt": "Alt",
    "meta": "Meta",
    "cmd": "Meta",
    "command": "Meta",
    "enter": "Enter",
    "return": "Enter",
    "esc": "Escape",
    "escape": "Escape",
    "tab": "Tab",
    "space": "Space",
    "backspace": "Backspace",
    "delete": "Delete",
    "del": "Delete",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
}


def _normalize_key(combo: str) -> str:
    parts = re.split(r"[+\s]+", combo.strip())
    return "+".join(_KEY_ALIASES.get(p.lower(), p if len(p) == 1 else p.capitalize()) for p in parts if p)


class Executor:
    def __init__(self, context: BrowserContext, page: Page, frames_dir: Path,
                 viewport: tuple[int, int] = (1280, 800)):
        self.context = context
        self.page = page
        self.viewport = viewport
        self.frames_dir = frames_dir
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.frame_idx = 0
        # Auto-switch to any newly opened tab so the VLA always sees the most
        # recent context (matches the agent's mental model of "the page in front of me").
        self.context.on("page", self._on_new_page)

    def _on_new_page(self, page: Page):
        try:
            page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:
            pass
        try:
            page.set_viewport_size({"width": self.viewport[0], "height": self.viewport[1]})
        except Exception:
            pass
        try:
            page.bring_to_front()
        except Exception:
            pass
        print(f"[executor] switched active tab -> {page.url}")
        self.page = page

    def _refresh_active_page(self):
        """Drop closed pages and prefer the most recently opened still-live page."""
        live = [p for p in self.context.pages if not p.is_closed()]
        if not live:
            return
        if self.page.is_closed() or self.page not in live:
            self.page = live[-1]

    def screenshot(self) -> Image.Image:
        self._refresh_active_page()
        png = self.page.screenshot(type="png", full_page=False)
        return Image.open(io.BytesIO(png)).convert("RGB")

    def save_frame(self, img: Image.Image, caption: str = "", marker: tuple[int, int] | None = None,
                   end_marker: tuple[int, int] | None = None):
        frame = img.copy()
        draw = ImageDraw.Draw(frame, "RGBA")
        if marker:
            x, y = marker
            r = 18
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 30, 30, 230), width=4)
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 30, 30, 230))
        if end_marker:
            x, y = end_marker
            r = 18
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(30, 120, 255, 230), width=4)
        if caption:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 20)
            except OSError:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), caption, font=font)
            pad = 8
            box_w = bbox[2] - bbox[0] + 2 * pad
            box_h = bbox[3] - bbox[1] + 2 * pad
            draw.rectangle([10, 10, 10 + box_w, 10 + box_h], fill=(0, 0, 0, 170))
            draw.text((10 + pad, 10 + pad - bbox[1]), caption, fill=(255, 255, 255, 255), font=font)
        # Hold each frame for several ffmpeg input-frames so the video is watchable.
        for _ in range(8):  # 8 frames @ 4fps -> 2s per step
            frame.save(self.frames_dir / f"frame_{self.frame_idx:05d}.png")
            self.frame_idx += 1

    def execute(self, act: Action) -> bool:
        """Run an action. Returns True if the agent loop should stop."""
        self._refresh_active_page()
        page = self.page
        url_before = page.url
        if act.type == "click" and act.point:
            page.mouse.click(*act.point)
        elif act.type == "left_double" and act.point:
            page.mouse.dblclick(*act.point)
        elif act.type == "right_single" and act.point:
            page.mouse.click(*act.point, button="right")
        elif act.type == "drag" and act.point and act.end_point:
            page.mouse.move(*act.point)
            page.mouse.down()
            page.mouse.move(*act.end_point, steps=20)
            page.mouse.up()
        elif act.type == "type" and act.content is not None:
            content = act.content
            submit = content.endswith("\n")
            if submit:
                content = content[:-1]
            if content:
                page.keyboard.type(content, delay=20)
            if submit:
                page.keyboard.press("Enter")
        elif act.type == "hotkey" and act.key:
            page.keyboard.press(_normalize_key(act.key))
        elif act.type == "scroll":
            dy = {"down": 600, "up": -600, "right": 0, "left": 0}.get(act.direction or "down", 600)
            dx = {"right": 600, "left": -600}.get(act.direction or "", 0)
            if act.point:
                page.mouse.move(*act.point)
            page.mouse.wheel(dx, dy)
        elif act.type == "wait":
            time.sleep(5)
        elif act.type == "finished":
            return True
        elif act.type == "call_user":
            return True
        else:
            print(f"[executor] unknown / malformed action: {act.type} | raw={act.raw!r}")
        # Settle: if the action triggered a navigation, wait for it; otherwise
        # give the page a beat to react before the next screenshot.
        try:
            page.wait_for_load_state("domcontentloaded", timeout=8000)
        except Exception:
            pass
        if page.url != url_before:
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
        page.wait_for_timeout(1500)
        return False


def connect(cdp_url: str, viewport=(1280, 800)):
    """Connect to a chromium running with --remote-debugging-port.
    Returns (pw, browser, context, page)."""
    pw = sync_playwright().start()
    browser = pw.chromium.connect_over_cdp(cdp_url)
    ctx = browser.contexts[0] if browser.contexts else browser.new_context(
        viewport={"width": viewport[0], "height": viewport[1]}
    )
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.set_viewport_size({"width": viewport[0], "height": viewport[1]})
    return pw, browser, ctx, page


def launch_local(viewport=(1280, 800), headed: bool = False):
    """Launch a local chromium (no docker/CDP needed).
    Returns (pw, browser, context, page)."""
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=not headed)
    ctx = browser.new_context(viewport={"width": viewport[0], "height": viewport[1]})
    page = ctx.new_page()
    return pw, browser, ctx, page
