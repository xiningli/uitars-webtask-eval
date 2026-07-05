"""UI-TARS-1.5 VLA agent: screenshot + task -> next action."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "ByteDance-Seed/UI-TARS-1.5-7B"

SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
click(start_box='(x,y)')
left_double(start_box='(x,y)')
right_single(start_box='(x,y)')
drag(start_box='(x1,y1)', end_box='(x2,y2)')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='(x,y)', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

## Note
- Coordinates (x, y) are integer pixel positions in the screenshot you are shown.
- Use English in `Thought` part.
- Summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}"""


@dataclass
class Action:
    type: str  # click, left_double, right_single, drag, hotkey, type, scroll, wait, finished, call_user
    point: Optional[tuple[int, int]] = None  # in original-image pixel space
    end_point: Optional[tuple[int, int]] = None
    content: Optional[str] = None
    key: Optional[str] = None
    direction: Optional[str] = None
    raw: str = ""
    thought: str = ""


# Match the various coordinate spellings UI-TARS may emit:
#   start_box='(186,242)'
#   start_box='<|box_start|>(186,242)<|box_end|>'
#   point='<point>186 242</point>'
#   start_box='<bbox>180 235 200 250</bbox>'   (4-coord bbox -> we use the center)
_PAREN_PT = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
_POINT_TAG = re.compile(r"<point>\s*(-?\d+)\s+(-?\d+)\s*</point>")
_BBOX_TAG = re.compile(r"<bbox>\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*</bbox>")


def _extract_points(s: str) -> list[tuple[int, int]]:
    """Pull out (x, y) pairs in *normalized 0..1000 space* from any supported spelling."""
    pts: list[tuple[int, int]] = []
    for x1, y1, x2, y2 in _BBOX_TAG.findall(s):
        pts.append(((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2))
    for x, y in _POINT_TAG.findall(s):
        pts.append((int(x), int(y)))
    if not pts:  # fall back to bare parenthesized pairs only if no tagged form was found
        pts.extend((int(x), int(y)) for x, y in _PAREN_PT.findall(s))
    return pts


def _parse_action(text: str, scale_x: float, scale_y: float) -> Action:
    """Parse model output. UI-TARS-1.5 emits pixel coords in the smart_resized image
    space; ``scale_x``/``scale_y`` map them back to the original screenshot."""
    thought = ""
    action_str = text.strip()
    if "Action:" in action_str:
        before, _, after = action_str.partition("Action:")
        thought = before.replace("Thought:", "").strip()
        action_str = after.strip()
    action_str = action_str.strip("` \n")

    name = action_str.split("(", 1)[0].strip()
    pts = [(int(round(x * scale_x)), int(round(y * scale_y)))
           for x, y in _extract_points(action_str)]

    def _unesc(s: str) -> str:
        # Only decode the common backslash-escapes; running unicode_escape over
        # the whole string would mangle multi-byte UTF-8 (e.g. CJK characters).
        return (s.replace(r"\n", "\n").replace(r"\t", "\t")
                 .replace(r"\'", "'").replace(r'\"', '"').replace(r"\\", "\\"))

    def _kw(arg: str) -> Optional[str]:
        m = re.search(rf"{arg}\s*=\s*'((?:\\'|[^'])*)'", action_str)
        if m:
            return _unesc(m.group(1))
        m = re.search(rf'{arg}\s*=\s*"((?:\\"|[^"])*)"', action_str)
        if m:
            return _unesc(m.group(1))
        return None

    act = Action(type=name, raw=action_str, thought=thought)
    if name in {"click", "left_double", "right_single"} and pts:
        act.point = pts[0]
    elif name == "drag" and len(pts) >= 2:
        act.point = pts[0]
        act.end_point = pts[1]
    elif name == "scroll":
        if pts:
            act.point = pts[0]
        act.direction = _kw("direction") or "down"
    elif name == "type":
        act.content = _kw("content") or ""
    elif name == "hotkey":
        act.key = _kw("key") or ""
    elif name == "finished":
        act.content = _kw("content") or ""
    return act


# Fallback smart_resize parameters; the real values are read from the loaded
# processor config at runtime (UI-TARS-1.5-7B ships max_pixels=12845056, NOT
# the 1M-pixel Qwen default — a mismatch here skews every coordinate).
_IMAGE_FACTOR = 28
_MIN_PIXELS = 56 * 56
_MAX_PIXELS = 14 * 14 * 4 * 1280


def _smart_resize(h: int, w: int, factor: int = _IMAGE_FACTOR,
                  min_pixels: int = _MIN_PIXELS, max_pixels: int = _MAX_PIXELS) -> tuple[int, int]:
    """Replicate Qwen2.5-VL's smart_resize so we know the model's perceived dimensions."""
    if max(h, w) / min(h, w) > 200:
        raise ValueError("aspect ratio too extreme for smart_resize")
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


class UITarsAgent:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16, load_in_4bit: bool = True,
                 max_history_screenshots: int = 5):
        self.processor = AutoProcessor.from_pretrained(model_id)
        # The processor decides the resized image the model actually sees; use
        # its own limits so our coordinate un-mapping matches it exactly.
        ip = self.processor.image_processor
        self.min_pixels = getattr(ip, "min_pixels", None) or _MIN_PIXELS
        self.max_pixels = getattr(ip, "max_pixels", None) or _MAX_PIXELS
        self.factor = getattr(ip, "patch_size", 14) * getattr(ip, "merge_size", 2)
        self.max_history_screenshots = max_history_screenshots
        kwargs: dict = {"low_cpu_mem_usage": True, "device_map": device}
        if load_in_4bit:
            # 4-bit NF4 keeps the 7B weights under ~5 GB so the vision tower + KV cache fit on 16 GB.
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = dtype
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.history: list[dict] = []  # list of {role, content}

    def reset(self):
        self.history = []

    @torch.inference_mode()
    def step(self, screenshot: Image.Image, instruction: str,
             reference_images: Optional[list[tuple[str, Image.Image]]] = None,
             max_new_tokens: int = 256) -> Action:
        # UI-TARS emits coords in the model's resized image space; compute the
        # mapping back to the screenshot's pixel space.
        orig_w, orig_h = screenshot.size
        new_h, new_w = _smart_resize(orig_h, orig_w, factor=self.factor,
                                     min_pixels=self.min_pixels,
                                     max_pixels=self.max_pixels)
        scale_x = orig_w / new_w
        scale_y = orig_h / new_h

        if not self.history:
            sys_content: list[dict] = [
                {"type": "text", "text": SYSTEM_PROMPT.format(instruction=instruction)}
            ]
            for name, img in (reference_images or []):
                sys_content.append({"type": "text",
                                    "text": f"\nReference image \"{name}\" (use this as a visual"
                                            f" target / hint for the task):"})
                sys_content.append({"type": "image", "image": img})
            self.history.append({"role": "system", "content": sys_content})

        self.history.append({
            "role": "user",
            "content": [{"type": "image", "image": screenshot}],
        })

        # Bound VRAM: keep only the newest N screenshots in the prompt (each is
        # ~1.3K vision tokens at this checkpoint's max_pixels). Older user turns
        # keep their position but their image is replaced with a placeholder;
        # the system prompt's reference images are always kept.
        prompt_history = []
        user_image_turns = [i for i, m in enumerate(self.history)
                            if m["role"] == "user"
                            and any(c.get("type") == "image" for c in m["content"])]
        drop = set(user_image_turns[:-self.max_history_screenshots])
        for i, m in enumerate(self.history):
            if i in drop:
                prompt_history.append({
                    "role": "user",
                    "content": [{"type": "text", "text": "(earlier screenshot omitted)"}],
                })
            else:
                prompt_history.append(m)

        text = self.processor.apply_chat_template(
            prompt_history, tokenize=False, add_generation_prompt=True
        )
        images = [c["image"] for m in prompt_history for c in m["content"]
                  if isinstance(c, dict) and c.get("type") == "image"]
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        gen = out[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.batch_decode(gen, skip_special_tokens=True)[0]

        self.history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": decoded}],
        })

        return _parse_action(decoded, scale_x, scale_y)
