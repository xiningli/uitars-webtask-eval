# UI-TARS × WebTask Arena

Evaluating [UI-TARS-1.5-7B](https://github.com/bytedance/UI-TARS) — an open
GUI-agent vision-language model — on
[WebTask Arena](https://github.com/xiningli/webtask-arena), a set of
deterministic, seeded web tasks with server-side subgoal rewards.

The model runs **locally on a single 16 GB GPU** (4-bit NF4), sees only
screenshots, and emits pixel-coordinate actions (`click/type/scroll/...`)
that a Playwright executor performs in a real chromium. No DOM access,
no accessibility tree. Every episode is graded by the arena's oracle API
and recorded as annotated frames + mp4 + a transcript of the model's
thoughts.

**[▶ 88-second demo reel](
https://github.com/user-attachments/assets/4393a589-28de-41ae-86cb-3fedb75e5abb
)** — one pass, two diagnosed failures.

## Results (seed 0, clean rendering)

| Task | Result | Diagnosis |
|---|---|---|
| `checkout-form` | **PASS** (14 steps) | Precise field clicks, correct typing, 3-step wizard + submit |
| `email-triage` | FAIL | **State-tracking**: archives *worked*, but the model never perceived rows disappearing — it re-clicked fixed coordinates while the list reflowed underneath, archiving all 3 targets plus 7 innocents while blaming "an unresponsive system" |
| `settings-panel` | FAIL (partial) | **Environment finding**: headless chromium renders native `<select>` dropdowns outside the page — the options are invisible to *any* screenshot agent. Dark mode was correctly enabled and saved |

Both failures fed design changes back into the arena: list-reflow-after-action
is a potent state-tracking trap worth promoting to an adversarial flag, and
tasks that need dropdowns should use custom-rendered ones (or run on a real
display) to be fair to screenshot agents.

*Caveats: n=1 per task, single seed, 4-bit quantization, 15-step cap. This is
a demonstration of the eval pipeline, not a benchmark claim about UI-TARS.*

## Run it

Requirements: Python 3.11+, CUDA GPU with ≥16 GB, ~31 GB disk for the model
(downloaded from Hugging Face on first run).

```bash
pip install -r requirements.txt
playwright install chromium

# terminal 1 — the environment (from a webtask-arena checkout)
uvicorn server.app:app

# terminal 2 — the agent
python3 run_arena.py --task all --seed 0
python3 run_arena.py --task checkout-form --seed 0 --headed   # watch live
```

Each episode writes `runs-arena/<stamp>_<task>_s<seed>/` with annotated
`frames/`, `run.mp4`, `transcript.txt`, and `result.json` (per-subgoal
grades). Rebuild the demo reel from the newest runs with
`python3 gen_demo_video.py`.

### Free-form web tasks (dockerized chromium)

`run.py` drives the same agent against arbitrary websites inside a docker
chromium exposed over CDP (`Dockerfile` + `cdp-proxy.sh`); the task is read
from a folder's `index.md` and may include reference images:

```bash
python3 run.py Dockerfile my-task-folder/
```

## Implementation notes

- **Coordinate mapping**: UI-TARS emits coordinates in the model's
  smart-resized image space. The resize limits are read from the loaded
  processor config at runtime — this checkpoint ships
  `max_pixels=12,845,056`, not the ~1M Qwen2.5-VL default; hardcoding the
  default skews every click by ~2%.
- **VRAM bound**: only the newest 5 screenshots stay in the prompt (~1.3K
  vision tokens each at this resolution); older turns keep their text.
- **Action parsing** handles the coordinate spellings UI-TARS emits
  (`start_box='(x,y)'`, `<point>`, `<bbox>`) and maps them back to
  screenshot pixel space.

## Related work

- [UI-TARS / UI-TARS-2](https://github.com/bytedance/UI-TARS) — the model family
  ([UI-TARS paper, arXiv:2501.12326](https://arxiv.org/abs/2501.12326))
- [WebTask Arena](https://github.com/xiningli/webtask-arena) — the environment
  and grading harness used here
- [OSWorld](https://os-world.github.io/) — real-OS benchmark for multimodal agents

## License

MIT
