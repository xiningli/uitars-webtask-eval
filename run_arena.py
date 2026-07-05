#!/usr/bin/env python3
"""Evaluate UI-TARS-1.5-7B on WebTask Arena tasks (server-graded).

The arena (https://github.com/xiningli/webtask-arena) serves deterministic, seeded web tasks with server-side subgoal rewards. This
runner resets an episode, lets the local UI-TARS model drive a real chromium
via screenshots, then grades the outcome through the arena's oracle API.

    # in another terminal, from a webtask-arena checkout: uvicorn server.app:app
    python run_arena.py --task email-triage --seed 0
    python run_arena.py --task all --seed 0 --latency-ms 800 --layout-shift

Output per episode: runs-arena/<stamp>_<task>_s<seed>/{frames/, run.mp4,
transcript.txt, result.json}
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import executor  # noqa: E402
from run import _stitch_video  # noqa: E402
from vla_agent import UITarsAgent  # noqa: E402

VIEWPORT = (1280, 800)
ALL_TASKS = ["email-triage", "settings-panel", "checkout-form"]


def _api(base: str, method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def run_episode(agent: UITarsAgent, base_url: str, task_id: str, seed: int,
                opts: dict, out_dir: Path, max_steps: int, headed: bool) -> dict:
    reset = _api(base_url, "POST", f"/api/{task_id}/reset", {"seed": seed, **opts})
    instruction = reset["instruction"]
    print(f"[arena] {task_id} seed={seed}: {instruction}")

    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    agent.reset()

    pw = browser = None
    try:
        pw, browser, ctx, page = executor.launch_local(VIEWPORT, headed=headed)
        page.goto(f"{base_url}/task/{task_id}", wait_until="domcontentloaded")
        page.wait_for_selector("#app", state="visible", timeout=20000)

        ex = executor.Executor(ctx, page, frames_dir, viewport=VIEWPORT)
        with (out_dir / "transcript.txt").open("w") as log:
            log.write(f"TASK: {instruction}\n\n")
            recent: list[str] = []
            for step in range(max_steps):
                shot = ex.screenshot()
                act = agent.step(shot, instruction)
                line = (f"[step {step:02d}] {act.type} point={act.point} "
                        f"end={act.end_point} key={act.key!r} "
                        f"dir={act.direction!r} content={act.content!r}")
                print(line)
                if act.thought:
                    print(f"           thought: {act.thought}")
                log.write(line + f"\n  thought: {act.thought}\n  raw: {act.raw}\n\n")
                log.flush()
                ex.save_frame(shot, caption=f"#{step}: {act.type}({act.point or ''})",
                              marker=act.point, end_marker=act.end_point)
                recent.append(act.raw)
                if len(recent) >= 4 and len(set(recent[-4:])) == 1:
                    print(f"[arena] aborting: agent looped on {act.raw!r}")
                    break
                if ex.execute(act):
                    break
            ex.save_frame(ex.screenshot(), caption="final state")
    finally:
        for closer in (lambda: browser.close(), lambda: pw.stop()):
            try:
                closer()
            except Exception:
                pass

    verdict = _api(base_url, "GET", f"/api/{task_id}/verify")
    result = {"task": task_id, "seed": seed, "opts": opts,
              "instruction": instruction,
              "success": verdict["success"], "subgoals": verdict["subgoals"]}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    _stitch_video(frames_dir, out_dir / "run.mp4")

    status = "PASS" if result["success"] else "FAIL"
    failed = [k for k, v in verdict["subgoals"].items() if not v]
    print(f"[arena] [{status}] {task_id} seed={seed}"
          + (f" failed_subgoals={failed}" if failed else ""))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="email-triage", help="task id or 'all'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--latency-ms", type=int, default=0)
    ap.add_argument("--layout-shift", action="store_true")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--out", default="runs-arena")
    args = ap.parse_args()

    try:
        _api(args.base_url, "GET", "/healthz")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"arena server not reachable at {args.base_url} ({e}) — start it "
                 f"from a webtask-arena checkout with: uvicorn server.app:app")

    print("[arena] loading UI-TARS-1.5-7B (4-bit)…")
    agent = UITarsAgent()

    tasks = ALL_TASKS if args.task == "all" else [args.task]
    opts = {"latency_ms": args.latency_ms, "layout_shift": args.layout_shift}
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    results = []
    for task_id in tasks:
        out_dir = Path(args.out) / f"{stamp}_{task_id}_s{args.seed}"
        results.append(run_episode(agent, args.base_url, task_id, args.seed,
                                   opts, out_dir, args.max_steps, args.headed))

    passed = sum(r["success"] for r in results)
    print(f"\n[arena] pass rate: {passed}/{len(results)}")
    return 0


if __name__ == "__main__":
    main()
