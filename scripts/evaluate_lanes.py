#!/usr/bin/env python3
"""
evaluate_lanes.py — measure the lane pipeline instead of eyeballing it.

Three suites:

  synthetic  Procedurally generated road frames with exactly known geometry.
             No download, deterministic, runs in a second. This is the one CI
             can run and the one to use when comparing parameter sets.

  tusimple   The real benchmark, if you have it. Scores against the dataset's
             own h_samples annotations.

  temporal   Does the tracker actually steady the output? Renders a synthetic
             drive and reports frame-to-frame jitter with smoothing on vs off.

Metrics
-------
detected   both lane lines returned at all
valid      the two lines converge at or above the ROI ceiling rather than
           crossing below it or diverging outward — the geometric plausibility
           check this project borrows from xpanvictor/road-computer-vision-engr
err_px     mean |x_predicted - x_truth| at the near and far probe rows
pass       detected AND valid AND err_px within tolerance

Examples
--------
  python scripts/evaluate_lanes.py
  python scripts/evaluate_lanes.py --frames 200 --difficulty 1.5
  python scripts/evaluate_lanes.py --set roi_y_top_frac=0.45 --set canny_low=30
  python scripts/evaluate_lanes.py --suite temporal
  python scripts/evaluate_lanes.py --suite tusimple --data-root data/TUSimple/train_set
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import statistics
import sys
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_road import render  # noqa: E402

from detection.config import LaneConfig  # noqa: E402
from detection.lanes import (  # noqa: E402
    EgoLaneTracker,
    LaneLine,
    detect_ego_lanes,
    mask_hood,
    vanishing_y,
)

VALID_BUFFER_FRAC = 0.06  # slack on the convergence check, as a fraction of height


# ── Metrics ────────────────────────────────────────────────────────────────


def is_valid(left: Optional[LaneLine], right: Optional[LaneLine], height: int,
             cfg: LaneConfig) -> bool:
    """Do these two lines converge where a real pair of lane lines would?

    Ported from the reference implementation's geometric plausibility check:
    the meeting point must sit at or above the ROI ceiling. A pair that meets
    below it is crossing in front of the car; a pair that never meets is
    diverging outward. Both are detector failures that a per-line error metric
    can miss when each line is individually plausible.
    """
    if left is None or right is None:
        return False
    y_int = vanishing_y(left, right)
    if y_int is None:
        return False  # parallel: no vanishing point, so not a perspective pair
    return y_int <= height * cfg.roi_y_top_frac + height * VALID_BUFFER_FRAC


def x_errors(left, right, scene, rows) -> Optional[list[float]]:
    """Absolute x error of both lines at each probe row, or None if incomplete."""
    if left is None or right is None:
        return None
    errs = []
    for y in rows:
        try:
            errs.append(abs(left.x_at(y) - scene.x_at("left", y)))
            errs.append(abs(right.x_at(y) - scene.x_at("right", y)))
        except ZeroDivisionError:
            return None
    return errs


# ── Suites ─────────────────────────────────────────────────────────────────


def run_synthetic(cfg: LaneConfig, frames: int, difficulty: float, seed: int,
                  hood_fraction: float, tol_frac: float) -> dict:
    detected = valid = passed = 0
    all_errs: list[float] = []

    for i in range(frames):
        img, scene = render(
            seed=seed + i, hood_fraction=hood_fraction, difficulty=difficulty
        )
        height, width = img.shape[:2]
        hood_line = int(height * hood_fraction)
        frame = mask_hood(img, hood_fraction)

        left_fit, right_fit = detect_ego_lanes(frame, cfg, y_bottom=hood_line)
        left = left_fit.line if left_fit else None
        right = right_fit.line if right_fit else None

        rows = (hood_line, height * cfg.roi_y_top_frac)
        ok_detected = left is not None and right is not None
        ok_valid = is_valid(left, right, height, cfg)
        errs = x_errors(left, right, scene, rows)

        detected += ok_detected
        valid += ok_valid
        if errs:
            all_errs.extend(errs)
            if ok_valid and max(errs) <= tol_frac * width:
                passed += 1

    return {
        "suite": "synthetic",
        "frames": frames,
        "difficulty": difficulty,
        "detected_pct": 100.0 * detected / frames,
        "valid_pct": 100.0 * valid / frames,
        "pass_pct": 100.0 * passed / frames,
        "mean_err_px": round(statistics.fmean(all_errs), 1) if all_errs else None,
        "median_err_px": round(statistics.median(all_errs), 1) if all_errs else None,
        "tolerance_px": round(tol_frac * 1280, 1),
    }


def run_temporal(cfg: LaneConfig, frames: int, difficulty: float, seed: int,
                 hood_fraction: float) -> dict:
    """Jitter of the drawn lane position, smoothing on vs off.

    The synthetic drive keeps one scene and dithers the camera, so any
    frame-to-frame movement in the output beyond that dither is the detector
    being unsteady rather than the road actually moving.
    """
    rng = np.random.default_rng(seed)
    base_img, scene = render(seed=seed, hood_fraction=hood_fraction,
                             difficulty=difficulty)
    height, width = base_img.shape[:2]
    hood_line = int(height * hood_fraction)
    probe_y = hood_line

    raw_x: list[float] = []
    smoothed_x: list[float] = []
    tracker = EgoLaneTracker(cfg)
    misses = 0

    for i in range(frames):
        # Re-render with fresh noise and a small camera shift: the road has
        # not moved, so an honest detector's output should barely move either.
        img, scene_i = render(
            seed=seed, hood_fraction=hood_fraction, difficulty=difficulty
        )
        shift = int(rng.normal(0, 1.5))
        img = np.roll(img, shift, axis=1)
        noise = rng.normal(0, 5, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        frame = mask_hood(img, hood_fraction)
        left_fit, right_fit = detect_ego_lanes(frame, cfg, y_bottom=hood_line)
        if left_fit is None or right_fit is None:
            misses += 1

        if left_fit and right_fit:
            raw_x.append(
                (left_fit.line.x_at(probe_y) + right_fit.line.x_at(probe_y)) / 2
            )
        drawables = tracker.update(left_fit, right_fit, height, y_bottom=hood_line)
        if len(drawables) == 2:
            smoothed_x.append((drawables[0].x1 + drawables[1].x1) / 2)

    def jitter(xs):
        if len(xs) < 3:
            return None
        return round(statistics.fmean(abs(b - a) for a, b in zip(xs, xs[1:])), 2)

    return {
        "suite": "temporal",
        "frames": frames,
        "raw_jitter_px": jitter(raw_x),
        "smoothed_jitter_px": jitter(smoothed_x),
        "frames_with_no_detection": misses,
        "frames_drawn_raw": len(raw_x),
        "frames_drawn_smoothed": len(smoothed_x),
    }


def _tusimple_frames(data_root: str) -> list[dict]:
    frames = []
    for name in sorted(os.listdir(data_root)):
        if not (name.startswith("label_data") and name.endswith(".json")):
            continue
        with open(os.path.join(data_root, name)) as fh:
            frames.extend(json.loads(line) for line in fh if line.strip())
    return frames


def _tusimple_ego_lanes(entry: dict, width: int, height: int):
    """The two annotated lanes nearest the frame centre at the lowest row."""
    h_samples = entry["h_samples"]
    candidates = []
    for xs in entry["lanes"]:
        pts = [(x, y) for x, y in zip(xs, h_samples) if x != -2]
        if len(pts) >= 2:
            candidates.append(np.array(pts, dtype=float))
    left = [p for p in candidates if p[-1][0] < width / 2]
    right = [p for p in candidates if p[-1][0] >= width / 2]
    return (
        max(left, key=lambda p: p[-1][0]) if left else None,
        min(right, key=lambda p: p[-1][0]) if right else None,
    )


def run_tusimple(cfg: LaneConfig, data_root: str, sample: int, seed: int,
                 tol_frac: float) -> dict:
    entries = _tusimple_frames(data_root)
    if not entries:
        raise SystemExit(
            f"No label_data*.json under {data_root}. Download TuSimple first — "
            "see the README."
        )
    random.seed(seed)
    entries = random.sample(entries, min(sample, len(entries)))

    detected = valid = passed = 0
    scored = 0
    all_errs: list[float] = []

    for entry in entries:
        path = os.path.join(data_root, entry["raw_file"])
        img = cv2.imread(path)
        if img is None:
            continue
        scored += 1
        height, width = img.shape[:2]
        left_fit, right_fit = detect_ego_lanes(img, cfg)
        left = left_fit.line if left_fit else None
        right = right_fit.line if right_fit else None

        detected += left is not None and right is not None
        ok_valid = is_valid(left, right, height, cfg)
        valid += ok_valid

        gt_left, gt_right = _tusimple_ego_lanes(entry, width, height)
        errs = []
        for line, gt in ((left, gt_left), (right, gt_right)):
            if line is None or gt is None:
                errs = []
                break
            for x_gt, y in gt:
                if y < height * cfg.roi_y_top_frac:
                    continue
                errs.append(abs(line.x_at(y) - x_gt))
        if errs:
            all_errs.extend(errs)
            if ok_valid and statistics.fmean(errs) <= tol_frac * width:
                passed += 1

    if not scored:
        raise SystemExit(f"No readable frames under {data_root}.")
    return {
        "suite": "tusimple",
        "frames": scored,
        "detected_pct": 100.0 * detected / scored,
        "valid_pct": 100.0 * valid / scored,
        "pass_pct": 100.0 * passed / scored,
        "mean_err_px": round(statistics.fmean(all_errs), 1) if all_errs else None,
        "median_err_px": round(statistics.median(all_errs), 1) if all_errs else None,
    }


# ── CLI ────────────────────────────────────────────────────────────────────


def _apply_overrides(cfg: LaneConfig, overrides: list[str]) -> LaneConfig:
    fields = {f.name: f.type for f in dataclasses.fields(LaneConfig)}
    values = {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects name=value, got {item!r}")
        name, raw = item.split("=", 1)
        if name not in fields:
            raise SystemExit(
                f"Unknown LaneConfig field {name!r}. Known: {', '.join(sorted(fields))}"
            )
        current = getattr(cfg, name)
        values[name] = type(current)(raw) if not isinstance(current, bool) else raw == "True"
    return dataclasses.replace(cfg, **values)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=("synthetic", "temporal", "tusimple"),
                    default="synthetic")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--difficulty", type=float, default=1.0,
                    help="synthetic nuisance level: 0 clean, 1 typical, 2 unkind")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hood-fraction", type=float, default=0.80)
    ap.add_argument("--tolerance", type=float, default=0.025,
                    help="max x error to count as a pass, as a fraction of width")
    ap.add_argument("--data-root", default="data/TUSimple/train_set")
    ap.add_argument("--sample", type=int, default=50, help="tusimple frames to score")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="FIELD=VALUE", help="override a LaneConfig field")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    ap.add_argument(
        "--min-pass",
        type=float,
        default=None,
        metavar="PCT",
        help="exit non-zero if the pass rate falls below this. Frames are "
        "generated from a fixed seed, so a given command scores identically "
        "every run and this is a regression gate, not a flaky threshold.",
    )
    args = ap.parse_args(argv)

    cfg = _apply_overrides(LaneConfig(), args.overrides)

    if args.suite == "synthetic":
        result = run_synthetic(cfg, args.frames, args.difficulty, args.seed,
                               args.hood_fraction, args.tolerance)
    elif args.suite == "temporal":
        result = run_temporal(cfg, args.frames, args.difficulty, args.seed,
                              args.hood_fraction)
    else:
        result = run_tusimple(cfg, args.data_root, args.sample, args.seed,
                              args.tolerance)

    if args.overrides:
        result["overrides"] = args.overrides

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        width = max(len(k) for k in result)
        for key, value in result.items():
            if isinstance(value, float):
                value = f"{value:.1f}"
            print(f"  {key:<{width}}  {value}")

    if args.min_pass is not None:
        actual = result.get("pass_pct")
        if actual is None:
            print(f"\n--min-pass does not apply to the {args.suite} suite.",
                  file=sys.stderr)
            return 2
        if actual < args.min_pass:
            print(
                f"\nFAIL: pass rate {actual:.1f}% is below the {args.min_pass:.1f}% "
                "floor. Lane accuracy has regressed.",
                file=sys.stderr,
            )
            return 1
        print(f"\nOK: pass rate {actual:.1f}% clears the {args.min_pass:.1f}% floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
