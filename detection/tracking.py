"""
tracking.py — keeping boxes honest between inference frames.

Frame skipping is what makes this pipeline usable on a CPU, but it has a cost
nobody was paying attention to: on a skipped frame the previous frame's boxes
were redrawn exactly where they were. At skip=5 on 30fps footage that is a
sixth of a second of a car moving while its box stands still, and it looks
precisely as wrong as it sounds.

Objects on a road move smoothly, so the right fix is to move the box too.
Ultralytics' tracker assigns each object a persistent id, which is what lets us
measure a per-object velocity between inference frames and advance the box
along it while inference is skipped. The linear-assignment solver the tracker
needs (lapx) has been in requirements.txt since the first commit; nothing in
the project used it until now.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detected object on one frame."""

    xyxy: tuple[float, float, float, float]
    cls_id: int
    cls_name: str
    confidence: float
    track_id: Optional[int] = None
    coasted: bool = False  # True when this box was extrapolated, not observed

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class BoxPredictor:
    """Carries tracked boxes across skipped frames at constant velocity.

    Velocity is measured in pixels per frame from the last two *inference*
    frames for each track id, so it is correct regardless of the skip rate.
    Untracked detections (tracking disabled, or an object the tracker has not
    given an id) fall back to being held in place, which is the old behaviour
    and the best that can be done without an identity to follow.
    """

    def __init__(self, max_coast_frames: int = 6):
        self.max_coast_frames = max_coast_frames
        self._last: dict[int, tuple[int, np.ndarray]] = {}      # id -> (frame, xyxy)
        self._velocity: dict[int, np.ndarray] = {}              # id -> px/frame
        self._meta: dict[int, Detection] = {}                   # id -> last Detection
        self._untracked: list[Detection] = []

    def reset(self) -> None:
        self._last.clear()
        self._velocity.clear()
        self._meta.clear()
        self._untracked.clear()

    def observe(self, detections: list[Detection], frame_idx: int) -> list[Detection]:
        """Record an inference frame's detections and return them unchanged."""
        self._untracked = [d for d in detections if d.track_id is None]
        seen = set()

        for det in detections:
            if det.track_id is None:
                continue
            tid = det.track_id
            seen.add(tid)
            xyxy = np.asarray(det.xyxy, dtype=float)
            previous = self._last.get(tid)
            if previous is not None:
                prev_idx, prev_xyxy = previous
                gap = frame_idx - prev_idx
                if gap > 0:
                    measured = (xyxy - prev_xyxy) / gap
                    # Smooth the velocity: a single noisy box should not send
                    # the coasted prediction flying off across the frame.
                    known = self._velocity.get(tid)
                    self._velocity[tid] = (
                        measured if known is None else 0.5 * measured + 0.5 * known
                    )
            self._last[tid] = (frame_idx, xyxy)
            self._meta[tid] = det

        # Forget anything that has been gone longer than we are willing to coast.
        stale = [
            tid
            for tid, (idx, _) in self._last.items()
            if tid not in seen and frame_idx - idx > self.max_coast_frames
        ]
        for tid in stale:
            self._last.pop(tid, None)
            self._velocity.pop(tid, None)
            self._meta.pop(tid, None)

        return detections

    def predict(self, frame_idx: int, frame_shape: tuple[int, int]) -> list[Detection]:
        """Boxes for a frame inference was skipped on."""
        height, width = frame_shape[:2]
        out: list[Detection] = []

        for tid, (idx, xyxy) in self._last.items():
            gap = frame_idx - idx
            if gap <= 0 or gap > self.max_coast_frames:
                continue
            velocity = self._velocity.get(tid)
            moved = xyxy if velocity is None else xyxy + velocity * gap
            x1, y1, x2, y2 = (
                float(np.clip(moved[0], 0, width)),
                float(np.clip(moved[1], 0, height)),
                float(np.clip(moved[2], 0, width)),
                float(np.clip(moved[3], 0, height)),
            )
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue  # coasted clean off the edge of the frame
            out.append(replace(self._meta[tid], xyxy=(x1, y1, x2, y2), coasted=True))

        # Untracked boxes have no identity to follow, so they stay put.
        out.extend(replace(d, coasted=True) for d in self._untracked)
        return out
