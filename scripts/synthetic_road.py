"""
synthetic_road.py — procedurally generated road frames with known lane geometry.

Why this exists: the only honest way to say a change to the lane pipeline is an
improvement is to measure it, and the obvious benchmark (TuSimple) is a ~10 GB
download that cannot run in CI and that nobody reviewing a pull request is going
to fetch. These frames are a weaker proxy than real footage — they will not tell
you how the pipeline handles rain or worn paint — but they exercise every stage
of it (perspective convergence, dashed lines, asphalt texture, shadows, vertical
distractors, hood occlusion) against exactly known geometry, they are
deterministic given a seed, and they run in under a second.

Use them to catch regressions and to compare parameter sets. Use real footage to
decide whether the pipeline is any good.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class Scene:
    """Ground truth for one generated frame."""

    width: int
    height: int
    vp_x: float           # vanishing point, x
    vp_y: float           # vanishing point, y
    left_x_bottom: float  # where the left lane line crosses the frame bottom
    right_x_bottom: float

    def left_line(self) -> tuple[float, float]:
        """Ground-truth left lane as (slope, intercept) in image coordinates."""
        return _line_through((self.left_x_bottom, self.height), (self.vp_x, self.vp_y))

    def right_line(self) -> tuple[float, float]:
        return _line_through((self.right_x_bottom, self.height), (self.vp_x, self.vp_y))

    def x_at(self, side: str, y: float) -> float:
        slope, intercept = self.left_line() if side == "left" else self.right_line()
        return (y - intercept) / slope


def _line_through(p1, p2) -> tuple[float, float]:
    (x1, y1), (x2, y2) = p1, p2
    slope = (y2 - y1) / (x2 - x1)
    return slope, y1 - slope * x1


def _stripe_quad(scene: Scene, x_bottom: float, width_bottom: float,
                 y_near: float, y_far: float) -> np.ndarray:
    """The painted stripe's cross-section between two rows.

    Both edges are computed at the rows asked for — a stripe segment high in
    the frame must not be anchored to the frame bottom, or every dash merges
    into one solid line.
    """
    slope, intercept = _line_through((x_bottom, scene.height), (scene.vp_x, scene.vp_y))

    def edge(y):
        x = (y - intercept) / slope
        # Stripe width shrinks linearly toward the vanishing point.
        t = (scene.height - y) / max(1.0, scene.height - scene.vp_y)
        half = width_bottom * max(0.15, 1.0 - t) / 2.0
        return x - half, x + half

    lx_near, rx_near = edge(y_near)
    lx_far, rx_far = edge(y_far)
    return np.array(
        [[lx_near, y_near], [rx_near, y_near], [rx_far, y_far], [lx_far, y_far]],
        dtype=np.int32,
    )


def _draw_stripe(img, scene, x_bottom, width_bottom, color, dashed, rng):
    """Draw a lane marking, solid or dashed, from the frame bottom to the horizon."""
    y_near = float(scene.height)
    y_far = scene.vp_y + 0.05 * scene.height
    if not dashed:
        cv2.fillPoly(img, [_stripe_quad(scene, x_bottom, width_bottom, y_near, y_far)], color)
        return
    # Dashes get shorter with distance, as they do in a real perspective view.
    span = y_near - y_far
    y = y_near
    while y > y_far:
        frac = (y_near - y) / span
        dash = max(6.0, 0.10 * span * (1.0 - 0.75 * frac))
        gap = dash * 1.6
        y_end = max(y_far, y - dash)
        cv2.fillPoly(img, [_stripe_quad(scene, x_bottom, width_bottom, y, y_end)], color)
        y = y_end - gap


def render(
    seed: int = 0,
    width: int = 1280,
    height: int = 720,
    hood_fraction: Optional[float] = 0.80,
    dashed_right: bool = True,
    distractors: bool = True,
    difficulty: float = 1.0,
) -> tuple[np.ndarray, Scene]:
    """Render one road frame and the ground truth that generated it.

    `difficulty` scales the nuisances — noise, shadows, paint wear, distractor
    strength. 0 is a clean diagram; 1 is roughly what a decent dashcam sees on
    an overcast day; 2 is unkind.
    """
    rng = np.random.default_rng(seed)

    vp_x = width * float(rng.uniform(0.42, 0.58))
    vp_y = height * float(rng.uniform(0.38, 0.48))
    half = width * float(rng.uniform(0.22, 0.34))
    centre = width * float(rng.uniform(0.44, 0.56))
    scene = Scene(width, height, vp_x, vp_y, centre - half, centre + half)

    # Asphalt: darker toward the horizon, with texture.
    img = np.zeros((height, width, 3), dtype=np.uint8)
    ramp = np.linspace(58, 118, height, dtype=np.float32)[:, None]
    img[:] = np.repeat(ramp, width, axis=1)[:, :, None].astype(np.uint8)
    # Sky above the horizon.
    img[: int(vp_y), :] = (200, 195, 185)

    # Paint. Real markings are never pure white and never uniform.
    paint = int(rng.integers(195, 250)) - int(30 * difficulty * rng.random())
    stripe_w = width * 0.020
    _draw_stripe(img, scene, scene.left_x_bottom, stripe_w, (paint,) * 3, False, rng)
    _draw_stripe(
        img, scene, scene.right_x_bottom, stripe_w, (paint,) * 3, dashed_right, rng
    )

    if distractors:
        # A third marking outside the ego lane — the multi-lane case, and a
        # chance for the ego detector to grab the wrong line.
        outer = scene.right_x_bottom + half * 0.9
        if outer < width * 1.15:
            _draw_stripe(img, scene, outer, stripe_w * 0.9, (paint - 20,) * 3, True, rng)
        # Guardrail: a near-horizontal edge just under the horizon.
        y_rail = int(vp_y + 0.03 * height)
        cv2.line(img, (0, y_rail), (width, y_rail), (150, 150, 150), 3)
        # Vertical posts — the classic near-vertical false positive.
        for x in range(0, width, max(40, width // 16)):
            cv2.line(img, (x, y_rail), (x, y_rail + int(0.03 * height)), (140, 140, 140), 2)
        # A vehicle ahead, casting a hard box edge into the ROI.
        cx = int(vp_x + rng.uniform(-0.12, 0.12) * width)
        cy = int(vp_y + 0.14 * height)
        cw, ch = int(0.10 * width), int(0.07 * height)
        cv2.rectangle(img, (cx - cw // 2, cy - ch), (cx + cw // 2, cy), (45, 45, 55), -1)

    if difficulty > 0:
        # Shadow bands across the road — the reason CLAHE is in the pipeline.
        for _ in range(int(2 * difficulty)):
            y0 = int(rng.uniform(vp_y, height))
            h = int(rng.uniform(0.03, 0.10) * height)
            band = img[y0 : y0 + h].astype(np.float32) * float(rng.uniform(0.55, 0.8))
            img[y0 : y0 + h] = band.astype(np.uint8)
        # Sensor noise.
        noise = rng.normal(0, 6 * difficulty, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if hood_fraction is not None:
        hood_top = int(height * hood_fraction)
        img[hood_top:] = (28, 28, 32)
        cv2.line(img, (0, hood_top), (width, hood_top), (90, 90, 95), 2)

    return img, scene
