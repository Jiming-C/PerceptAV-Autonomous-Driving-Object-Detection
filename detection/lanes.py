"""
lanes.py — classical-CV lane detection.

    frame
      │
      ├─ grayscale → CLAHE → Gaussian blur → Canny            (edges)
      ├─ trapezoidal region-of-interest mask                  (road only)
      ├─ probabilistic Hough transform                        (segments)
      ├─ slope filter → left/right classification
      ├─ length-weighted average + MAD outlier rejection      (one line each)
      ├─ temporal EMA with per-lane confidence and gating     (steady lines)
      └─ vanishing-point clamp                                (drawable lines)

Nothing in this module holds global state. A run's temporal state lives in an
`EgoLaneTracker` or `MultiLaneTracker` instance owned by the caller, so two
videos processed concurrently cannot bleed into each other — which the previous
module-global implementation could not promise.

Geometry convention: a lane is stored as `LaneLine(slope, intercept)` in image
coordinates (y grows downward), the form `y = slope * x + intercept`. Storing
the fit rather than two endpoints is what lets the tracker average lanes across
frames without the average drifting as the drawn endpoints move.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, NamedTuple, Optional, Sequence

import cv2
import numpy as np

from .config import LANE_COLOR, LaneConfig


class LaneLine(NamedTuple):
    """A fitted lane line: y = slope * x + intercept, in image coordinates."""

    slope: float
    intercept: float

    def x_at(self, y: float) -> float:
        """The x where this line crosses row `y`. Undefined for a flat line."""
        if self.slope == 0:
            raise ZeroDivisionError("horizontal lane line has no unique x")
        return (y - self.intercept) / self.slope


@dataclass(frozen=True)
class LaneFit:
    """A fitted line plus the rows over which Hough actually saw evidence.

    Keeping `y_near`/`y_far` alongside the fit is what lets multi-lane mode
    draw each line only as far as it was observed, instead of extrapolating a
    two-metre scrap of paint across the whole frame.
    """

    line: LaneLine
    y_near: float  # largest y — closest to the car
    y_far: float   # smallest y — furthest away


class Drawable(NamedTuple):
    """A lane line resolved to pixel endpoints, with a fade weight."""

    x1: int
    y1: int
    x2: int
    y2: int
    alpha: float


# ── Preprocessing ──────────────────────────────────────────────────────────


def mask_hood(frame: np.ndarray, hood_fraction: float) -> np.ndarray:
    """Return a copy of `frame` with the car's own hood blacked out.

    Returns a copy rather than mutating in place: the caller usually still
    wants the untouched frame, and an in-place mask on a VideoCapture buffer
    is the kind of aliasing bug that only shows up once something else starts
    reading the same array.
    """
    if not 0.0 < hood_fraction <= 1.0:
        raise ValueError(f"hood_fraction must be in (0, 1], got {hood_fraction}")
    masked = frame.copy()
    hood_top = int(frame.shape[0] * hood_fraction)
    masked[hood_top:, :] = 0
    return masked


def _edges(frame: np.ndarray, cfg: LaneConfig) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=(cfg.clahe_tile_grid, cfg.clahe_tile_grid),
    )
    gray = clahe.apply(gray)
    k = cfg.blur_kernel | 1  # cv2 requires an odd kernel
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    return cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)


def roi_vertices(
    height: int, width: int, cfg: LaneConfig, y_bottom: Optional[int] = None
) -> np.ndarray:
    """Trapezoid over the road ahead, as a single OpenCV polygon.

    `y_bottom` defaults to the frame bottom but should be set to the hood line
    when the hood is masked: the mask's own straight edge is a strong Canny
    response spanning the full width, and keeping it out of the ROI is cheaper
    than relying on the slope filter to reject it afterwards.
    """
    y_bottom = height if y_bottom is None else int(y_bottom)
    y_top = int(height * cfg.roi_y_top_frac)
    apex_half = cfg.roi_apex_half_width_frac * width
    cx = width / 2.0
    return np.array(
        [
            [
                (0, y_bottom),
                (width, y_bottom),
                (int(cx + apex_half), y_top),
                (int(cx - apex_half), y_top),
            ]
        ],
        dtype=np.int32,
    )


def region_of_interest(img: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)


# ── Segment fitting ────────────────────────────────────────────────────────


def _segment_fits(
    segments: Iterable[Sequence[float]], cfg: LaneConfig
) -> tuple[list, list]:
    """Split Hough segments into (left, right) lists of (slope, intercept, length).

    Rejects vertical segments (no slope) and anything outside the slope band.
    """
    left, right = [], []
    for x1, y1, x2, y2 in segments:
        if x2 == x1:
            continue  # vertical: slope undefined, and never a lane line anyway
        slope = (y2 - y1) / (x2 - x1)
        if not cfg.slope_min_abs < abs(slope) < cfg.slope_max_abs:
            continue
        intercept = y1 - slope * x1
        length = float(np.hypot(x2 - x1, y2 - y1))
        entry = (slope, intercept, length, min(y1, y2), max(y1, y2))
        (left if slope < 0 else right).append(entry)
    return left, right


def _reject_outliers(fits: Sequence[tuple], cfg: LaneConfig) -> list[tuple]:
    """Drop segments whose slope sits far from the group's median slope.

    Uses median absolute deviation rather than standard deviation: the
    outliers being filtered are exactly what inflates a standard deviation,
    so a std-based threshold widens to accommodate them and keeps them.
    """
    if len(fits) < cfg.min_segments_for_outlier_rejection:
        return list(fits)
    slopes = np.array([f[0] for f in fits], dtype=float)
    median = float(np.median(slopes))
    mad = float(np.median(np.abs(slopes - median)))
    if mad <= 1e-9:
        return list(fits)  # already unanimous
    keep = np.abs(slopes - median) <= cfg.outlier_mad_scale * mad
    kept = [f for f, k in zip(fits, keep) if k]
    return kept or list(fits)


def _average(fits: Sequence[tuple], cfg: LaneConfig) -> Optional[LaneFit]:
    """Length-weighted average of a group of segments into a single LaneFit."""
    fits = _reject_outliers(fits, cfg)
    if not fits:
        return None
    slopes = np.array([f[0] for f in fits], dtype=float)
    intercepts = np.array([f[1] for f in fits], dtype=float)
    weights = np.array([f[2] for f in fits], dtype=float)
    total = weights.sum()
    if total <= 0:
        return None
    line = LaneLine(
        slope=float((slopes * weights).sum() / total),
        intercept=float((intercepts * weights).sum() / total),
    )
    return LaneFit(
        line=line,
        y_near=float(max(f[4] for f in fits)),
        y_far=float(min(f[3] for f in fits)),
    )


def _hough(frame: np.ndarray, cfg: LaneConfig, y_bottom: Optional[int]):
    height, width = frame.shape[:2]
    edges = _edges(frame, cfg)
    edges = region_of_interest(edges, roi_vertices(height, width, cfg, y_bottom))
    lines = cv2.HoughLinesP(
        edges,
        rho=cfg.hough_rho,
        theta=cfg.hough_theta,
        threshold=cfg.hough_threshold,
        minLineLength=cfg.hough_min_line_length,
        maxLineGap=cfg.hough_max_line_gap,
    )
    if lines is None:
        return []
    return [tuple(line.reshape(4)) for line in lines]


# ── Detection ──────────────────────────────────────────────────────────────


def _innermost(
    fits: Sequence[tuple], reference_y: float, ego_x: float, cfg: LaneConfig
) -> list[tuple]:
    """Of the lane markings on one side, keep only the one nearest the car.

    A dashcam frame usually shows more markings than the two bounding the
    current lane — the next lane over, a hard shoulder, an exit slip. Averaging
    every same-signed segment into a single line, which is the obvious
    implementation and the one this pipeline used to have, silently blends
    those outer markings in and drags the reported lane toward the middle of
    the road. Clustering first and keeping the cluster closest to the car
    leaves the ego lane intact and costs one extra pass over the segments.
    """
    clusters = cluster_by_position(fits, reference_y, cfg.cluster_gap_px)
    if len(clusters) <= 1:
        return clusters[0] if clusters else []

    def distance(cluster) -> float:
        xs = [
            (reference_y - intercept) / slope
            for slope, intercept, *_ in cluster
            if slope != 0
        ]
        return abs(statistics.fmean(xs) - ego_x) if xs else float("inf")

    return min(clusters, key=distance)


def detect_ego_lanes(
    frame: np.ndarray,
    cfg: Optional[LaneConfig] = None,
    y_bottom: Optional[int] = None,
) -> tuple[Optional[LaneFit], Optional[LaneFit]]:
    """Detect the left and right lane lines bounding this car's lane.

    Pure: no state is kept between calls. Either side may be None when the
    frame offers no usable evidence for it.
    """
    cfg = (cfg or LaneConfig()).scaled_for(frame.shape[0])
    height, width = frame.shape[:2]
    reference_y = float(height if y_bottom is None else y_bottom)
    left_segs, right_segs = _segment_fits(_hough(frame, cfg, y_bottom), cfg)

    if cfg.ego_select_innermost:
        ego_x = width / 2.0
        left_segs = _innermost(left_segs, reference_y, ego_x, cfg)
        right_segs = _innermost(right_segs, reference_y, ego_x, cfg)

    return _average(left_segs, cfg), _average(right_segs, cfg)


def cluster_by_position(
    fits: Sequence[tuple], reference_y: float, gap_px: float
) -> list[list[tuple]]:
    """Group segments by where they extrapolate to on a shared reference row.

    Two fragments of the same lane marking can sit at completely different
    heights in the frame, so comparing their raw x coordinates groups them
    wrongly. Projecting every segment onto one row — the frame bottom — makes
    them directly comparable no matter where in the frame they were found.
    """
    if not fits:
        return []
    scored = []
    for fit in fits:
        slope, intercept = fit[0], fit[1]
        if slope == 0:
            continue
        scored.append(((reference_y - intercept) / slope, fit))
    if not scored:
        return []
    scored.sort(key=lambda t: t[0])

    clusters = [[scored[0]]]
    for x_ref, fit in scored[1:]:
        if x_ref - clusters[-1][-1][0] <= gap_px:
            clusters[-1].append((x_ref, fit))
        else:
            clusters.append([(x_ref, fit)])
    return [[fit for _, fit in cluster] for cluster in clusters]


def detect_multilanes(
    frame: np.ndarray,
    cfg: Optional[LaneConfig] = None,
    y_bottom: Optional[int] = None,
) -> list[LaneFit]:
    """Detect every lane line in the frame, not just the two ego-lane ones."""
    cfg = (cfg or LaneConfig()).scaled_for(frame.shape[0])
    height = frame.shape[0]
    # Same reference row the ego detector and the multi-lane tracker use, so a
    # lane's identity means the same thing at every stage.
    reference_y = float(height if y_bottom is None else y_bottom)
    left_segs, right_segs = _segment_fits(_hough(frame, cfg, y_bottom), cfg)

    lanes: list[LaneFit] = []
    for side in (left_segs, right_segs):
        for cluster in cluster_by_position(side, reference_y, cfg.cluster_gap_px):
            fit = _average(cluster, cfg)
            if fit is not None:
                lanes.append(fit)
    lanes.sort(key=lambda f: f.line.x_at(reference_y))
    return lanes


# ── Geometry ───────────────────────────────────────────────────────────────


def vanishing_y(left: LaneLine, right: LaneLine) -> Optional[float]:
    """Row at which two lane lines meet, or None if they are parallel."""
    if left.slope == right.slope:
        return None
    x = (right.intercept - left.intercept) / (left.slope - right.slope)
    return left.slope * x + left.intercept


def clamp_to_vanishing_point(
    left: Optional[LaneLine],
    right: Optional[LaneLine],
    y_bottom: float,
    y_top: float,
) -> float:
    """The row both lines should stop at, so they meet rather than cross.

    Drawing both lines up to a fixed row is what produces the X-shaped overlay
    you see when the true vanishing point sits *below* that row: past the
    meeting point each line continues out the far side of the other. Stopping
    at whichever is lower in the image — the fixed ceiling or the actual
    intersection — makes the overlay converge the way a real pair of lane
    lines does.
    """
    if left is None or right is None:
        return y_top
    y_int = vanishing_y(left, right)
    if y_int is None:
        return y_top
    return max(y_top, min(y_int, y_bottom))


def endpoints(line: LaneLine, y_bottom: float, y_top: float) -> Optional[Drawable]:
    """Resolve a fit to integer pixel endpoints spanning y_top..y_bottom."""
    if line.slope == 0:
        return None
    try:
        x1 = int(round(line.x_at(y_bottom)))
        x2 = int(round(line.x_at(y_top)))
    except (ZeroDivisionError, OverflowError, ValueError):
        return None
    return Drawable(x1, int(round(y_bottom)), x2, int(round(y_top)), 1.0)


# ── Temporal smoothing ─────────────────────────────────────────────────────


def _blend(prev: LaneLine, new: LaneLine, alpha: float) -> LaneLine:
    return LaneLine(
        slope=alpha * new.slope + (1 - alpha) * prev.slope,
        intercept=alpha * new.intercept + (1 - alpha) * prev.intercept,
    )


class _Track:
    """One lane line followed across frames."""

    __slots__ = ("line", "age", "y_near", "y_far")

    def __init__(self, fit: LaneFit):
        self.line = fit.line
        self.y_near = fit.y_near
        self.y_far = fit.y_far
        self.age = 1

    def hit(self, fit: LaneFit, cfg: LaneConfig) -> None:
        self.line = _blend(self.line, fit.line, cfg.smoothing)
        self.y_near = _blend_scalar(self.y_near, fit.y_near, cfg.smoothing)
        self.y_far = _blend_scalar(self.y_far, fit.y_far, cfg.smoothing)
        self.age = min(cfg.fade_frames, self.age + 1)

    def miss(self) -> None:
        self.age -= 1

    def alpha(self, cfg: LaneConfig) -> float:
        return max(0.0, min(1.0, self.age / cfg.fade_frames))


def _blend_scalar(prev: float, new: float, alpha: float) -> float:
    return alpha * new + (1 - alpha) * prev


class EgoLaneTracker:
    """Smooths the two ego-lane lines across frames.

    Three behaviours the single-shot detector cannot provide on its own:

    * **Fade, per lane.** Each side carries its own confidence, so losing the
      right line for a few frames dims only the right line. The previous
      implementation shared one confidence between both, which meant a solid
      left line held a long-gone right line at full opacity.
    * **Gating.** A detection that moves a lane's slope further in one frame
      than any real lane line can move is counted as a miss rather than
      blended in, which stops one bad frame from dragging the average.
    * **Coasting.** With no detection at all the last good line is still
      drawn, fading out over `fade_frames`, rather than blinking off.
    """

    def __init__(self, cfg: Optional[LaneConfig] = None):
        self.cfg = cfg or LaneConfig()
        self._tracks: dict[str, Optional[_Track]] = {"left": None, "right": None}

    def reset(self) -> None:
        self._tracks = {"left": None, "right": None}

    def _step(self, key: str, fit: Optional[LaneFit]) -> None:
        cfg = self.cfg
        track = self._tracks[key]

        if fit is not None and track is not None and track.age >= cfg.gate_after_age:
            if abs(fit.line.slope - track.line.slope) > cfg.gate_slope_delta:
                fit = None  # implausible jump — treat as a miss

        if fit is None:
            if track is not None:
                track.miss()
                if track.age <= 0:
                    self._tracks[key] = None  # let the lane re-acquire cleanly
            return

        if track is None:
            self._tracks[key] = _Track(fit)
        else:
            track.hit(fit, cfg)

    def update(
        self,
        left: Optional[LaneFit],
        right: Optional[LaneFit],
        height: int,
        y_bottom: Optional[float] = None,
    ) -> list[Drawable]:
        """Fold this frame's detections in and return what to draw."""
        cfg = self.cfg
        self._step("left", left)
        self._step("right", right)

        y_bottom = float(height if y_bottom is None else y_bottom)
        y_top = float(height * cfg.roi_y_top_frac)

        lt, rt = self._tracks["left"], self._tracks["right"]
        y_stop = clamp_to_vanishing_point(
            lt.line if lt else None, rt.line if rt else None, y_bottom, y_top
        )
        if y_bottom - y_stop < height * cfg.min_draw_span_frac:
            return []  # lines meet at the bumper: degenerate, draw nothing

        drawables = []
        for track in (lt, rt):
            if track is None or track.age <= 0:
                continue
            drawable = endpoints(track.line, y_bottom, y_stop)
            if drawable is not None:
                drawables.append(drawable._replace(alpha=track.alpha(cfg)))
        return drawables

    @property
    def lines(self) -> tuple[Optional[LaneLine], Optional[LaneLine]]:
        """Current smoothed (left, right) fits — useful for tests and metrics."""
        lt, rt = self._tracks["left"], self._tracks["right"]
        return (lt.line if lt else None, rt.line if rt else None)


class MultiLaneTracker:
    """Smooths a variable number of lane lines across frames.

    The reference implementation this pipeline borrows from lists unstable
    frame-to-frame lane counts as an open limitation, because multi-lane mode
    re-detects from scratch every frame. This closes it: each detected lane is
    matched to an existing track by where it crosses the bottom row, matched
    tracks are averaged rather than replaced, and an unmatched track fades out
    over `fade_frames` instead of vanishing the instant one frame misses it.
    """

    def __init__(self, cfg: Optional[LaneConfig] = None):
        self.cfg = cfg or LaneConfig()
        self._tracks: list[_Track] = []

    def reset(self) -> None:
        self._tracks = []

    def update(
        self, fits: Sequence[LaneFit], height: int, y_bottom: Optional[float] = None
    ) -> list[Drawable]:
        cfg = self.cfg.scaled_for(height)
        y_bottom = float(height if y_bottom is None else y_bottom)

        unmatched = list(self._tracks)
        for fit in sorted(fits, key=lambda f: f.line.x_at(y_bottom)):
            x_new = fit.line.x_at(y_bottom)
            best, best_dist = None, cfg.multilane_match_px
            for track in unmatched:
                if track.line.slope == 0:
                    continue
                dist = abs(track.line.x_at(y_bottom) - x_new)
                if dist < best_dist:
                    best, best_dist = track, dist
            if best is None:
                self._tracks.append(_Track(fit))
            else:
                best.hit(fit, cfg)
                unmatched.remove(best)

        for track in unmatched:
            track.miss()
        self._tracks = [t for t in self._tracks if t.age > 0]

        drawables = []
        for track in self._tracks:
            drawable = endpoints(track.line, track.y_near, track.y_far)
            if drawable is None:
                continue
            if abs(drawable.y1 - drawable.y2) < height * cfg.min_draw_span_frac:
                continue
            drawables.append(drawable._replace(alpha=track.alpha(cfg)))
        return drawables

    @property
    def lane_count(self) -> int:
        return len(self._tracks)


# ── Drawing ────────────────────────────────────────────────────────────────


def draw_lanes(
    frame: np.ndarray,
    drawables: Sequence[Drawable],
    color: tuple[int, int, int] = LANE_COLOR,
    thickness: int = 6,
) -> np.ndarray:
    """Composite lane lines onto a frame, preserving both colour and brightness.

    Two things this gets right that the obvious implementations do not:

    * It does not dim the frame. The previous version blended at
      `addWeighted(frame, 0.8, overlay, ...)`, darkening every pixel of every
      output frame by 20%.
    * It alpha-composites rather than adding. Adding a colour to grey asphalt
      saturates the bright channels — a blue line drawn additively over mid-grey
      road comes out pale cyan. Interpolating toward the colour instead keeps
      blue blue, at every fade level.

    Per-line fade is carried in an 8-bit coverage mask, which also gives the
    antialiased line edges a correctly weighted blend instead of a hard step.
    """
    if not drawables:
        return frame

    visible = [d for d in drawables if d.alpha > 0.0]
    if not visible:
        return frame

    height, width = frame.shape[:2]
    pad = thickness + 2
    y0 = max(0, min(min(d.y1, d.y2) for d in visible) - pad)
    y1 = min(height, max(max(d.y1, d.y2) for d in visible) + pad)
    if y1 <= y0:
        return frame

    # Only the band the lines actually occupy is composited; on a dashcam
    # frame that is roughly a third of the image.
    band = frame[y0:y1]
    mask = np.zeros(band.shape[:2], dtype=np.uint8)
    for d in visible:
        weight = int(round(max(0.0, min(1.0, d.alpha)) * 255))
        cv2.line(
            mask,
            (d.x1, d.y1 - y0),
            (d.x2, d.y2 - y0),
            weight,
            thickness,
            cv2.LINE_AA,
        )

    out = frame.copy()
    coverage = (mask.astype(np.float32) / 255.0)[:, :, None]
    tint = np.asarray(color, dtype=np.float32)
    out[y0:y1] = (band.astype(np.float32) * (1.0 - coverage) + tint * coverage).astype(
        np.uint8
    )
    return out
