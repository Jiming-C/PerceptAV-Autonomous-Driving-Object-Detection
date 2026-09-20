"""
config.py — every tuneable parameter for the perception pipeline, in one place.

The previous version of this project buried its thresholds inside function
bodies, which made them impossible to sweep and impossible to document
honestly. Each value below carries the reasoning for why it is what it is.

Two conventions worth knowing before you change anything:

1. Image coordinates. OpenCV's y axis grows *downward*, so "above the
   horizon" means a *smaller* y. A lane line's slope is negative on the left
   of the frame and positive on the right.

2. Resolution scaling. The Hough parameters were tuned at 720p. Vote counts
   and pixel lengths do not transfer between resolutions on their own — the
   same lane marking produces roughly twice as many votes at 1440p as at
   720p. `scaled_for()` rescales the pixel-denominated fields so a config
   tuned once keeps working on 480p phone clips and 4K dashcams alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

# Height the defaults below were tuned at. See scaled_for().
REFERENCE_HEIGHT = 720

# COCO class ids that matter for a driving scene. YOLOv8 knows 80 classes;
# reporting "potted plant" and "handbag" on a dashcam is noise, and filtering
# at inference time is cheaper than filtering afterwards.
PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRAIN, TRUCK = 0, 1, 2, 3, 5, 6, 7
TRAFFIC_LIGHT, STOP_SIGN = 9, 11

AV_CLASSES: tuple[int, ...] = (
    PERSON,
    BICYCLE,
    CAR,
    MOTORCYCLE,
    BUS,
    TRAIN,
    TRUCK,
    TRAFFIC_LIGHT,
    STOP_SIGN,
)

# Boxes are coloured by what the object means to a driver, not by class id:
# something you can hit, something that can hit you, something telling you
# what to do. Three colours read faster than nine at 30 fps. BGR.
VEHICLE_COLOR = (0, 200, 60)      # green
VRU_COLOR = (0, 165, 255)         # orange — pedestrians, cyclists, riders
CONTROL_COLOR = (0, 255, 255)     # yellow — lights and signs
LANE_COLOR = (255, 90, 0)         # blue — kept distinct from every box colour

CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    PERSON: VRU_COLOR,
    BICYCLE: VRU_COLOR,
    MOTORCYCLE: VRU_COLOR,
    CAR: VEHICLE_COLOR,
    BUS: VEHICLE_COLOR,
    TRAIN: VEHICLE_COLOR,
    TRUCK: VEHICLE_COLOR,
    TRAFFIC_LIGHT: CONTROL_COLOR,
    STOP_SIGN: CONTROL_COLOR,
}


@dataclass(frozen=True)
class LaneConfig:
    """Parameters for the classical lane pipeline."""

    # ── Preprocessing ──────────────────────────────────────────────────
    # CLAHE equalises local contrast, which recovers worn paint in shadow
    # and under overexposed sky. Global histogram equalisation blows out
    # the sky instead; the tiled version does not.
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    blur_kernel: int = 5  # must be odd; 5 is the smallest that kills
    # asphalt speckle without eating dashed-line ends

    # ── Canny ──────────────────────────────────────────────────────────
    # Deliberately low: faint/worn paint matters more than the extra
    # asphalt-texture edges it lets through, because the slope filter and
    # the length-weighted average downstream reject that noise anyway.
    canny_low: int = 45
    canny_high: int = 150

    # ── Region of interest ─────────────────────────────────────────────
    # A trapezoid over the road ahead. The apex is wider than the obvious
    # choice (0.4-0.6 of the width) because a narrow apex clips the inside
    # lane on anything but a dead-straight road.
    roi_y_top_frac: float = 0.60
    roi_apex_half_width_frac: float = 0.18

    # ── Probabilistic Hough transform ──────────────────────────────────
    # rho=2 rather than 1: a 2px accumulator bin merges the two edges of a
    # single painted stripe into one line instead of reporting both.
    hough_rho: int = 2
    hough_theta: float = math.pi / 180  # 1 degree
    hough_threshold: int = 40           # min votes  (scaled by height)
    hough_min_line_length: int = 40     # px         (scaled by height)
    hough_max_line_gap: int = 100       # px         (scaled by height) —
    # generous, so a dashed line reads as one segment rather than five

    # ── Slope filter ───────────────────────────────────────────────────
    # Lower bound rejects near-horizontal edges (shadows, tar seams, the
    # hood mask boundary, the horizon). Upper bound rejects near-vertical
    # ones (guardrail posts, sign poles, the sides of nearby vehicles).
    slope_min_abs: float = 0.5
    slope_max_abs: float = 5.0

    # ── Robust averaging ───────────────────────────────────────────────
    # Segments are averaged weighted by length, then filtered by median
    # absolute deviation. MAD is used instead of standard deviation
    # because the outliers we are trying to drop inflate the standard
    # deviation itself, which makes a std-based filter keep them.
    outlier_mad_scale: float = 3.0
    min_segments_for_outlier_rejection: int = 3

    # ── Temporal smoothing ─────────────────────────────────────────────
    # EMA over (slope, intercept). Lower = steadier but slower to follow a
    # real curve.
    smoothing: float = 0.15
    fade_frames: int = 15  # frames a lost lane takes to fade out entirely
    # A lane that jumps by more than this much slope in one frame is a
    # mis-detection, not a manoeuvre: at 30fps no real lane line moves that
    # fast. Gated frames count as misses, so if the geometry really did
    # change the track fades out and re-acquires within fade_frames.
    gate_slope_delta: float = 0.75
    gate_after_age: int = 3

    # ── Ego-lane selection ─────────────────────────────────────────────
    # A frame usually contains more lane markings than the two bounding this
    # car. Averaging every right-sloping segment into one "right lane" pulls
    # the result toward the middle of the road when a second marking is
    # visible further out — on the synthetic benchmark that single failure
    # mode accounts for every large error. Instead, cluster each side and keep
    # the cluster nearest the car. Set False to recover the naive behaviour
    # (useful for reproducing the comparison in scripts/evaluate_lanes.py).
    ego_select_innermost: bool = True

    # ── Multi-lane mode ────────────────────────────────────────────────
    # Segments are grouped by where they extrapolate to on a shared
    # reference row (the frame bottom), so segments belonging to the same
    # lane but sitting at different heights still cluster together.
    cluster_gap_px: int = 150  # scaled by height
    multilane_match_px: int = 120  # scaled by height; how far a tracked
    # lane may move between frames and still be considered the same lane
    min_draw_span_frac: float = 0.05  # skip lines shorter than this

    def scaled_for(self, height: int) -> "LaneConfig":
        """Rescale the pixel- and vote-denominated fields for `height`.

        Everything else (angles, slopes, fractions, EMA rates) is already
        resolution independent and is left alone.
        """
        if height <= 0:
            raise ValueError(f"height must be positive, got {height}")
        s = height / REFERENCE_HEIGHT
        return replace(
            self,
            hough_threshold=max(10, round(self.hough_threshold * s)),
            hough_min_line_length=max(8, round(self.hough_min_line_length * s)),
            hough_max_line_gap=max(4, round(self.hough_max_line_gap * s)),
            cluster_gap_px=max(20, round(self.cluster_gap_px * s)),
            multilane_match_px=max(16, round(self.multilane_match_px * s)),
        )


@dataclass(frozen=True)
class DetectorConfig:
    """Parameters for the YOLOv8 stage."""

    model_name: str = "yolov8n.pt"
    imgsz: int = 640
    confidence: float = 0.4
    iou: float = 0.45  # NMS IoU; 0.45 keeps two cars in a queue separate
    classes: tuple[int, ...] = AV_CLASSES

    # ByteTrack assigns a persistent id to each object across frames. lapx
    # (the linear-assignment solver it needs) has been in requirements.txt
    # since the first commit but nothing used it until now. Tracking is what
    # makes motion compensation on skipped frames possible, and it turns the
    # run summary from "boxes drawn" into "distinct objects seen".
    track: bool = True
    tracker: str = "bytetrack.yaml"

    # On a skipped frame a tracked box is advanced by the velocity measured
    # between its last two inference frames instead of being redrawn where
    # it was. A track unseen for longer than this is dropped rather than
    # extrapolated into fiction.
    max_coast_frames: int = 6


@dataclass(frozen=True)
class PipelineConfig:
    """Everything the end-to-end pipeline needs."""

    # Fraction of the frame height at which the car's own hood begins. The
    # bottom of a dashcam frame is usually the car itself; masking it stops
    # YOLO from finding "a car" in your own bonnet and stops Canny from
    # tracing the reflection of the road in it.
    hood_fraction: float = 0.80
    apply_hood_mask: bool = True

    # "ego" = the two lane lines either side of this car.
    # "multi" = every lane line found, however many that is.
    # "off"  = skip lane detection entirely.
    lane_mode: str = "ego"

    frame_skip: int = 2
    lane: LaneConfig = LaneConfig()
    detector: DetectorConfig = DetectorConfig()

    def __post_init__(self) -> None:
        if self.lane_mode not in ("ego", "multi", "off"):
            raise ValueError(
                f"lane_mode must be 'ego', 'multi' or 'off', got {self.lane_mode!r}"
            )
        if not 0.0 < self.hood_fraction <= 1.0:
            raise ValueError(
                f"hood_fraction must be in (0, 1], got {self.hood_fraction}"
            )
        if self.frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {self.frame_skip}")
