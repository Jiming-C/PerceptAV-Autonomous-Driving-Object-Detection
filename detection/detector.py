"""
detector.py — the end-to-end dashcam perception pipeline.

    frame ──▶ hood mask ──┬──▶ YOLOv8 (+ ByteTrack)  ──▶ boxes ─┐
                          │                                     ├──▶ annotated
                          └──▶ classical lane pipeline ──▶ lanes┘      frame

The ordering above is the important part, and it is not what this module used
to do. Lane lines were previously drawn onto the frame *before* YOLO ran, and
the compositing step dimmed every pixel by 20% on the way past, so the detector
was being handed a darkened image with blue lines painted across the road and
asked to find cars in it. Detection and lane-finding both read the same clean
frame now, and the overlays go on afterwards, where they belong.

Public API:
    process_video(path, ...) -> (output_path, RunSummary)
    process_image(path, ...) -> (output_path, RunSummary)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from .config import (
    CLASS_COLORS,
    VEHICLE_COLOR,
    DetectorConfig,
    LaneConfig,
    PipelineConfig,
)
from .lanes import (
    EgoLaneTracker,
    MultiLaneTracker,
    detect_ego_lanes,
    detect_multilanes,
    draw_lanes,
    mask_hood,
)
from .tracking import BoxPredictor, Detection

ProgressFn = Optional[Callable[[float, str], None]]

# ── Model ──────────────────────────────────────────────────────────────────

_models: dict[str, "object"] = {}


def get_model(name: str = "yolov8n.pt"):
    """Load a YOLO model once per name and cache it for the process."""
    if name not in _models:
        from ultralytics import YOLO  # imported lazily: it pulls in torch

        _models[name] = YOLO(name)
    return _models[name]


def _reset_tracker(model) -> None:
    """Clear ByteTrack's state so one video's ids never leak into the next.

    Ultralytics hangs tracker state off the predictor, which is cached on the
    model alongside the weights. Without this, the second video processed in a
    session starts counting objects from wherever the first one left off — and
    worse, can match a new object to a track from the previous clip.
    """
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", []) or []:
        reset = getattr(tracker, "reset", None)
        if callable(reset):
            reset()


# ── Summary ────────────────────────────────────────────────────────────────


@dataclass
class RunSummary:
    """What actually happened during a run.

    The module docstring has promised callers a summary alongside the output
    path since the first commit; until now it returned only the path.
    """

    frames: int = 0
    inference_frames: int = 0
    seconds: float = 0.0
    objects: dict[str, int] = field(default_factory=dict)
    tracked: bool = False
    lane_frames: int = 0
    resolution: tuple[int, int] = (0, 0)

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0

    def to_markdown(self) -> str:
        width, height = self.resolution
        counted = "distinct objects tracked" if self.tracked else "peak objects in frame"
        skipped = self.frames - self.inference_frames

        if self.frames == 1:
            lines = [f"**1 frame** at {width}×{height} in {self.seconds:.2f}s"]
        else:
            lines = [
                f"**{self.frames} frames** at {width}×{height} "
                f"in {self.seconds:.1f}s ({self.fps:.1f} fps)",
                f"Inference ran on {self.inference_frames} of them"
                + (
                    f"; the other {skipped} reused tracked boxes advanced by "
                    "their measured velocity"
                    if skipped
                    else ""
                ),
            ]
        if self.lane_frames:
            lines.append(
                "Lane lines drawn on "
                + ("this frame" if self.frames == 1
                   else f"{self.lane_frames} frames")
            )
        if self.objects:
            rows = "\n".join(
                f"| {name} | {count} |"
                for name, count in sorted(
                    self.objects.items(), key=lambda kv: -kv[1]
                )
            )
            lines.append(f"\n| class | {counted} |\n|---|---|\n{rows}")
        else:
            lines.append("\nNo objects detected above the confidence threshold.")
        return "\n\n".join(lines)


class _ObjectCounter:
    """Counts distinct tracked objects, or peak simultaneous ones without ids."""

    def __init__(self):
        self._ids: dict[str, set] = {}
        self._peak: dict[str, int] = {}

    def add(self, detections: list[Detection]) -> None:
        per_frame: dict[str, int] = {}
        for det in detections:
            if det.coasted:
                continue  # coasted boxes are re-draws, not new observations
            per_frame[det.cls_name] = per_frame.get(det.cls_name, 0) + 1
            if det.track_id is not None:
                self._ids.setdefault(det.cls_name, set()).add(det.track_id)
        for name, count in per_frame.items():
            self._peak[name] = max(self._peak.get(name, 0), count)

    def result(self) -> tuple[dict[str, int], bool]:
        if self._ids:
            return {k: len(v) for k, v in self._ids.items()}, True
        return dict(self._peak), False


# ── Inference ──────────────────────────────────────────────────────────────


def _to_detections(result, model) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    detections = []
    ids = boxes.id.int().tolist() if getattr(boxes, "id", None) is not None else None
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        detections.append(
            Detection(
                xyxy=(x1, y1, x2, y2),
                cls_id=cls_id,
                cls_name=model.names[cls_id],
                confidence=float(box.conf[0]),
                track_id=ids[i] if ids else None,
            )
        )
    return detections


def _infer(model, frame: np.ndarray, cfg: DetectorConfig) -> list[Detection]:
    kwargs = dict(
        conf=cfg.confidence,
        iou=cfg.iou,
        imgsz=cfg.imgsz,
        classes=list(cfg.classes) if cfg.classes else None,
        verbose=False,
    )
    if cfg.track:
        result = model.track(frame, persist=True, tracker=cfg.tracker, **kwargs)[0]
    else:
        result = model(frame, **kwargs)[0]
    return _to_detections(result, model)


# ── Drawing ────────────────────────────────────────────────────────────────


def _draw_boxes(frame: np.ndarray, detections: list[Detection],
                show_track_ids: bool = True) -> None:
    """Draw boxes in place, coloured by what the object means to a driver."""
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy)
        color = CLASS_COLORS.get(det.cls_id, VEHICLE_COLOR)
        # A coasted box is a prediction, not an observation. Drawing it thinner
        # is a small honesty: the viewer can see which frames were inferred.
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1 if det.coasted else 2)

        label = det.cls_name
        if show_track_ids and det.track_id is not None:
            label += f" #{det.track_id}"
        label += f" {det.confidence:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(y1 - th - 6, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 4, top + th + 6), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, top + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


# ── Lane stage ─────────────────────────────────────────────────────────────


class _LaneStage:
    """Runs whichever lane mode was asked for and keeps its temporal state."""

    def __init__(self, mode: str, cfg: LaneConfig, hood_line: Optional[int]):
        self.mode = mode
        self.cfg = cfg
        self.hood_line = hood_line
        self.tracker = (
            EgoLaneTracker(cfg)
            if mode == "ego"
            else MultiLaneTracker(cfg)
            if mode == "multi"
            else None
        )

    def __call__(self, frame: np.ndarray) -> list:
        if self.tracker is None:
            return []
        height = frame.shape[0]
        if self.mode == "ego":
            left, right = detect_ego_lanes(frame, self.cfg, y_bottom=self.hood_line)
            return self.tracker.update(left, right, height, y_bottom=self.hood_line)
        fits = detect_multilanes(frame, self.cfg, y_bottom=self.hood_line)
        return self.tracker.update(fits, height, y_bottom=self.hood_line)


def _build_config(
    confidence: float,
    frame_skip: int,
    apply_hood_mask: bool,
    lane_mode: str,
    config: Optional[PipelineConfig],
) -> PipelineConfig:
    """Fold the loose keyword arguments the UI passes into a PipelineConfig."""
    base = config or PipelineConfig()
    import dataclasses

    return dataclasses.replace(
        base,
        apply_hood_mask=apply_hood_mask,
        lane_mode=lane_mode,
        frame_skip=max(1, int(frame_skip)),
        detector=dataclasses.replace(base.detector, confidence=float(confidence)),
    )


# ── Public API ─────────────────────────────────────────────────────────────


def process_video(
    video_path: str,
    confidence: float = 0.4,
    frame_skip: int = 2,
    apply_hood_mask: bool = True,
    lane_mode: str = "ego",
    config: Optional[PipelineConfig] = None,
    progress: ProgressFn = None,
) -> tuple[str, RunSummary]:
    """Run the perception pipeline over a video.

    Args:
        video_path: Input video.
        confidence: Minimum detection confidence, 0-1.
        frame_skip: Run inference every Nth frame. Skipped frames keep their
            tracked boxes moving at the velocity measured between the last two
            inference frames, rather than freezing them where they were.
        apply_hood_mask: Black out the car's own hood before anything reads
            the frame.
        lane_mode: "ego" (the two lines around this car), "multi" (every line
            found) or "off".
        config: Full PipelineConfig, if you want control beyond the above.
        progress: Optional callback, called as progress(fraction, message).

    Returns:
        (output_path, summary) — an H.264 file playable in a browser, and a
        RunSummary describing the run.
    """
    cfg = _build_config(confidence, frame_skip, apply_hood_mask, lane_mode, config)

    # Validate the input before loading the model: a typo in a path should not
    # cost a weights download and a few seconds of torch startup first.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        model = get_model(cfg.detector.model_name)
        _reset_tracker(model)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if width <= 0 or height <= 0:
            raise ValueError(f"Video reports an empty frame size: {video_path}")

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = tmp.name
        tmp.close()
        writer = cv2.VideoWriter(
            tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            os.unlink(tmp_path)
            raise RuntimeError(
                f"Cannot open a video writer for {width}x{height} @ {fps:.1f}fps"
            )

        hood_line = int(height * cfg.hood_fraction) if cfg.apply_hood_mask else None
        lane_stage = _LaneStage(cfg.lane_mode, cfg.lane, hood_line)
        predictor = BoxPredictor(cfg.detector.max_coast_frames)
        counter = _ObjectCounter()
        summary = RunSummary(resolution=(width, height))
        started = time.perf_counter()

        try:
            frame_idx = 0
            while True:
                ok, raw = cap.read()
                if not ok:
                    break

                # One clean frame feeds both stages. Nothing is drawn on it.
                frame = mask_hood(raw, cfg.hood_fraction) if cfg.apply_hood_mask else raw

                if frame_idx % cfg.frame_skip == 0:
                    detections = predictor.observe(
                        _infer(model, frame, cfg.detector), frame_idx
                    )
                    summary.inference_frames += 1
                else:
                    detections = predictor.predict(frame_idx, frame.shape)

                counter.add(detections)
                drawables = lane_stage(frame)
                summary.lane_frames += bool(drawables)

                annotated = draw_lanes(frame, drawables)
                if annotated is frame:
                    annotated = frame.copy()  # never scribble on the source frame
                _draw_boxes(annotated, detections)
                writer.write(annotated)

                frame_idx += 1
                if progress and total and frame_idx % 10 == 0:
                    progress(frame_idx / total, f"frame {frame_idx}/{total}")
        finally:
            writer.release()

        summary.frames = frame_idx
        summary.seconds = time.perf_counter() - started
        summary.objects, summary.tracked = counter.result()
    finally:
        cap.release()

    if summary.frames == 0:
        os.unlink(tmp_path)
        raise ValueError(f"No frames could be read from: {video_path}")

    if progress:
        progress(1.0, "encoding")
    return _transcode_to_h264(tmp_path), summary


def process_image(
    image_path: str,
    confidence: float = 0.4,
    apply_hood_mask: bool = True,
    lane_mode: str = "ego",
    config: Optional[PipelineConfig] = None,
) -> tuple[str, RunSummary]:
    """Run the perception pipeline over a single image.

    Returns (output_path, summary). Tracking is disabled for a single frame —
    there is nothing to track across — so the summary counts objects found.
    """
    cfg = _build_config(confidence, 1, apply_hood_mask, lane_mode, config)
    import dataclasses

    cfg = dataclasses.replace(cfg, detector=dataclasses.replace(cfg.detector, track=False))

    raw = cv2.imread(image_path)
    if raw is None:
        raise ValueError(f"Cannot open image: {image_path}")

    model = get_model(cfg.detector.model_name)

    height, width = raw.shape[:2]
    frame = mask_hood(raw, cfg.hood_fraction) if cfg.apply_hood_mask else raw

    started = time.perf_counter()
    detections = _infer(model, frame, cfg.detector)
    hood_line = int(height * cfg.hood_fraction) if cfg.apply_hood_mask else None
    drawables = _LaneStage(cfg.lane_mode, cfg.lane, hood_line)(frame)

    annotated = draw_lanes(frame, drawables)
    if annotated is frame:
        annotated = frame.copy()
    _draw_boxes(annotated, detections)

    counter = _ObjectCounter()
    counter.add(detections)
    objects, tracked = counter.result()
    summary = RunSummary(
        frames=1,
        inference_frames=1,
        seconds=time.perf_counter() - started,
        objects=objects,
        tracked=tracked,
        lane_frames=int(bool(drawables)),
        resolution=(width, height),
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    output_path = tmp.name
    tmp.close()
    cv2.imwrite(output_path, annotated)
    return output_path, summary


# ── H.264 transcoding ──────────────────────────────────────────────────────


def _transcode_to_h264(input_path: str) -> str:
    """Transcode to web-safe H.264. OpenCV writes mp4v, which browsers reject."""
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    try:
        import imageio_ffmpeg  # inside the try: a missing ffmpeg is a fallback,
        # not a crash, and that is exactly what the except below is for

        completed = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-i", input_path,
                "-vcodec", "libx264",
                "-crf", "28",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                # Put the index at the front so the browser can start playing
                # before the whole file has downloaded.
                "-movflags", "+faststart",
                output_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        del completed
        os.remove(input_path)
        return output_path
    except (subprocess.CalledProcessError, OSError, ImportError) as exc:
        # Returning the mp4v file is better than returning nothing, but it will
        # very likely not play in a browser — so say why, loudly, rather than
        # letting it look like a mysteriously blank video player.
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            detail = exc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:]
            detail = f" — {detail[0]}" if detail else ""
        print(
            f"ffmpeg transcode failed{detail}. Returning the raw mp4v file, "
            "which most browsers cannot play."
        )
        if os.path.exists(output_path):
            os.remove(output_path)
        return input_path
