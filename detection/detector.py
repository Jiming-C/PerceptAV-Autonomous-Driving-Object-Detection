"""
detector.py — YOLOv8 object detection for dashcam videos and images.

Provides two public functions:
  - process_video(video_path, confidence) -> (output_path, summary)
  - process_image(image_path, confidence) -> (output_path, summary)
"""

import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO

# ── Model ──────────────────────────────────────────────────────────────────

MODEL_NAME = "yolov8n.pt"
_model = None


def get_model():
    """Load the YOLOv8 model once and cache it globally."""
    global _model
    if _model is None:
        _model = YOLO(MODEL_NAME)
    return _model


# ── Hood mask ──────────────────────────────────────────────────────────────
# Many dashcam videos show the car's own hood at the bottom of the frame.
# We mask that region so YOLO doesn't waste detections on it.


def _mask_hood(frame):
    """Black-out the bottom 20% of the frame (the car hood region)."""
    h = frame.shape[0]
    hood_top = int(h * 0.80)
    frame[hood_top:, :] = 0
    return frame


# ── Lane detection ─────────────────────────────────────────────────────────

# Exponential moving average state for left/right lanes: (slope, intercept) or None
_lane_state = {"left": None, "right": None}
# How many consecutive frames each lane has been detected (used to fade stale lines)
_lane_age = {"left": 0, "right": 0}
_FADE_FRAMES = 15   # frames until a lost lane fully fades out
_SMOOTH = 0.15  # EMA factor — lower = smoother but slower to react to real changes
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _reset_lane_state():
    """Reset lane EMA state between videos so they don't bleed into each other."""
    _lane_state["left"] = None
    _lane_state["right"] = None
    _lane_age["left"] = 0
    _lane_age["right"] = 0


def _detect_lanes(frame):
    """Detect and draw averaged/extrapolated lane lines.

    Improvements:
      1. CLAHE preprocessing for better edge detection under varying lighting.
      2. Segments weighted by length so long markings dominate short noise.
      3. Both a lower (0.5) and upper (5.0) slope bound to reject noise.
      4. Temporal EMA smoothing across frames to eliminate flicker.
      5. Fallback to the last valid state when Hough finds nothing.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = _CLAHE.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 150)

    h, w = edges.shape

    # Only look at the road region (a trapezoid in the lower half)
    roi = np.array(
        [[(0, h), (w, h), (int(w * 0.6), int(h * 0.6)), (int(w * 0.4), int(h * 0.6))]],
        dtype=np.int32,
    )
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi, 255)
    edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=100
    )

    if lines is not None:
        left_s, left_i, left_w = [], [], []
        right_s, right_i, right_w = [], [], []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Reject nearly horizontal and near-vertical lines
            if abs(slope) < 0.5 or abs(slope) > 5.0:
                continue

            length = np.hypot(x2 - x1, y2 - y1)

            if slope < 0:
                left_s.append(slope * length)
                left_i.append(intercept * length)
                left_w.append(length)
            else:
                right_s.append(slope * length)
                right_i.append(intercept * length)
                right_w.append(length)

        def _reject_outliers(slopes, intercepts, weights):
            """Drop segments whose plain slope is more than 1 std dev from the median."""
            if len(slopes) < 3:
                return slopes, intercepts, weights  # not enough data to filter
            plain_slopes = [s / w for s, w in zip(slopes, weights)]
            median = np.median(plain_slopes)
            std = np.std(plain_slopes)
            keep = [abs(ps - median) <= std for ps in plain_slopes]
            return (
                [v for v, k in zip(slopes, keep) if k],
                [v for v, k in zip(intercepts, keep) if k],
                [v for v, k in zip(weights, keep) if k],
            )

        def _update(key, slopes, intercepts, weights):
            slopes, intercepts, weights = _reject_outliers(slopes, intercepts, weights)
            if not weights:
                _lane_age[key] = max(0, _lane_age[key] - 1)
                return
            total = np.sum(weights)
            avg_s = np.sum(slopes) / total
            avg_i = np.sum(intercepts) / total
            if _lane_state[key] is None:
                _lane_state[key] = (avg_s, avg_i)
            else:
                prev_s, prev_i = _lane_state[key]
                _lane_state[key] = (
                    _SMOOTH * avg_s + (1 - _SMOOTH) * prev_s,
                    _SMOOTH * avg_i + (1 - _SMOOTH) * prev_i,
                )
            _lane_age[key] = min(_FADE_FRAMES, _lane_age[key] + 1)

        _update("left", left_s, left_i, left_w)
        _update("right", right_s, right_i, right_w)
    else:
        # No lines found at all — age both lanes down
        for key in ("left", "right"):
            _lane_age[key] = max(0, _lane_age[key] - 1)

    # Draw from smoothed state (also serves as fallback when lines is None).
    # Alpha fades from 1.0 (confident) to 0.0 (stale) based on age.
    overlay = np.zeros_like(frame)
    y_bottom = h
    y_top = int(h * 0.6)

    for key in ("left", "right"):
        if _lane_state[key] is None or _lane_age[key] == 0:
            continue
        s, i = _lane_state[key]
        if s == 0:
            continue
        x_bottom = int((y_bottom - i) / s)
        x_top = int((y_top - i) / s)
        cv2.line(overlay, (x_bottom, y_bottom), (x_top, y_top), (255, 0, 0), 4)

    lane_alpha = max(_lane_age["left"], _lane_age["right"]) / _FADE_FRAMES
    return cv2.addWeighted(frame, 0.8, overlay, lane_alpha, 0.0)


# ── Drawing helpers ────────────────────────────────────────────────────────


def _draw_boxes(frame, results, model):
    """Draw bounding boxes and labels on the frame."""
    if results.boxes is None:
        return

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_name = model.names[int(box.cls[0])]

        # Green box
        color = (0, 200, 60)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background + text
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            frame,
            (x1, max(y1 - th - 6, 0)),
            (x1 + tw, max(y1 - th - 6, 0) + th + 6),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 4, th + 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


# ── Public API ─────────────────────────────────────────────────────────────


def process_video(video_path, confidence=0.4, frame_skip=2, apply_hood_mask=True):
    """
    Run YOLOv8 detection on a video.

    Args:
        video_path:  Path to the input video file.
        confidence:  Minimum detection confidence (0-1).
        frame_skip:  Run inference every Nth frame (1 = every frame).
                     Skipped frames reuse the previous detection results.
        apply_hood_mask: Whether to blackout the bottom 20% of the hood region.

    Returns:
        output_path — path to the annotated H.264 video.
    """
    model = get_model()
    _reset_lane_state()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Write to a temp file first (OpenCV uses mp4v codec)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    last_results = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if apply_hood_mask:
            frame = _mask_hood(frame)
        frame = _detect_lanes(frame)

        # Only run inference every Nth frame for speed
        if frame_idx % frame_skip == 0:
            last_results = model(frame, conf=confidence, imgsz=640, verbose=False)[0]

        if last_results is not None:
            _draw_boxes(frame, last_results, model)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # Convert to H.264 so browsers can play it
    return _transcode_to_h264(tmp_path)


def process_image(image_path, confidence=0.4, apply_hood_mask=True):
    """
    Run YOLOv8 detection on a single image.

    Returns:
        output_path — path to the annotated image.
    """
    model = get_model()
    _reset_lane_state()

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Cannot open image: {image_path}")

    if apply_hood_mask:
        frame = _mask_hood(frame)
    frame = _detect_lanes(frame)

    results = model(frame, conf=confidence, imgsz=640, verbose=False)[0]
    _draw_boxes(frame, results, model)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    output_path = tmp.name
    tmp.close()
    cv2.imwrite(output_path, frame)

    return output_path


# ── H.264 transcoding ─────────────────────────────────────────────────────
# OpenCV writes mp4v which most browsers can't play. We convert to H.264.


def _transcode_to_h264(input_path):
    """Transcode a video file to web-safe H.264 using ffmpeg."""
    import subprocess
    import imageio_ffmpeg

    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                input_path,
                "-vcodec",
                "libx264",
                "-crf",
                "28",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                output_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(input_path)
        return output_path
    except Exception as e:
        print(f"FFmpeg transcode failed: {e}")
        return input_path
