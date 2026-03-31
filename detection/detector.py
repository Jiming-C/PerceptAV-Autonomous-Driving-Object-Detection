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
    """Black-out the bottom 12% of the frame (the car hood region)."""
    h = frame.shape[0]
    hood_top = int(h * 0.80)
    frame[hood_top:, :] = 0
    return frame


# ── Lane detection ─────────────────────────────────────────────────────────


def _detect_lanes(frame):
    """Detect and draw averaged/extrapolated lane lines.

    Steps:
      1. Convert to grayscale, blur, and run Canny edge detection.
      2. Mask a trapezoidal Region-of-Interest (the road area).
      3. Find line segments with Hough transform.
      4. Separate segments into left vs. right lane by slope.
      5. Average each group and extrapolate into one solid line per side.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        # Group segments by slope: negative slope → left lane, positive → right
        left_slopes, left_intercepts = [], []
        right_slopes, right_intercepts = [], []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue  # skip vertical lines (infinite slope)
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Filter out nearly horizontal lines (|slope| < 0.3)
            if abs(slope) < 0.5:
                continue

            if slope < 0:
                left_slopes.append(slope)
                left_intercepts.append(intercept)
            else:
                right_slopes.append(slope)
                right_intercepts.append(intercept)

        overlay = np.zeros_like(frame)

        # Draw the bottom → middle range for each averaged lane
        y_bottom = h
        y_top = int(h * 0.6)

        # Left lane (average of all left segments)
        if left_slopes:
            avg_slope = np.mean(left_slopes)
            avg_intercept = np.mean(left_intercepts)
            x_bottom = int((y_bottom - avg_intercept) / avg_slope)
            x_top = int((y_top - avg_intercept) / avg_slope)
            cv2.line(overlay, (x_bottom, y_bottom), (x_top, y_top), (255, 0, 0), 4)

        # Right lane (average of all right segments)
        if right_slopes:
            avg_slope = np.mean(right_slopes)
            avg_intercept = np.mean(right_intercepts)
            x_bottom = int((y_bottom - avg_intercept) / avg_slope)
            x_top = int((y_top - avg_intercept) / avg_slope)
            cv2.line(overlay, (x_bottom, y_bottom), (x_top, y_top), (255, 0, 0), 4)

        frame = cv2.addWeighted(frame, 0.8, overlay, 1.0, 0.0)

    return frame


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


def process_video(video_path, confidence=0.4, frame_skip=2):
    """
    Run YOLOv8 detection on a video.

    Args:
        video_path:  Path to the input video file.
        confidence:  Minimum detection confidence (0-1).
        frame_skip:  Run inference every Nth frame (1 = every frame).
                     Skipped frames reuse the previous detection results.

    Returns:
        output_path — path to the annotated H.264 video.
    """
    model = get_model()

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


def process_image(image_path, confidence=0.4):
    """
    Run YOLOv8 detection on a single image.

    Returns:
        output_path — path to the annotated image.
    """
    model = get_model()

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Cannot open image: {image_path}")

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
