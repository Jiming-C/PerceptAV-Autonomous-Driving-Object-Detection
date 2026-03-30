import cv2
import tempfile
import os
from collections import defaultdict
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
_model = None


def get_model() -> YOLO:
    """Load YOLOv8n model once and reuse across calls."""
    global _model
    if _model is None:
        _model = YOLO(MODEL_NAME)
    return _model


def _draw_detections(frame, detections, color=(0, 200, 60)):
    """Draw bounding boxes and labels for a list of cached detections."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    for x1, y1, x2, y2, conf_score, cls_name in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} {conf_score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        label_y1 = max(y1 - text_h - baseline - 4, 0)
        cv2.rectangle(
            frame,
            (x1, label_y1),
            (x1 + text_w, label_y1 + text_h + baseline + 4),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (x1, label_y1 + text_h + 1),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )


def process_video(
    video_path: str, confidence: float, frame_skip: int = 2
) -> tuple[str, str]:
    """
    Run YOLOv8 inference on the input video with frame skipping.

    Args:
        video_path: Path to the uploaded video file.
        confidence: Minimum confidence threshold for detections.
        frame_skip: Process every Nth frame (1 = all frames, 2 = every other, etc.).

    Returns:
        A tuple of (annotated_video_path, detection_summary_text).
    """
    model = get_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detection_counts: dict[str, int] = defaultdict(int)
    last_detections: list[tuple] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            results = model(frame, conf=confidence, imgsz=640, verbose=False)[0]

            last_detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_score = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                detection_counts[cls_name] += 1
                last_detections.append((x1, y1, x2, y2, conf_score, cls_name))

        _draw_detections(frame, last_detections)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    summary = _build_summary(detection_counts)
    return output_path, summary


def _build_summary(counts: dict[str, int]) -> str:
    """Format per-class detection totals into a readable summary string."""
    if not counts:
        return "No objects detected."

    total = sum(counts.values())
    lines = [f"Total detections: {total}\n"]
    for cls_name, count in sorted(counts.items(), key=lambda x: -x[1]):
        label = "detection" if count == 1 else "detections"
        lines.append(f"  {cls_name}: {count} {label}")

    return "\n".join(lines)
