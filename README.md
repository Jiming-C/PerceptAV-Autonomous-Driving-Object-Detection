---
title: YOLOv8 Video Object Detection
emoji: 🎯
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.0
app_file: app.py
pinned: false
license: apache-2.0
---

# YOLOv8 Video Object Detection

A Gradio app that runs **YOLOv8n** inference on every frame of an uploaded video,
draws annotated bounding boxes, and outputs a detection summary.

## How it works

1. Upload a video file (MP4, AVI, MOV, etc.)
2. Adjust the **confidence threshold** (default 0.4)
3. Click **Detect Objects**
4. Download or preview the annotated video
5. Read the per-class detection summary

## Stack

| Component | Library |
|-----------|---------|
| Object detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (`yolov8n.pt`) |
| Frame processing | OpenCV (`opencv-python-headless`) |
| UI | [Gradio](https://gradio.app) ≥ 4.0 |

## Project structure

```
.
├── app.py                  # Gradio interface and entry point
├── detection/
│   ├── __init__.py
│   └── detector.py         # Model loading and per-frame inference
├── requirements.txt
└── README.md
```

## Adding example videos

Place sample video files in an `examples/` folder and uncomment the entry
in the `gr.Examples` section of `app.py`:

```python
gr.Examples(
    examples=[
        ["examples/sample.mp4", 0.4],
    ],
    ...
)
```

## Local development

```bash
pip install -r requirements.txt
python app.py
```

The YOLOv8n weights (`yolov8n.pt`, ~6 MB) are downloaded automatically on
first run by the Ultralytics library.
