

# 🚘 Autonomous Driving Object Detection

A dashcam perception pipeline combining YOLOv8 object detection and classical computer vision for real-time vehicle, pedestrian, and lane identification. Built as a clean implementation of core AV perception techniques — hood masking, Hough transform lane detection, multi-object tracking, and inference optimization via frame skipping.
<img width="1494" height="429" alt="image" src="https://github.com/user-attachments/assets/06980589-f072-47e7-b3c5-d497c5379ae9" />

---

## Demo

### Object Detection + Lane Identification
<img src="tmp48pugu2z-ezgif.com-optimize.gif" alt="Object detection and lane identification demo" width="100%" />

### Real-World Dashcam Footage (720p, CPU, frame skip=2)
<img src="tmplvkxmjyu-ezgif.com-optimize.gif" alt="Real dashcam footage detection demo" width="100%" />

---

## Features

| Feature | Description |
|---|---|
| 🟢 **Object Detection** | Detects cars, trucks, buses, pedestrians, cyclists, traffic lights and stop signs using YOLOv8 Nano |
| 🔗 **Object Tracking** | ByteTrack assigns each object a persistent id, so the run summary counts *distinct* objects rather than boxes drawn |
| 🔵 **Lane Identification** | Finds lane lines using CLAHE + Canny + Hough transforms, with temporal smoothing and vanishing-point convergence |
| 🛣️ **Multi-Lane Mode** | Optionally finds every lane line in the frame, not just the two around this car |
| ⚡ **Frame Skipping** | Skipped frames advance each tracked box at its measured velocity instead of freezing it in place |
| 🎭 **Hood Masking** | Blacks out the car hood before anything reads the frame, so it can't produce false detections or false edges |
| 📊 **Measured, Not Guessed** | A benchmark suite scores the lane pipeline against known geometry — `python scripts/evaluate_lanes.py` |
| 🌐 **Web Interface** | Gradio UI with a pre-processed instant demo video and a per-run summary |

---

## Tech Stack

- **[YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)** — Real-time object detection and ByteTrack multi-object tracking
- **[OpenCV](https://opencv.org/)** — Classical computer vision (CLAHE, Canny, Hough transforms)
- **[Gradio](https://www.gradio.app/)** — Web interface
- **[FFmpeg](https://ffmpeg.org/)** — H.264 video transcoding for browser compatibility

---

## How It Works

Every frame goes through this pipeline:

```
                 Input Frame
                      │
                      ▼
              1. Hood Mask                  Black out the car's own bonnet,
                      │                     before anything else reads the frame
        ┌─────────────┴─────────────┐
        ▼                           ▼
2. Object Detection          3. Lane Detection
   YOLOv8 + ByteTrack           CLAHE → Canny → ROI → Hough
        │                           → innermost-cluster select
        │                           → temporal EMA → vanishing-point clamp
        └─────────────┬─────────────┘
                      ▼
              4. Compositing               Lane overlay, then boxes
                      │
                      ▼
              5. H.264 Encode              OpenCV's default output isn't
                                           browser-playable; ffmpeg fixes that
```

**Both stages read the same clean frame.** This matters: an earlier version drew the lane overlay *first* and dimmed the whole image by 20% doing it, then handed that darkened, line-painted frame to YOLO. The detector was being asked to find cars in a picture that had already been drawn on.

### Lane Detection Deep Dive
1. Grayscale, then **CLAHE** local contrast equalisation — recovers worn paint in shadow without blowing out the sky
2. Gaussian blur, then **Canny** edge detection
3. Mask a trapezoidal **region of interest** — bounded above by the horizon and below by the hood line, so the hood mask's own hard edge never enters the transform
4. **`HoughLinesP`** to find line segments
5. **Slope filter** — rejects near-horizontal (shadows, tar seams) and near-vertical (guardrail posts, sign poles) segments
6. **Cluster by position** — segments are projected onto a shared reference row and grouped, so fragments of one marking found at different heights still group together
7. **Keep the innermost cluster per side** — the lane line nearest this car, rather than an average of every marking on that side (see below)
8. **Length-weighted average** with **median-absolute-deviation** outlier rejection
9. **Temporal EMA** with per-lane confidence, gating against impossible jumps, and fade-out when a lane is lost
10. **Vanishing-point clamp** — both lines stop where they converge, so they meet rather than crossing into an X

---

## Measured

The lane pipeline is scored against procedurally generated road frames whose geometry is exactly known — perspective convergence, dashed markings, asphalt texture, shadows, guardrails, hood occlusion. They are a weaker proxy than real footage, but they are deterministic, they run in a second, and they make "this change is an improvement" a claim you can check.

```bash
python scripts/evaluate_lanes.py                      # 100 frames, typical conditions
python scripts/evaluate_lanes.py --frames 300 --difficulty 2.0
python scripts/evaluate_lanes.py --suite temporal     # does smoothing actually help?
python scripts/evaluate_lanes.py --set canny_low=30   # sweep any LaneConfig field
```

A frame **passes** when both lane lines are found, they converge at or above the ROI ceiling (the geometric plausibility check borrowed from [xpanvictor/road-computer-vision-engr](https://github.com/xpanvictor/road-computer-vision-engr)), and every probe row is within 2.5% of the frame width of ground truth. 300 frames per row:

| Conditions | Pass rate | Mean x-error | Median x-error |
|---|---|---|---|
| Clean (difficulty 0.5) | **95.7%** | 4.4 px | 1.4 px |
| Typical (difficulty 1.0) | **95.0%** | 5.1 px | 1.6 px |
| Unkind (difficulty 2.0) | **94.0%** | 5.5 px | 1.4 px |

### What actually moved the number

Almost all of it came from **one algorithmic change, not from parameter tuning**. Averaging every same-signed Hough segment into a single "right lane" quietly blends in the *next lane over* whenever a second marking is visible, landing the reported lane somewhere between the two. Clustering first and keeping the cluster nearest the car fixes it:

| | Pass rate | Mean x-error |
|---|---|---|
| Average every segment on each side (`ego_select_innermost=False`) | 28.7% | 36.4 px |
| Keep the cluster nearest the car (default) | **95.0%** | **5.1 px** |

Reproduce with `python scripts/evaluate_lanes.py --frames 300 --set ego_select_innermost=False`.

Parameter sweeps, by contrast, found nothing worth changing — every alternative that looked better on 120 frames either fell within noise or collapsed on harder scenes at 300. **The Canny, Hough and slope thresholds are unchanged from the original hand-tuned values.** That is a result too, and the harness is what turned "these look fine" into something checkable.

Temporal smoothing, measured on a static scene where any output movement is the detector being unsteady (`--suite temporal`):

| | Frame-to-frame jitter |
|---|---|
| Raw per-frame detections | 3.0 px |
| With `EgoLaneTracker` | **0.3 px** |

### Running against TuSimple

The synthetic suite is a proxy. For the real benchmark, download the [TuSimple lane detection dataset](https://www.kaggle.com/datasets/manideep1108/tusimple) into `data/TUSimple/` and run:

```bash
python scripts/evaluate_lanes.py --suite tusimple --data-root data/TUSimple/train_set --sample 50
```

---

## Project Structure

```
PerceptAV-Autonomous-Driving-Object-Detection/
├── app.py                      # Gradio web interface
├── detection/
│   ├── __init__.py
│   ├── config.py               # every tuneable parameter, in one place
│   ├── lanes.py                # classical lane pipeline (stateless) + trackers
│   ├── tracking.py             # motion compensation across skipped frames
│   └── detector.py             # end-to-end pipeline, YOLO stage, encoding
├── scripts/
│   ├── synthetic_road.py       # road frames with exactly known geometry
│   └── evaluate_lanes.py       # benchmark suites
├── tests/                      # pytest suite
├── examples/
│   └── demo.MP4                # pre-loaded dashcam demo video
├── requirements.txt
├── yolov8n.pt                  # YOLOv8 Nano weights (auto-downloaded)
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- macOS / Linux

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Jiming-C/PerceptAV-Autonomous-Driving-Object-Detection.git
cd PerceptAV-Autonomous-Driving-Object-Detection

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Open **http://127.0.0.1:7860** in your browser. The demo video is pre-loaded — just click it for instant results.

### Using it as a library

```python
from detection import process_video

output_path, summary = process_video(
    "dashcam.mp4",
    confidence=0.4,
    frame_skip=2,
    lane_mode="ego",      # or "multi", or "off"
)
print(summary.to_markdown())
```

### Tests

```bash
pip install pytest
pytest                                                  # skips the model-dependent tests
PERCEPTAV_TEST_MODEL=yolov8n.pt pytest                  # runs everything
```

---

## Configuration

Every tuneable parameter lives in `detection/config.py`, as three dataclasses. Each value carries the reasoning for why it is what it is.

| Parameter | Config | Default | Description |
|---|---|---|---|
| `hood_fraction` | `PipelineConfig` | `0.80` | Black out the bottom N% of the frame |
| `lane_mode` | `PipelineConfig` | `"ego"` | `"ego"`, `"multi"` or `"off"` |
| `frame_skip` | `PipelineConfig` | `2` | Run inference every Nth frame |
| `canny_low` / `canny_high` | `LaneConfig` | `45` / `150` | Canny thresholds |
| `roi_y_top_frac` | `LaneConfig` | `0.60` | Top of the region of interest |
| `slope_min_abs` / `slope_max_abs` | `LaneConfig` | `0.5` / `5.0` | Slope band a segment must fall in |
| `ego_select_innermost` | `LaneConfig` | `True` | Keep the lane nearest the car instead of averaging every marking |
| `smoothing` | `LaneConfig` | `0.15` | Temporal EMA rate — lower is steadier |
| `fade_frames` | `LaneConfig` | `15` | Frames a lost lane takes to fade out |
| `confidence` | `DetectorConfig` | `0.4` | YOLOv8 detection confidence |
| `classes` | `DetectorConfig` | 9 COCO ids | Which classes to detect at all |
| `track` | `DetectorConfig` | `True` | ByteTrack persistent object ids |
| `max_coast_frames` | `DetectorConfig` | `6` | How long a box may coast unobserved |

Hough parameters are stored at their 720p values and rescaled to the input's resolution automatically, so a config tuned once works on 480p phone clips and 4K dashcams alike.

```python
import dataclasses
from detection import PipelineConfig, LaneConfig, process_video

config = PipelineConfig(
    hood_fraction=0.72,
    lane=dataclasses.replace(LaneConfig(), smoothing=0.3),
)
output, summary = process_video("dashcam.mp4", config=config)
```

---

## Known limitations

- The lane stage fits **straight lines**. It approximates a gentle curve acceptably and degrades on a tight one; a polynomial or spline fit is the natural next step.
- Lane output is in **image-pixel coordinates**. Projecting it into a vehicle-relative frame needs camera calibration and inverse perspective mapping, so it is not planner-ready as it stands.
- The headline numbers above are from the **synthetic suite**, which has clean paint and no rain, night, or non-US markings. Treat them as a regression guard, not as a claim about real-world accuracy.
- **Frame skipping costs tracking continuity.** Motion compensation keeps the boxes in the right place, but ByteTrack sees fewer frames and re-assigns ids more often, so distinct-object counts drift upward at high skip rates.
- Multi-lane mode is temporally smoothed but does not label lanes by **position relative to the car** (ego-left, ego-right, next-over).

---

## Credits

The classical lane work here builds on [xpanvictor/road-computer-vision-engr](https://github.com/xpanvictor/road-computer-vision-engr), which contributed the vanishing-point convergence clamp, the position-based clustering that makes multi-lane detection possible, the geometric validity metric this repo's benchmark is built around, and the general discipline of validating each stage against ground truth rather than eyeballing the overlay.
