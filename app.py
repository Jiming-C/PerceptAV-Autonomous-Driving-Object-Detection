"""
app.py — Gradio web interface for the AV perception demo.

Run with:  python app.py
"""

import gradio as gr

from detection.detector import process_image, process_video

LANE_MODES = {
    "Ego lane": "ego",
    "All lanes": "multi",
    "Off": "off",
}

LANE_MODE_INFO = (
    "**Ego lane** finds the two lines either side of this car. "
    "**All lanes** finds every line in the frame, however many that is."
)


# ── Callback functions ─────────────────────────────────────────────────────


def run_video(video_path, confidence, frame_skip, apply_hood_mask, lane_mode,
              progress=gr.Progress()):
    """Process a dashcam video and return the annotated result plus a summary."""
    if video_path is None:
        return None, ""
    try:
        return process_video(
            video_path,
            confidence=confidence,
            frame_skip=int(frame_skip),
            apply_hood_mask=apply_hood_mask,
            lane_mode=LANE_MODES.get(lane_mode, "ego"),
            progress=lambda fraction, message: progress(fraction, desc=message),
        )
    except Exception as exc:
        raise gr.Error(str(exc))


def run_image(image_path, confidence, apply_hood_mask, lane_mode):
    """Process a dashcam image and return the annotated result plus a summary."""
    if image_path is None:
        return None, ""
    try:
        return process_image(
            image_path,
            confidence=confidence,
            apply_hood_mask=apply_hood_mask,
            lane_mode=LANE_MODES.get(lane_mode, "ego"),
        )
    except Exception as exc:
        raise gr.Error(str(exc))


def _unpack(result):
    """Gradio wants a tuple of outputs; the pipeline returns (path, summary)."""
    path, summary = result
    return path, summary.to_markdown()


def video_callback(*args, **kwargs):
    result = run_video(*args, **kwargs)
    return result if result[0] is None else _unpack(result)


def image_callback(*args):
    result = run_image(*args)
    return result if result[0] is None else _unpack(result)


# ── UI ─────────────────────────────────────────────────────────────────────

with gr.Blocks(title="AV Object Detection") as demo:
    gr.Markdown(
        "# 🚘 Autonomous Driving Object Detection\n"
        "Detect vehicles, pedestrians, and lane lines on dashcam footage "
        "using YOLOv8 with ByteTrack, plus a classical lane pipeline."
    )

    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column():
                vid_in = gr.Video(label="Input Video", sources=["upload"])
                vid_conf = gr.Slider(
                    0.1, 0.9, value=0.4, step=0.05, label="Confidence Threshold"
                )
                vid_skip = gr.Slider(
                    1,
                    5,
                    value=2,
                    step=1,
                    label="Frame Skip",
                    info="Run inference every Nth frame. Skipped frames keep "
                    "tracked boxes moving at their measured velocity, so higher "
                    "is faster at some cost in tracking continuity.",
                )
                vid_lane = gr.Radio(
                    list(LANE_MODES),
                    value="Ego lane",
                    label="Lane Detection",
                    info=LANE_MODE_INFO,
                )
                vid_mask = gr.Checkbox(
                    value=True,
                    label="Apply Hood Mask — black box on the bottom to filter "
                    "out the hood",
                )
                vid_btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                vid_out = gr.Video(label="Annotated Video", interactive=False)
                vid_summary = gr.Markdown(label="Run Summary")

        vid_inputs = [vid_in, vid_conf, vid_skip, vid_mask, vid_lane]
        vid_outputs = [vid_out, vid_summary]
        vid_btn.click(video_callback, vid_inputs, vid_outputs)

        # Pre-processed demo — click to see results instantly
        gr.Markdown("### 👇 Try the demo — click below for instant results")
        gr.Examples(
            examples=[["examples/demo.MP4", 0.3, 1, True, "Ego lane"]],
            inputs=vid_inputs,
            outputs=vid_outputs,
            fn=video_callback,
            cache_examples=True,
            label="Demo Dashcam Video",
        )

    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(
                    label="Input Image", type="filepath", sources=["upload"]
                )
                img_conf = gr.Slider(
                    0.1, 0.9, value=0.4, step=0.05, label="Confidence Threshold"
                )
                img_lane = gr.Radio(
                    list(LANE_MODES),
                    value="Ego lane",
                    label="Lane Detection",
                    info=LANE_MODE_INFO,
                )
                img_mask = gr.Checkbox(value=True, label="Apply Hood Mask")
                img_btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                img_out = gr.Image(
                    label="Annotated Image", type="filepath", interactive=False
                )
                img_summary = gr.Markdown(label="Run Summary")

        img_btn.click(
            image_callback,
            [img_in, img_conf, img_mask, img_lane],
            [img_out, img_summary],
        )

if __name__ == "__main__":
    demo.launch()
