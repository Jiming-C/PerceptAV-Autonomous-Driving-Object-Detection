"""
app.py — Gradio web interface for the AV Object Detection demo.

Run with:  python app.py
"""

import gradio as gr
from detection.detector import process_video, process_image


# ── Callback functions ─────────────────────────────────────────────────────


def run_video(video_path, confidence, frame_skip, apply_hood_mask):
    """Process a dashcam video and return the annotated result."""
    if video_path is None:
        return None
    try:
        return process_video(video_path, confidence, int(frame_skip), apply_hood_mask)
    except Exception as e:
        raise gr.Error(str(e))


def run_image(image_path, confidence, apply_hood_mask):
    """Process a dashcam image and return the annotated result."""
    if image_path is None:
        return None
    try:
        return process_image(image_path, confidence, apply_hood_mask)
    except Exception as e:
        raise gr.Error(str(e))


# ── UI ─────────────────────────────────────────────────────────────────────

with gr.Blocks(title="AV Object Detection") as demo:
    gr.Markdown(
        "# 🚘 Autonomous Driving Object Detection\n"
        "Detect vehicles, pedestrians, and lane lines on dashcam footage "
        "using YOLOv8."
    )

    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column():
                vid_in = gr.Video(label="Input Video", sources=["upload"])
                vid_conf = gr.Slider(
                    0.1,
                    0.9,
                    value=0.4,
                    step=0.05,
                    label="Confidence Threshold",
                )
                vid_skip = gr.Slider(
                    1,
                    5,
                    value=2,
                    step=1,
                    label="Frame Skip",
                    info="Process every Nth frame. Higher = faster.",
                )
                vid_mask = gr.Checkbox(value=True, label="Apply Hood Mask - black box on the bottom to filter out the hood")
                vid_btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                vid_out = gr.Video(label="Annotated Video", interactive=False)

        vid_btn.click(run_video, [vid_in, vid_conf, vid_skip, vid_mask], [vid_out])

        # Pre-processed demo — click to see results instantly
        gr.Markdown("### 👇 Try the demo — click below for instant results")
        gr.Examples(
            examples=[["examples/demo.MP4", 0.3, 1, True]],
            inputs=[vid_in, vid_conf, vid_skip, vid_mask],
            outputs=[vid_out],
            fn=run_video,
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
                    0.1,
                    0.9,
                    value=0.4,
                    step=0.05,
                    label="Confidence Threshold",
                )
                img_mask = gr.Checkbox(value=True, label="Apply Hood Mask")
                img_btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                img_out = gr.Image(
                    label="Annotated Image", type="filepath", interactive=False
                )

        img_btn.click(run_image, [img_in, img_conf, img_mask], [img_out])

if __name__ == "__main__":
    demo.launch()
