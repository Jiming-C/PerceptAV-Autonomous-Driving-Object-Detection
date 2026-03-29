import gradio as gr
from detection.detector import process_video


def run_detection(video_path: str | None, confidence: float):
    if video_path is None:
        return None, "Please upload a video file."
    annotated_path, summary = process_video(video_path, confidence)
    return annotated_path, summary


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="YOLOv8 Video Object Detection") as demo:
    gr.Markdown(
        """
# YOLOv8 Video Object Detection
Upload a video clip, set a confidence threshold, and click **Detect Objects**.
Annotated frames are written to an output video — bounding boxes include the
class label and confidence score for every detection.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(
                label="Input Video",
                sources=["upload"],
            )
            confidence_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.4,
                step=0.05,
                label="Confidence Threshold",
                info="Detections below this score are ignored.",
            )
            detect_btn = gr.Button("Detect Objects", variant="primary")

        with gr.Column(scale=1):
            video_output = gr.Video(
                label="Annotated Video",
                interactive=False,
            )
            summary_output = gr.Textbox(
                label="Detection Summary",
                lines=12,
                interactive=False,
                placeholder="Detection counts will appear here after processing.",
            )

    detect_btn.click(
        fn=run_detection,
        inputs=[video_input, confidence_slider],
        outputs=[video_output, summary_output],
    )

    # -----------------------------------------------------------------------
    # Examples — replace the placeholder path with a real sample video to
    # enable one-click examples on Hugging Face Spaces.
    # -----------------------------------------------------------------------
    gr.Examples(
        examples=[
            # ["examples/sample.mp4", 0.4],  # add a sample video to enable
        ],
        inputs=[video_input, confidence_slider],
        outputs=[video_output, summary_output],
        fn=run_detection,
        cache_examples=False,
        label="Examples (add sample videos to the examples/ folder to enable)",
    )


if __name__ == "__main__":
    demo.launch()
