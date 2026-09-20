"""Tests for the pipeline's plumbing.

Anything needing YOLO weights is skipped unless PERCEPTAV_TEST_MODEL points at
a .pt file, so the suite stays runnable without a 6 MB download.
"""

import dataclasses
import os

import cv2
import numpy as np
import pytest
from synthetic_road import render

from detection.config import (
    CLASS_COLORS,
    CONTROL_COLOR,
    VEHICLE_COLOR,
    VRU_COLOR,
    DetectorConfig,
    LaneConfig,
    PipelineConfig,
)
from detection.detector import (
    RunSummary,
    _build_config,
    _draw_boxes,
    _LaneStage,
    _ObjectCounter,
    process_image,
    process_video,
)
from detection.tracking import Detection

MODEL = os.environ.get("PERCEPTAV_TEST_MODEL")
needs_model = pytest.mark.skipif(
    not (MODEL and os.path.exists(MODEL)),
    reason="set PERCEPTAV_TEST_MODEL to a YOLO .pt file to run end-to-end tests",
)


def det(name="car", cls_id=2, track_id=None, coasted=False, conf=0.9):
    return Detection((10.0, 10.0, 60.0, 60.0), cls_id, name, conf, track_id, coasted)


# ── Config plumbing ────────────────────────────────────────────────────────


def test_build_config_threads_the_ui_arguments_through():
    cfg = _build_config(0.55, 3, False, "multi", None)
    assert cfg.detector.confidence == 0.55
    assert cfg.frame_skip == 3
    assert cfg.apply_hood_mask is False
    assert cfg.lane_mode == "multi"


def test_build_config_keeps_the_rest_of_a_supplied_config():
    base = PipelineConfig(
        hood_fraction=0.7,
        detector=DetectorConfig(model_name="custom.pt", imgsz=960),
    )
    cfg = _build_config(0.4, 1, True, "ego", base)
    assert cfg.hood_fraction == 0.7
    assert cfg.detector.model_name == "custom.pt"
    assert cfg.detector.imgsz == 960


def test_build_config_clamps_a_nonsense_frame_skip():
    assert _build_config(0.4, 0, True, "ego", None).frame_skip == 1
    assert _build_config(0.4, -5, True, "ego", None).frame_skip == 1


def test_pipeline_config_rejects_an_unknown_lane_mode():
    with pytest.raises(ValueError):
        PipelineConfig(lane_mode="sideways")


def test_pipeline_config_rejects_a_bad_hood_fraction():
    with pytest.raises(ValueError):
        PipelineConfig(hood_fraction=0)


# ── Colours ────────────────────────────────────────────────────────────────


def test_road_users_are_coloured_by_what_they_mean_to_a_driver():
    assert CLASS_COLORS[0] == VRU_COLOR        # person
    assert CLASS_COLORS[1] == VRU_COLOR        # bicycle
    assert CLASS_COLORS[2] == VEHICLE_COLOR    # car
    assert CLASS_COLORS[7] == VEHICLE_COLOR    # truck
    assert CLASS_COLORS[9] == CONTROL_COLOR    # traffic light


def test_every_detected_class_has_a_colour():
    assert set(DetectorConfig().classes) <= set(CLASS_COLORS)


# ── Drawing ────────────────────────────────────────────────────────────────


def test_boxes_are_drawn_in_their_class_colour():
    frame = np.zeros((200, 200, 3), np.uint8)
    _draw_boxes(frame, [det("person", cls_id=0)])
    assert (frame == np.array(VRU_COLOR)).all(axis=2).any()


def test_a_coasted_box_is_drawn_thinner_than_an_observed_one():
    observed, coasted = np.zeros((200, 200, 3), np.uint8), np.zeros((200, 200, 3), np.uint8)
    _draw_boxes(observed, [det()])
    _draw_boxes(coasted, [det(coasted=True)])
    assert np.count_nonzero(coasted) < np.count_nonzero(observed)


def test_a_box_at_the_top_edge_keeps_its_label_on_screen():
    frame = np.zeros((200, 200, 3), np.uint8)
    top_box = dataclasses.replace(det(), xyxy=(10.0, 0.0, 60.0, 40.0))
    _draw_boxes(frame, [top_box])  # must not raise
    assert np.count_nonzero(frame[:20]) > 0


def test_track_ids_can_be_turned_off():
    with_id, without = np.zeros((200, 200, 3), np.uint8), np.zeros((200, 200, 3), np.uint8)
    _draw_boxes(with_id, [det(track_id=12)], show_track_ids=True)
    _draw_boxes(without, [det(track_id=12)], show_track_ids=False)
    assert np.count_nonzero(with_id) != np.count_nonzero(without)


# ── Counting ───────────────────────────────────────────────────────────────


def test_tracked_objects_are_counted_once_each():
    counter = _ObjectCounter()
    for _ in range(10):
        counter.add([det(track_id=1), det(track_id=2)])
    objects, tracked = counter.result()
    assert objects == {"car": 2}
    assert tracked is True


def test_coasted_boxes_are_not_counted_as_new_observations():
    counter = _ObjectCounter()
    counter.add([det(track_id=1)])
    counter.add([det(track_id=2, coasted=True)])
    assert counter.result()[0] == {"car": 1}


def test_untracked_objects_fall_back_to_a_peak_count():
    counter = _ObjectCounter()
    counter.add([det(), det(), det()])
    counter.add([det()])
    objects, tracked = counter.result()
    assert objects == {"car": 3}
    assert tracked is False


def test_counting_nothing_is_empty():
    assert _ObjectCounter().result() == ({}, False)


# ── Summary ────────────────────────────────────────────────────────────────


def test_summary_reports_fps():
    assert RunSummary(frames=60, seconds=2.0).fps == pytest.approx(30.0)


def test_summary_fps_of_an_empty_run_is_zero_not_a_crash():
    assert RunSummary().fps == 0.0


def test_summary_markdown_explains_skipped_frames():
    text = RunSummary(frames=60, inference_frames=15, seconds=2.0,
                      resolution=(1280, 720)).to_markdown()
    assert "45" in text and "velocity" in text


def test_summary_markdown_omits_the_skip_note_when_nothing_was_skipped():
    text = RunSummary(frames=60, inference_frames=60, seconds=2.0).to_markdown()
    assert "velocity" not in text


def test_summary_markdown_says_so_when_nothing_was_found():
    assert "No objects" in RunSummary(frames=10, seconds=1.0).to_markdown()


def test_summary_markdown_lists_classes_by_count():
    text = RunSummary(frames=1, seconds=1.0, tracked=True,
                      objects={"car": 2, "person": 9}).to_markdown()
    assert text.index("person") < text.index("car")


# ── Lane stage ─────────────────────────────────────────────────────────────


def test_lane_stage_off_does_nothing():
    frame, _ = render(seed=0)
    assert _LaneStage("off", LaneConfig(), None)(frame) == []


def test_lane_stage_ego_finds_the_pair():
    frame, scene = render(seed=0, hood_fraction=0.80)
    hood = int(scene.height * 0.80)
    stage = _LaneStage("ego", LaneConfig(), hood)
    assert len(stage(frame)) == 2


def test_lane_stage_keeps_its_own_state():
    """Two stages must not share temporal state, or concurrent runs interfere."""
    frame, scene = render(seed=0, hood_fraction=0.80)
    hood = int(scene.height * 0.80)
    a, b = _LaneStage("ego", LaneConfig(), hood), _LaneStage("ego", LaneConfig(), hood)
    for _ in range(5):
        a(frame)
    assert a.tracker.lines[0] is not None
    assert b.tracker.lines[0] is None


# ── End to end ─────────────────────────────────────────────────────────────


@pytest.fixture
def clip(tmp_path):
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(12):
        frame, _ = render(seed=i, hood_fraction=0.80, difficulty=0.5)
        writer.write(frame)
    writer.release()
    return path


def _cfg():
    return PipelineConfig(detector=DetectorConfig(model_name=MODEL))


@needs_model
def test_process_video_returns_a_playable_file_and_a_summary(clip):
    out, summary = process_video(clip, confidence=0.4, frame_skip=1, config=_cfg())
    assert os.path.getsize(out) > 0
    assert summary.frames == 12
    assert summary.inference_frames == 12
    assert summary.resolution == (1280, 720)
    cap = cv2.VideoCapture(out)
    assert cap.isOpened() and cap.read()[0]
    cap.release()


@needs_model
def test_frame_skip_reduces_inference_without_dropping_frames(clip):
    _, summary = process_video(clip, frame_skip=4, config=_cfg())
    assert summary.frames == 12
    assert summary.inference_frames == 3


@needs_model
def test_every_lane_mode_runs(clip):
    for mode in ("ego", "multi", "off"):
        _, summary = process_video(clip, lane_mode=mode, config=_cfg())
        assert summary.frames == 12
        assert (summary.lane_frames > 0) == (mode != "off")


@needs_model
def test_process_image_returns_a_summary(tmp_path):
    frame, _ = render(seed=0, hood_fraction=0.80)
    path = str(tmp_path / "frame.png")
    cv2.imwrite(path, frame)
    out, summary = process_image(path, config=_cfg())
    assert os.path.exists(out)
    assert summary.frames == 1
    assert cv2.imread(out) is not None


def test_a_missing_video_is_a_clear_error():
    with pytest.raises(ValueError, match="Cannot open video"):
        process_video("/nonexistent/nope.mp4")


def test_a_missing_image_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="Cannot open image"):
        process_image(str(tmp_path / "nope.jpg"), config=PipelineConfig())
