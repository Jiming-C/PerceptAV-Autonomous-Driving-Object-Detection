"""Tests for box motion compensation across skipped frames."""

import pytest

from detection.tracking import BoxPredictor, Detection


def car(xyxy, track_id=1, name="car", cls_id=2, conf=0.9):
    return Detection(xyxy=xyxy, cls_id=cls_id, cls_name=name, confidence=conf,
                     track_id=track_id)


def test_detection_area():
    assert car((10, 20, 30, 50)).area == 600


def test_area_of_an_inverted_box_is_zero():
    assert car((30, 50, 10, 20)).area == 0


def test_a_tracked_box_moves_at_its_measured_velocity():
    """The point of the whole module: a skipped frame must not freeze the box."""
    predictor = BoxPredictor(max_coast_frames=6)
    predictor.observe([car((100, 200, 160, 260))], 0)
    predictor.observe([car((130, 200, 190, 260))], 3)  # 10 px/frame

    for frame in (4, 5, 6):
        predicted = predictor.predict(frame, (720, 1280))[0]
        assert predicted.xyxy[0] == pytest.approx(130 + 10 * (frame - 3))
        assert predicted.coasted is True


def test_the_first_observation_has_no_velocity_to_coast_on():
    predictor = BoxPredictor()
    predictor.observe([car((100, 200, 160, 260))], 0)
    assert predictor.predict(1, (720, 1280))[0].xyxy[0] == pytest.approx(100)


def test_velocity_is_per_frame_not_per_inference():
    """Measured across a gap of 5 frames, the box must advance 1 frame's worth."""
    predictor = BoxPredictor()
    predictor.observe([car((100, 100, 200, 200))], 0)
    predictor.observe([car((150, 100, 250, 200))], 5)  # 10 px/frame
    assert predictor.predict(6, (720, 1280))[0].xyxy[0] == pytest.approx(160)


def test_velocity_is_smoothed_against_a_noisy_box():
    predictor = BoxPredictor()
    predictor.observe([car((0, 0, 50, 50))], 0)
    predictor.observe([car((10, 0, 60, 50))], 1)    # 10 px/frame
    predictor.observe([car((110, 0, 160, 50))], 2)  # a 100 px jump
    # Half-weighting the new measurement keeps the prediction sane.
    predicted = predictor.predict(3, (720, 1280))[0].xyxy[0]
    assert 110 < predicted < 210


def test_a_track_is_dropped_once_it_stops_being_seen():
    predictor = BoxPredictor(max_coast_frames=3)
    predictor.observe([car((100, 100, 200, 200))], 0)
    assert predictor.predict(3, (720, 1280))
    assert predictor.predict(4, (720, 1280)) == []


def test_a_stale_track_is_forgotten_on_the_next_observation():
    predictor = BoxPredictor(max_coast_frames=2)
    predictor.observe([car((100, 100, 200, 200), track_id=1)], 0)
    predictor.observe([car((0, 0, 10, 10), track_id=2)], 5)
    assert [d.track_id for d in predictor.predict(6, (720, 1280))] == [2]


def test_coasted_boxes_are_clipped_to_the_frame():
    predictor = BoxPredictor()
    predictor.observe([car((1000, 100, 1100, 200))], 0)
    predictor.observe([car((1200, 100, 1300, 200))], 1)  # heading off the edge
    predicted = predictor.predict(2, (720, 1280))
    assert all(d.xyxy[2] <= 1280 for d in predicted)


def test_a_box_that_coasts_clean_off_the_frame_is_dropped():
    predictor = BoxPredictor(max_coast_frames=10)
    predictor.observe([car((1000, 100, 1100, 200))], 0)
    predictor.observe([car((1260, 100, 1360, 200))], 1)
    assert predictor.predict(5, (720, 1280)) == []


def test_untracked_detections_are_held_in_place():
    """Without an id there is no identity to follow, so holding is the best we can do."""
    predictor = BoxPredictor()
    predictor.observe([car((100, 100, 200, 200), track_id=None)], 0)
    predicted = predictor.predict(1, (720, 1280))
    assert len(predicted) == 1
    assert predicted[0].xyxy == (100, 100, 200, 200)
    assert predicted[0].coasted is True


def test_untracked_detections_do_not_persist_past_the_next_inference():
    predictor = BoxPredictor()
    predictor.observe([car((100, 100, 200, 200), track_id=None)], 0)
    predictor.observe([], 1)
    assert predictor.predict(2, (720, 1280)) == []


def test_observe_returns_its_input_unchanged():
    predictor = BoxPredictor()
    detections = [car((1, 2, 3, 4))]
    assert predictor.observe(detections, 0) is detections


def test_reset_clears_everything():
    predictor = BoxPredictor()
    predictor.observe([car((100, 100, 200, 200))], 0)
    predictor.observe([car((110, 100, 210, 200))], 1)
    predictor.reset()
    assert predictor.predict(2, (720, 1280)) == []


def test_coasted_metadata_survives():
    predictor = BoxPredictor()
    predictor.observe([car((100, 100, 200, 200), track_id=7, name="bus", cls_id=5)], 0)
    predicted = predictor.predict(1, (720, 1280))[0]
    assert (predicted.cls_name, predicted.cls_id, predicted.track_id) == ("bus", 5, 7)
