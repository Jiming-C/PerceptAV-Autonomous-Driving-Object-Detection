"""Tests for the classical lane pipeline.

These lean on exact geometry wherever possible: a synthetic frame whose lane
lines were drawn from a known (slope, intercept) can be scored against that
ground truth rather than against whatever the detector happened to produce the
day the test was written.
"""

import numpy as np
import pytest
from synthetic_road import render

from detection.config import REFERENCE_HEIGHT, LaneConfig
from detection.lanes import (
    Drawable,
    EgoLaneTracker,
    LaneFit,
    LaneLine,
    MultiLaneTracker,
    clamp_to_vanishing_point,
    cluster_by_position,
    detect_ego_lanes,
    detect_multilanes,
    draw_lanes,
    endpoints,
    mask_hood,
    vanishing_y,
)

# ── Geometry ───────────────────────────────────────────────────────────────


def test_x_at_inverts_the_line_equation():
    line = LaneLine(slope=-1.5, intercept=900.0)
    assert line.x_at(720) == pytest.approx((720 - 900) / -1.5)


def test_x_at_rejects_a_horizontal_line():
    with pytest.raises(ZeroDivisionError):
        LaneLine(slope=0.0, intercept=10.0).x_at(100)


def test_vanishing_y_finds_the_meeting_row():
    left = LaneLine(-1.0, 1240.0)
    right = LaneLine(1.0, -40.0)
    assert vanishing_y(left, right) == pytest.approx(600.0)


def test_parallel_lines_have_no_vanishing_point():
    assert vanishing_y(LaneLine(-1.0, 0.0), LaneLine(-1.0, 50.0)) is None


def test_clamp_stops_lines_where_they_meet():
    """The X-shaped-overlay bug: lines meeting below the ROI ceiling.

    Without the clamp each line continues past the meeting point and out the
    far side of the other, so the left line ends up to the right of the right
    one. With it, both stop exactly where they converge.
    """
    left, right = LaneLine(-1.0, 1240.0), LaneLine(1.0, -40.0)  # meet at y=600
    y_top, y_bottom = 432.0, 720.0

    naive_left = endpoints(left, y_bottom, y_top)
    naive_right = endpoints(right, y_bottom, y_top)
    assert naive_left.x2 > naive_right.x2  # crossed over

    y_stop = clamp_to_vanishing_point(left, right, y_bottom, y_top)
    assert y_stop == pytest.approx(600.0)
    assert endpoints(left, y_bottom, y_stop).x2 == endpoints(right, y_bottom, y_stop).x2


def test_clamp_leaves_lines_alone_when_they_meet_above_the_ceiling():
    left, right = LaneLine(-1.0, 940.0), LaneLine(1.0, -340.0)  # meet at y=300
    assert clamp_to_vanishing_point(left, right, 720, 432) == 432


def test_clamp_needs_both_lines():
    assert clamp_to_vanishing_point(LaneLine(-1.0, 940.0), None, 720, 432) == 432
    assert clamp_to_vanishing_point(None, None, 720, 432) == 432


# ── Hood mask ──────────────────────────────────────────────────────────────


def test_mask_hood_does_not_mutate_its_input():
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    masked = mask_hood(frame, 0.8)
    assert masked[90, 50].tolist() == [0, 0, 0]
    assert frame[90, 50].tolist() == [200, 200, 200], "input frame was modified"


def test_mask_hood_leaves_the_road_alone():
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    assert mask_hood(frame, 0.8)[79, 50].tolist() == [200, 200, 200]


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_mask_hood_rejects_a_nonsense_fraction(bad):
    with pytest.raises(ValueError):
        mask_hood(np.zeros((10, 10, 3), np.uint8), bad)


# ── Clustering ─────────────────────────────────────────────────────────────


def _seg(slope, intercept):
    return (slope, intercept, 50.0, 0.0, 100.0)


def test_clustering_separates_two_markings():
    near = _seg(1.0, -40.0)     # crosses y=720 at x=760
    far = _seg(1.0, -400.0)     # crosses y=720 at x=1120
    clusters = cluster_by_position([near, far], reference_y=720, gap_px=150)
    assert len(clusters) == 2


def test_clustering_merges_fragments_of_one_marking():
    fragments = [_seg(1.0, -40.0), _seg(1.02, -60.0), _seg(0.98, -20.0)]
    assert len(cluster_by_position(fragments, 720, 150)) == 1


def test_clustering_compares_at_a_shared_row_not_raw_x():
    """Two fragments of one line, found at opposite ends of the frame.

    Their raw coordinates are far apart; extrapolated to the same row they
    coincide, which is the whole point of projecting before grouping.
    """
    line = LaneLine(1.0, -40.0)
    low = (line.slope, line.intercept, 40.0, 700.0, 720.0)
    high = (line.slope, line.intercept, 40.0, 400.0, 420.0)
    assert len(cluster_by_position([low, high], 720, 150)) == 1


def test_clustering_of_nothing_is_empty():
    assert cluster_by_position([], 720, 150) == []


# ── Detection against known geometry ───────────────────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_ego_lanes_match_the_generated_geometry(seed):
    img, scene = render(seed=seed, hood_fraction=0.80, difficulty=1.0)
    hood_line = int(scene.height * 0.80)
    left, right = detect_ego_lanes(mask_hood(img, 0.80), y_bottom=hood_line)

    assert left is not None and right is not None
    assert abs(left.line.x_at(hood_line) - scene.x_at("left", hood_line)) < 40
    assert abs(right.line.x_at(hood_line) - scene.x_at("right", hood_line)) < 40


def test_ego_selection_ignores_the_lane_beyond_the_ego_lane():
    """The failure this pipeline's innermost-cluster selection exists to fix.

    With an extra marking visible outside the ego lane, averaging every
    same-signed segment lands the reported right lane between the two markings
    — hundreds of pixels out. Keeping only the cluster nearest the car does not.
    """
    img, scene = render(seed=2, hood_fraction=0.80, difficulty=1.0)
    hood_line = int(scene.height * 0.80)
    frame = mask_hood(img, 0.80)
    truth = scene.x_at("right", hood_line)

    _, naive = detect_ego_lanes(frame, LaneConfig(ego_select_innermost=False),
                                y_bottom=hood_line)
    _, fixed = detect_ego_lanes(frame, LaneConfig(ego_select_innermost=True),
                                y_bottom=hood_line)

    naive_err = abs(naive.line.x_at(hood_line) - truth)
    fixed_err = abs(fixed.line.x_at(hood_line) - truth)
    assert naive_err > 100, "expected the naive average to be badly wrong here"
    assert fixed_err < 40
    assert fixed_err < naive_err / 4


def test_ego_selection_is_better_on_average_not_just_on_one_frame():
    """Aggregate version of the test above, so it cannot pass by cherry-picking."""
    naive_errs, fixed_errs = [], []
    for seed in range(12):
        img, scene = render(seed=seed, hood_fraction=0.80, difficulty=1.0)
        hood_line = int(scene.height * 0.80)
        frame = mask_hood(img, 0.80)
        truth = scene.x_at("right", hood_line)
        _, naive = detect_ego_lanes(frame, LaneConfig(ego_select_innermost=False),
                                    y_bottom=hood_line)
        _, fixed = detect_ego_lanes(frame, LaneConfig(ego_select_innermost=True),
                                    y_bottom=hood_line)
        if naive is None or fixed is None:
            continue
        naive_errs.append(abs(naive.line.x_at(hood_line) - truth))
        fixed_errs.append(abs(fixed.line.x_at(hood_line) - truth))

    assert len(fixed_errs) >= 10
    assert sum(fixed_errs) / len(fixed_errs) < 40
    assert sum(fixed_errs) < sum(naive_errs) / 3


def test_multilane_finds_more_lines_than_ego_mode():
    img, scene = render(seed=0, hood_fraction=0.80, difficulty=0.5)
    lanes = detect_multilanes(mask_hood(img, 0.80), y_bottom=int(scene.height * 0.80))
    assert len(lanes) >= 3, f"expected the outer marking too, got {len(lanes)}"


def test_multilane_results_are_ordered_left_to_right():
    img, scene = render(seed=0, hood_fraction=0.80, difficulty=0.5)
    hood = int(scene.height * 0.80)
    lanes = detect_multilanes(mask_hood(img, 0.80), y_bottom=hood)
    xs = [lane.line.x_at(hood) for lane in lanes]
    assert xs == sorted(xs)


def test_multilane_clusters_at_the_row_it_is_given():
    """Ordering is reported at the same row the tracker later matches on."""
    img, scene = render(seed=0, hood_fraction=0.80, difficulty=0.5)
    frame = mask_hood(img, 0.80)
    at_hood = detect_multilanes(frame, y_bottom=int(scene.height * 0.80))
    at_bottom = detect_multilanes(frame, y_bottom=None)
    assert len(at_hood) >= 3 and len(at_bottom) >= 3


def test_detection_is_stateless_across_calls():
    """Two different frames in a row must not contaminate each other."""
    a, scene_a = render(seed=11, hood_fraction=0.80)
    b, _ = render(seed=12, hood_fraction=0.80)
    hood = int(scene_a.height * 0.80)

    first = detect_ego_lanes(mask_hood(a, 0.80), y_bottom=hood)
    detect_ego_lanes(mask_hood(b, 0.80), y_bottom=hood)
    again = detect_ego_lanes(mask_hood(a, 0.80), y_bottom=hood)

    assert first[0].line == again[0].line
    assert first[1].line == again[1].line


# ── Trackers ───────────────────────────────────────────────────────────────


def _fit(slope, intercept, near=720.0, far=430.0):
    return LaneFit(LaneLine(slope, intercept), y_near=near, y_far=far)


def test_tracker_smooths_toward_new_observations():
    cfg = LaneConfig(smoothing=0.5)
    tracker = EgoLaneTracker(cfg)
    tracker.update(_fit(-1.0, 1000.0), _fit(1.0, -100.0), 720)
    tracker.update(_fit(-2.0, 1000.0), _fit(1.0, -100.0), 720)
    assert tracker.lines[0].slope == pytest.approx(-1.5)


def test_tracker_coasts_then_fades_out_a_lost_lane():
    cfg = LaneConfig(fade_frames=4, smoothing=1.0)
    tracker = EgoLaneTracker(cfg)
    for _ in range(4):
        tracker.update(_fit(-1.0, 1000.0), _fit(1.0, -100.0), 720)

    alphas = []
    for _ in range(4):
        drawn = tracker.update(None, None, 720)
        alphas.append(max((d.alpha for d in drawn), default=0.0))
    assert alphas == sorted(alphas, reverse=True), "fade must be monotonic"
    assert alphas[0] > 0, "a just-lost lane should still be drawn"
    assert alphas[-1] == 0.0, "a long-lost lane should be gone"


def test_tracker_fades_each_lane_independently():
    """Losing one lane must not hold the other at full opacity, or vice versa."""
    cfg = LaneConfig(fade_frames=6, smoothing=1.0)
    tracker = EgoLaneTracker(cfg)
    for _ in range(6):
        tracker.update(_fit(-1.0, 1000.0), _fit(1.0, -100.0), 720)
    for _ in range(3):
        drawn = tracker.update(_fit(-1.0, 1000.0), None, 720)

    assert len(drawn) == 2
    assert drawn[0].alpha == pytest.approx(1.0)
    assert drawn[1].alpha < 1.0


def test_tracker_rejects_an_impossible_jump():
    cfg = LaneConfig(gate_slope_delta=0.5, gate_after_age=2, smoothing=1.0)
    tracker = EgoLaneTracker(cfg)
    for _ in range(4):
        tracker.update(_fit(-1.0, 1000.0), _fit(1.0, -100.0), 720)
    tracker.update(_fit(-4.0, 1000.0), _fit(1.0, -100.0), 720)
    assert tracker.lines[0].slope == pytest.approx(-1.0), "gate let the jump through"


def test_tracker_accepts_a_gradual_change():
    cfg = LaneConfig(gate_slope_delta=0.5, gate_after_age=2, smoothing=1.0)
    tracker = EgoLaneTracker(cfg)
    for _ in range(4):
        tracker.update(_fit(-1.0, 1000.0), _fit(1.0, -100.0), 720)
    tracker.update(_fit(-1.3, 1000.0), _fit(1.0, -100.0), 720)
    assert tracker.lines[0].slope == pytest.approx(-1.3)


def test_tracker_reset_clears_state():
    tracker = EgoLaneTracker()
    tracker.update(_fit(-1.0, 1000.0), _fit(1.0, -100.0), 720)
    tracker.reset()
    assert tracker.lines == (None, None)


def test_multilane_tracker_holds_a_lane_through_a_dropped_frame():
    """The instability the reference implementation lists as an open limitation."""
    cfg = LaneConfig(fade_frames=5, smoothing=1.0)
    tracker = MultiLaneTracker(cfg)
    fits = [_fit(-1.0, 1000.0), _fit(1.0, -100.0)]
    for _ in range(5):
        tracker.update(fits, 720)
    assert tracker.lane_count == 2

    tracker.update(fits[:1], 720)  # right lane missing for one frame
    assert tracker.lane_count == 2, "a one-frame miss should not delete the lane"


def test_multilane_tracker_drops_a_lane_that_stays_gone():
    cfg = LaneConfig(fade_frames=3, smoothing=1.0)
    tracker = MultiLaneTracker(cfg)
    fits = [_fit(-1.0, 1000.0), _fit(1.0, -100.0)]
    for _ in range(3):
        tracker.update(fits, 720)
    for _ in range(4):
        tracker.update(fits[:1], 720)
    assert tracker.lane_count == 1


def test_multilane_tracker_matches_a_lane_to_itself_not_its_neighbour():
    cfg = LaneConfig(smoothing=1.0)
    tracker = MultiLaneTracker(cfg)
    fits = [_fit(-1.0, 1000.0), _fit(1.0, -100.0)]
    for _ in range(3):
        tracker.update(fits, 720)
    assert tracker.lane_count == 2, "a stable pair must not spawn extra tracks"


# ── Drawing ────────────────────────────────────────────────────────────────


def test_draw_lanes_preserves_hue_instead_of_saturating():
    """Additively blending blue onto grey asphalt yields pale cyan; this must not."""
    frame = np.full((200, 400, 3), 110, np.uint8)
    out = draw_lanes(frame, [Drawable(50, 190, 200, 60, 1.0)], color=(255, 90, 0),
                     thickness=8)
    painted = out[(out != 110).any(axis=2)]
    assert (painted == np.array([255, 90, 0])).all(axis=1).any()


def test_draw_lanes_does_not_dim_the_frame():
    frame = np.full((200, 400, 3), 110, np.uint8)
    out = draw_lanes(frame, [Drawable(50, 190, 60, 60, 1.0)], thickness=4)
    assert out[5, 350].tolist() == [110, 110, 110]


def test_draw_lanes_respects_alpha():
    frame = np.zeros((200, 400, 3), np.uint8)
    full = draw_lanes(frame, [Drawable(200, 190, 200, 20, 1.0)], color=(200, 200, 200),
                      thickness=10)
    half = draw_lanes(frame, [Drawable(200, 190, 200, 20, 0.5)], color=(200, 200, 200),
                      thickness=10)
    assert int(half[100, 200][0]) < int(full[100, 200][0])


def test_draw_lanes_without_input_returns_the_frame_untouched():
    frame = np.zeros((20, 20, 3), np.uint8)
    assert draw_lanes(frame, []) is frame
    assert draw_lanes(frame, [Drawable(0, 0, 5, 5, 0.0)]) is frame


# ── Config ─────────────────────────────────────────────────────────────────


def test_config_scaling_is_identity_at_the_reference_height():
    cfg = LaneConfig()
    assert cfg.scaled_for(REFERENCE_HEIGHT) == cfg


def test_config_scaling_tracks_resolution():
    cfg = LaneConfig()
    assert cfg.scaled_for(1440).hough_min_line_length == 2 * cfg.hough_min_line_length
    assert cfg.scaled_for(360).hough_threshold < cfg.hough_threshold


def test_config_scaling_keeps_resolution_free_fields_alone():
    cfg = LaneConfig()
    scaled = cfg.scaled_for(1440)
    assert scaled.slope_min_abs == cfg.slope_min_abs
    assert scaled.smoothing == cfg.smoothing
    assert scaled.roi_y_top_frac == cfg.roi_y_top_frac


def test_config_scaling_never_collapses_to_zero():
    tiny = LaneConfig().scaled_for(32)
    assert tiny.hough_threshold >= 10
    assert tiny.hough_min_line_length >= 8


def test_config_scaling_rejects_a_bad_height():
    with pytest.raises(ValueError):
        LaneConfig().scaled_for(0)
