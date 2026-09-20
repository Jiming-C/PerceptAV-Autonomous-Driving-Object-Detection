"""Dashcam perception: YOLOv8 object detection plus classical lane finding."""

from .config import DetectorConfig, LaneConfig, PipelineConfig
from .detector import RunSummary, get_model, process_image, process_video
from .lanes import (
    EgoLaneTracker,
    LaneFit,
    LaneLine,
    MultiLaneTracker,
    detect_ego_lanes,
    detect_multilanes,
    draw_lanes,
    mask_hood,
)

__all__ = [
    "DetectorConfig",
    "EgoLaneTracker",
    "LaneConfig",
    "LaneFit",
    "LaneLine",
    "MultiLaneTracker",
    "PipelineConfig",
    "RunSummary",
    "detect_ego_lanes",
    "detect_multilanes",
    "draw_lanes",
    "get_model",
    "mask_hood",
    "process_image",
    "process_video",
]
