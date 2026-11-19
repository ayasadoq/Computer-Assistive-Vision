"""Spatial descriptions and heuristic distance estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from detector import Detection


@dataclass(frozen=True)
class SpatialDetection:
    detection: Detection
    horizontal_position: str
    distance_m: float
    range_label: str


class SpatialAnalyzer:
    """Estimate object position and distance from its bounding box."""

    def __init__(
        self,
        focal_length_px: float = 700.0,
        object_heights_m: Mapping[str, float] | None = None,
        near_distance_m: float = 3.0,
        default_object_height_m: float = 1.7,
    ) -> None:
        if focal_length_px <= 0 or near_distance_m <= 0 or default_object_height_m <= 0:
            raise ValueError("Calibration and distance values must be positive")
        self.focal_length_px = focal_length_px
        self.object_heights_m = dict(object_heights_m or {})
        self.near_distance_m = near_distance_m
        self.default_object_height_m = default_object_height_m

    def analyze(
        self, detections: list[Detection], frame_width: int
    ) -> list[SpatialDetection]:
        if frame_width <= 0:
            raise ValueError("frame_width must be positive")
        analyzed: list[SpatialDetection] = []
        for detection in detections:
            distance_m = self.estimate_distance(detection)
            analyzed.append(
                SpatialDetection(
                    detection=detection,
                    horizontal_position=self._horizontal_position(detection, frame_width),
                    distance_m=distance_m,
                    range_label="near" if distance_m <= self.near_distance_m else "far",
                )
            )
        return sorted(analyzed, key=lambda item: item.distance_m)

    def estimate_distance(self, detection: Detection) -> float:
        """Use pinhole-camera geometry; accuracy depends on object-size calibration."""
        height_px = detection.height_px
        if height_px <= 0:
            return float("inf")
        object_height_m = self.object_heights_m.get(
            detection.label, self.default_object_height_m
        )
        if object_height_m <= 0:
            raise ValueError(f"Object height for {detection.label!r} must be positive")
        return (object_height_m * self.focal_length_px) / height_px

    @staticmethod
    def _horizontal_position(detection: Detection, frame_width: int) -> str:
        center_x = (detection.x1 + detection.x2) / 2
        relative_x = center_x / frame_width
        if relative_x < 1 / 3:
            return "left"
        if relative_x > 2 / 3:
            return "right"
        return "center"
