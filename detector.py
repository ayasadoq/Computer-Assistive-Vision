"""YOLO-based object detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class Detection:
    """A detected object in pixel coordinates."""

    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def height_px(self) -> int:
        return max(0, self.y2 - self.y1)


class ObjectDetector:
    """Load an Ultralytics model once and expose a small detection API."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.45) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required. Install project dependencies with "
                "'pip install -r requirements.txt'."
            ) from exc

        self.confidence = confidence
        self.model: Any = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for one OpenCV BGR frame."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            names = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                detections.append(
                    Detection(
                        label=str(names[class_id]),
                        confidence=float(box.conf[0].item()),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )
        return detections
