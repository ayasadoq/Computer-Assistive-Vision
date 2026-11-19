"""Convert spatial detections into concise, rate-limited speech text."""

from __future__ import annotations

import time

from spatial import SpatialDetection


class Narrator:
    def __init__(self, cooldown_seconds: float = 4.0, language: str = "en") -> None:
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")
        self.cooldown_seconds = cooldown_seconds
        self.language = language.lower().split("-")[0]
        self._last_announcement_at = 0.0
        self._last_key: tuple[str, str, str] | None = None

    def describe(
        self, detections: list[SpatialDetection], now: float | None = None
    ) -> str | None:
        """Announce the closest useful object, throttling repeated messages."""
        if not detections:
            return None

        current_time = time.monotonic() if now is None else now
        closest = min(
            detections,
            key=lambda item: (
                item.distance_m,
                -item.detection.confidence,
            ),
        )
        distance = closest.distance_m
        if distance == float("inf"):
            return None
        distance_m = max(0, int(round(distance)))
        key = (
            closest.detection.label,
            closest.horizontal_position,
            closest.range_label,
        )
        if (
            key == self._last_key
            and current_time - self._last_announcement_at < self.cooldown_seconds
        ):
            return None

        self._last_key = key
        self._last_announcement_at = current_time
        return self._format(closest)

    def describe_scene(self, detections: list[SpatialDetection]) -> str:
        if not detections:
            return "Aucun objet détecté." if self.language == "fr" else "No objects detected."
        closest = min(detections, key=lambda item: item.distance_m)
        return self._format(closest)

    def _format(self, item: SpatialDetection) -> str:
        distance_m = max(1, int(round(item.distance_m)))
        label = item.detection.label[:1].upper() + item.detection.label[1:]
        if self.language == "fr":
            side = {
                "left": "sur votre gauche",
                "center": "devant vous",
                "right": "sur votre droite",
            }[item.horizontal_position]
            unit = "mètre" if distance_m == 1 else "mètres"
            return f"{label} {side}, à environ {distance_m} {unit}."
        side = {
            "left": "on your left",
            "center": "ahead",
            "right": "on your right",
        }[item.horizontal_position]
        unit = "meter" if distance_m == 1 else "meters"
        return f"{label} {side}, about {distance_m} {unit}."
