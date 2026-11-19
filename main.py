"""Command-line voice-guided vision assistant."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Any

import cv2
import yaml

from detector import ObjectDetector
from narrator import Narrator
from spatial import SpatialAnalyzer, SpatialDetection
from voice import SpeechOutput, VoiceCommandListener


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice-guided computer vision assistant")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration")
    parser.add_argument("--video", default=None, help="Camera index or video file path")
    parser.add_argument("--no-voice", action="store_true", help="Disable TTS and voice input")
    return parser.parse_args()


def _draw_detections(
    frame: Any, detections: list[SpatialDetection], narration: str | None
) -> None:
    for item in detections:
        detection = item.detection
        label = (
            f"{detection.label} {detection.confidence:.0%} | "
            f"{item.horizontal_position}, {item.range_label} | {item.distance_m:.1f}m"
        )
        cv2.rectangle(frame, (detection.x1, detection.y1), (detection.x2, detection.y2), (0, 220, 0), 2)
        cv2.putText(
            frame,
            label,
            (detection.x1, max(20, detection.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
        )
    if narration:
        cv2.putText(frame, narration[:100], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    elif not detections:
        cv2.putText(frame, "No objects detected", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def run() -> None:
    args = parse_args()
    config = load_config(args.config)
    detector_config = config.get("detector", {})
    spatial_config = config.get("spatial", {})
    speech_config = config.get("speech", {})

    detector = ObjectDetector(
        model_path=detector_config.get("model_path", "yolov8n.pt"),
        confidence=float(detector_config.get("confidence", 0.45)),
    )
    analyzer = SpatialAnalyzer(
        focal_length_px=float(spatial_config.get("focal_length_px", 700)),
        object_heights_m=spatial_config.get("object_heights_m", {}),
        near_distance_m=float(spatial_config.get("near_distance_m", 3.0)),
        default_object_height_m=float(
            spatial_config.get("default_object_height_m", 1.7)
        ),
    )
    language = speech_config.get("language", "en")
    narrator = Narrator(
        cooldown_seconds=float(speech_config.get("announcement_cooldown_seconds", 4)),
        language=language,
    )

    source: int | str = args.video if args.video is not None else 0
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    latest_scene: list[SpatialDetection] = []
    scene_lock = threading.Lock()
    stop_event = threading.Event()
    speaker = None if args.no_voice else SpeechOutput(language=language)

    def query_scene() -> None:
        with scene_lock:
            scene = list(latest_scene)
        message = narrator.describe_scene(scene)
        print(message)
        if speaker is not None:
            speaker.speak(message)

    listener = None
    if not args.no_voice:
        listener = VoiceCommandListener(
            on_query=query_scene,
            on_quit=stop_event.set,
            language=speech_config.get("recognition_language", "en-US"),
        )
        listener.start()

    print("Assistant running. Press Q to quit.")
    try:
        while not stop_event.is_set():
            success, frame = capture.read()
            if not success:
                if isinstance(source, int):
                    raise RuntimeError("Could not read a frame from the camera")
                break

            detected = detector.detect(frame)
            scene = analyzer.analyze(detected, frame.shape[1])
            with scene_lock:
                latest_scene = scene

            narration = narrator.describe(scene)
            if narration:
                print(narration)
                if speaker is not None:
                    speaker.speak(narration)
            _draw_detections(frame, scene, narration)
            cv2.imshow("Voice-guided vision assistant", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_event.set()
        capture.release()
        cv2.destroyAllWindows()
        if listener is not None:
            listener.close()
        if speaker is not None:
            speaker.close()


if __name__ == "__main__":
    run()
