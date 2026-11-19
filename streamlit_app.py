"""Live browser demo for the voice-guided vision assistant."""

from __future__ import annotations

import threading
from pathlib import Path

import av
import cv2
import streamlit as st
import yaml
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from detector import ObjectDetector
from narrator import Narrator
from spatial import SpatialAnalyzer, SpatialDetection


class LatestScene:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.detections: list[SpatialDetection] = []
        self.narration = "Starting camera..."

    def update(self, detections: list[SpatialDetection], narration: str) -> None:
        with self.lock:
            self.detections = detections
            self.narration = narration

    def read(self) -> tuple[list[SpatialDetection], str]:
        with self.lock:
            return list(self.detections), self.narration


class VisionProcessor(VideoProcessorBase):
    def __init__(
        self,
        detector: ObjectDetector,
        analyzer: SpatialAnalyzer,
        narrator: Narrator,
        latest_scene: LatestScene,
    ) -> None:
        self.detector = detector
        self.analyzer = analyzer
        self.narrator = narrator
        self.latest_scene = latest_scene

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        detections = self.detector.detect(image)
        scene = self.analyzer.analyze(detections, image.shape[1])
        narration = self.narrator.describe(scene)
        self.latest_scene.update(scene, narration or ("No objects detected." if not scene else ""))
        for item in scene:
            detection = item.detection
            cv2.rectangle(
                image,
                (detection.x1, detection.y1),
                (detection.x2, detection.y2),
                (0, 220, 0),
                2,
            )
            cv2.putText(
                image,
                f"{detection.label} {detection.confidence:.0%} | "
                f"{item.horizontal_position}, {item.range_label} | {item.distance_m:.1f}m",
                (detection.x1, max(20, detection.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 0),
                2,
            )
        return av.VideoFrame.from_ndarray(image, format="bgr24")


@st.cache_resource
def load_runtime(
    model_path: str,
    confidence: float,
    focal_length_px: float,
    near_distance_m: float,
    default_object_height_m: float,
    object_heights_m: dict[str, float],
):
    detector = ObjectDetector(model_path, confidence)
    analyzer = SpatialAnalyzer(
        focal_length_px=focal_length_px,
        near_distance_m=near_distance_m,
        default_object_height_m=default_object_height_m,
        object_heights_m=object_heights_m,
    )
    return detector, analyzer


def main() -> None:
    st.set_page_config(page_title="Vision Assistant", page_icon="👁️", layout="wide")
    st.title("Voice-guided vision assistant")
    st.caption("Live object detection with spatial hints. Distance is a calibrated estimate, not a safety measurement.")
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")) or {}
    detector_config = config.get("detector", {})
    spatial_config = config.get("spatial", {})
    speech_config = config.get("speech", {})

    try:
        detector, analyzer = load_runtime(
            detector_config.get("model_path", "yolov8n.pt"),
            float(detector_config.get("confidence", 0.45)),
            float(spatial_config.get("focal_length_px", 700)),
            float(spatial_config.get("near_distance_m", 3.0)),
            float(spatial_config.get("default_object_height_m", 1.7)),
            spatial_config.get("object_heights_m", {}),
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.info("Install dependencies and make the configured model file available.")
        return

    latest_scene = st.session_state.setdefault("latest_scene", LatestScene())
    narrator = Narrator(
        cooldown_seconds=float(speech_config.get("announcement_cooldown_seconds", 4)),
        language=speech_config.get("language", "en"),
    )
    st.write("Allow camera access to start the live detector.")
    webrtc_streamer(
        key="vision-assistant",
        video_processor_factory=lambda: VisionProcessor(
            detector, analyzer, narrator, latest_scene
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    @st.fragment(run_every="1s")
    def scene_panel() -> None:
        detections, narration = latest_scene.read()
        st.subheader("Scene narration")
        st.write(narration or "No new announcement (cooldown active).")
        if detections:
            st.dataframe(
                [
                    {
                        "Object": item.detection.label,
                        "Confidence": f"{item.detection.confidence:.0%}",
                        "Position": item.horizontal_position,
                        "Range": item.range_label,
                        "Distance (estimate)": f"{item.distance_m:.1f} m",
                    }
                    for item in detections
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No objects detected in the latest frame.")

    scene_panel()


if __name__ == "__main__":
    main()
