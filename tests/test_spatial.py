import unittest

from detector import Detection
from spatial import SpatialAnalyzer


class SpatialAnalyzerTests(unittest.TestCase):
    def test_horizontal_position_and_calibrated_distance(self):
        analyzer = SpatialAnalyzer(
            focal_length_px=600,
            object_heights_m={"person": 1.8},
            near_distance_m=3,
        )
        detections = [
            Detection("person", 0.9, 0, 0, 80, 360),
            Detection("person", 0.8, 460, 0, 540, 420),
        ]

        scene = analyzer.analyze(detections, frame_width=600)

        self.assertEqual(scene[0].horizontal_position, "right")
        self.assertEqual(round(scene[0].distance_m, 2), 2.57)
        self.assertEqual(scene[0].range_label, "near")
        self.assertEqual(scene[1].horizontal_position, "left")

    def test_zero_height_detection_has_no_finite_distance(self):
        analyzer = SpatialAnalyzer()
        detection = Detection("person", 0.9, 10, 10, 40, 10)

        self.assertEqual(analyzer.estimate_distance(detection), float("inf"))
