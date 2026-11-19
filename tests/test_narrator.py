import unittest

from detector import Detection
from narrator import Narrator
from spatial import SpatialDetection


def make_scene(label: str, position: str, distance: float) -> SpatialDetection:
    detection = Detection(label, 0.9, 10, 10, 30, 50)
    return SpatialDetection(
        detection=detection,
        horizontal_position=position,
        distance_m=distance,
        range_label="near" if distance <= 3 else "far",
    )


class NarratorTests(unittest.TestCase):
    def test_announces_closest_object_and_throttles_repeats(self):
        narrator = Narrator(cooldown_seconds=4, language="en")
        scene = [make_scene("chair", "right", 2.8), make_scene("person", "left", 1.2)]

        self.assertEqual(
            narrator.describe(scene, now=10), "Person on your left, about 1 meter."
        )
        self.assertIsNone(narrator.describe(scene, now=12))
        self.assertEqual(
            narrator.describe(scene, now=15), "Person on your left, about 1 meter."
        )

    def test_scene_query_reports_empty_frame_in_french(self):
        narrator = Narrator(language="fr")

        self.assertEqual(narrator.describe_scene([]), "Aucun objet détecté.")
