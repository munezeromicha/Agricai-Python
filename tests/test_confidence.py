"""Confidence banding tests — run with: python -m unittest discover -s tests"""

import unittest

from app.inference.confidence import confidence_guidance, confidence_level, is_actionable


class ConfidenceLevelTests(unittest.TestCase):
    def test_strong_well_separated_prediction_is_high(self):
        self.assertEqual(confidence_level(94, 40), "high")
        self.assertEqual(confidence_level(85, 12), "high")

    def test_thin_margin_demotes_exactly_one_band(self):
        self.assertEqual(confidence_level(92, 3), "medium")
        self.assertEqual(confidence_level(70, 2), "low")

    def test_mid_and_low_ranges(self):
        self.assertEqual(confidence_level(70, 20), "medium")
        self.assertEqual(confidence_level(55, 30), "low")
        self.assertEqual(confidence_level(44, 30), "very_low")

    def test_missing_margin_uses_score_alone(self):
        self.assertEqual(confidence_level(90, None), "high")
        self.assertEqual(confidence_level(66), "medium")

    def test_bad_input_never_raises(self):
        self.assertEqual(confidence_level(None), "very_low")
        self.assertEqual(confidence_level("bogus"), "very_low")
        self.assertEqual(confidence_level(10_000, None), "high")
        self.assertEqual(confidence_level(-50), "very_low")

    def test_matches_node_implementation_thresholds(self):
        """Same table as Agricai-Node/src/lib/confidence.mjs — keep both in step."""
        cases = [
            (94, 40, "high"),
            (85, 12, "high"),
            (92, 3, "medium"),
            (70, 20, "medium"),
            (65, 6, "medium"),
            (55, 30, "low"),
            (44, 30, "very_low"),
        ]
        for score, margin, expected in cases:
            with self.subTest(score=score, margin=margin):
                self.assertEqual(confidence_level(score, margin), expected)


class GuidanceTests(unittest.TestCase):
    def test_every_band_has_bilingual_guidance(self):
        for level in ("high", "medium", "low", "very_low"):
            en, rw = confidence_guidance(level)
            self.assertGreater(len(en), 20, level)
            self.assertGreater(len(rw), 20, level)

    def test_only_high_and_medium_are_actionable(self):
        self.assertTrue(is_actionable("high"))
        self.assertTrue(is_actionable("medium"))
        self.assertFalse(is_actionable("low"))
        self.assertFalse(is_actionable("very_low"))


if __name__ == "__main__":
    unittest.main()
