"""DetectResponse contract tests — the fields the mobile app depends on."""

import unittest

from app.schemas import DetectionResult, DetectResponse


def result(**overrides) -> DetectionResult:
    base = dict(
        diseaseName="Late Blight",
        diseaseNameRw="Kirabiranya",
        confidence=91.0,
        type="disease",
        explanation="Dark water-soaked lesions.",
        explanationRw="Ibisebe byijimye.",
        treatment="Remove affected leaves.",
        treatmentRw="Kuraho amababi arwaye.",
        prevention="Rotate crops.",
        preventionRw="Simbuza ibihingwa.",
        care="Water at the base.",
        careRw="Uhire hasi.",
    )
    base.update(overrides)
    return DetectionResult(**base)


def response(**overrides) -> DetectResponse:
    base = dict(
        result=result(),
        model_version="tomato-3",
        request_id="req-1",
        inference_mode="roboflow",
    )
    base.update(overrides)
    return DetectResponse(**base)


class DetectResponseConfidenceTests(unittest.TestCase):
    def test_derives_a_high_band_from_top_confidence(self):
        res = response(top_confidence_pct=91.0, confidence_margin_pct=35.0)
        self.assertEqual(res.confidence_level, "high")
        self.assertTrue(res.actionable)
        self.assertIn("High confidence", res.confidence_guidance)
        self.assertTrue(res.confidence_guidance_rw)

    def test_falls_back_to_result_confidence_when_top_pct_missing(self):
        res = response(result=result(confidence=70.0))
        self.assertEqual(res.confidence_level, "medium")

    def test_thin_margin_demotes_the_band(self):
        res = response(top_confidence_pct=91.0, confidence_margin_pct=4.0)
        self.assertEqual(res.confidence_level, "medium")

    def test_rejected_results_are_never_actionable(self):
        res = response(
            result=result(type="unknown", confidence=0.0),
            rejection_reason="wrong_crop",
            top_confidence_pct=99.0,
        )
        self.assertEqual(res.confidence_level, "very_low")
        self.assertFalse(res.actionable)
        self.assertIn("do not act", res.confidence_guidance.lower())

    def test_unknown_type_without_rejection_is_also_very_low(self):
        res = response(result=result(type="unknown", confidence=88.0), top_confidence_pct=88.0)
        self.assertEqual(res.confidence_level, "very_low")

    def test_serialised_payload_exposes_the_new_fields(self):
        payload = response(top_confidence_pct=88.0, confidence_margin_pct=30.0).model_dump()
        for key in ("confidence_level", "confidence_guidance", "confidence_guidance_rw", "actionable"):
            self.assertIn(key, payload)
        self.assertEqual(payload["confidence_level"], "high")

    def test_confidence_out_of_range_is_rejected_by_validation(self):
        with self.assertRaises(Exception):
            result(confidence=140.0)


if __name__ == "__main__":
    unittest.main()
