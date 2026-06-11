import json
from dataclasses import dataclass
from pathlib import Path

from app.schemas import DetectionResult, DiseaseType


@dataclass(frozen=True)
class ClassEntry:
    class_id: str
    type: DiseaseType
    diseaseName: str
    diseaseNameRw: str
    explanation: str
    explanationRw: str
    treatment: str
    treatmentRw: str
    prevention: str
    preventionRw: str
    care: str
    careRw: str


class KnowledgeBase:
    def __init__(self, path: Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._by_id: dict[str, ClassEntry] = {}
        self._order: list[str] = []
        for item in raw["classes"]:
            e = ClassEntry(
                class_id=item["class_id"],
                type=item["type"],
                diseaseName=item["diseaseName"],
                diseaseNameRw=item["diseaseNameRw"],
                explanation=item["explanation"],
                explanationRw=item["explanationRw"],
                treatment=item["treatment"],
                treatmentRw=item["treatmentRw"],
                prevention=item["prevention"],
                preventionRw=item["preventionRw"],
                care=item["care"],
                careRw=item["careRw"],
            )
            self._by_id[e.class_id] = e
            self._order.append(e.class_id)

    @property
    def class_ids(self) -> list[str]:
        return list(self._order)

    @property
    def trainable_class_ids(self) -> list[str]:
        """Model output classes in knowledge order (excludes knowledge-only ``unknown``)."""
        return [cid for cid in self._order if cid != "unknown"]

    @property
    def disease_class_ids(self) -> list[str]:
        """Tomato disease classes only — excludes Not_Tomato reject class."""
        return [cid for cid in self.trainable_class_ids if cid != "Not_Tomato"]

    @property
    def num_classes(self) -> int:
        return len(self._order)

    @property
    def num_trainable_classes(self) -> int:
        return len(self.trainable_class_ids)

    def get(self, class_id: str) -> ClassEntry:
        return self._by_id[class_id]

    def try_get(self, class_id: str) -> ClassEntry | None:
        return self._by_id.get(class_id)

    def to_detection(self, entry: ClassEntry, confidence_pct: float) -> DetectionResult:
        return DetectionResult(
            diseaseName=entry.diseaseName,
            diseaseNameRw=entry.diseaseNameRw,
            confidence=round(confidence_pct, 1),
            type=entry.type,
            explanation=entry.explanation,
            explanationRw=entry.explanationRw,
            treatment=entry.treatment,
            treatmentRw=entry.treatmentRw,
            prevention=entry.prevention,
            preventionRw=entry.preventionRw,
            care=entry.care,
            careRw=entry.careRw,
        )
