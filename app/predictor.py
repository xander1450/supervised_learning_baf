from __future__ import annotations

from typing import Iterable

from app.model_store import ModelBundle


class TextClassifierService:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self.pipeline = bundle.pipeline
        self.metadata = bundle.metadata
        self.classes = list(bundle.metadata["classes"])
        self.model_version = bundle.metadata["model_version"]

    def predict_texts(self, texts: Iterable[str]) -> list[dict]:
        text_list = list(texts)
        probabilities = self.pipeline.predict_proba(text_list)
        predictions = self.pipeline.predict(text_list)

        results: list[dict] = []
        for prediction, prob_row in zip(predictions, probabilities):
            prob_map = {
                class_name: float(prob_value)
                for class_name, prob_value in zip(self.classes, prob_row)
            }
            confidence = max(prob_map.values())
            results.append(
                {
                    "prediction": str(prediction),
                    "confidence": confidence,
                    "probabilities": prob_map,
                    "model_version": self.model_version,
                }
            )
        return results

    def predict_records(self, records: list[dict]) -> list[dict]:
        texts = [record["text"] for record in records]
        base_results = self.predict_texts(texts)

        enriched_results: list[dict] = []
        for record, base_result in zip(records, base_results):
            enriched_results.append(
                {
                    "id": record.get("id"),
                    **base_result,
                }
            )
        return enriched_results
