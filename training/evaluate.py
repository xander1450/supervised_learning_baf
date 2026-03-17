from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss



def evaluate_classifier(
    *,
    y_true: list[str],
    y_pred: list[str],
    y_proba: list[list[float]],
    labels: list[str],
) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_proba, labels=labels)),
        "classification_report": report,
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion,
        },
    }
    return metrics
