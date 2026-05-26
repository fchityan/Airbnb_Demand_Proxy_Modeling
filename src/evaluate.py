from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_and_save_outputs(
    y_true: pd.Series,
    y_proba: np.ndarray,
    output_dir: Path,
) -> dict[str, float]:
    """Calculate metrics and save evaluation artifacts to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred_default = (y_proba >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred_default)),
        "precision": float(precision_score(y_true, y_pred_default, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_default, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_default, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }

    metrics_frame = pd.DataFrame([metrics])
    metrics_frame.to_csv(output_dir / "validation_metrics.csv", index=False)

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)

    threshold_table = pd.DataFrame(
        {
            "threshold": np.append(thresholds, 1.0),
            "precision": precision,
            "recall": recall,
            "f1": f1_scores,
        }
    )
    threshold_table.to_csv(output_dir / "threshold_tuning.csv", index=False)

    best_row = threshold_table.loc[threshold_table["f1"].idxmax()]
    summary = {
        "metrics": metrics,
        "best_threshold": {
            "threshold": float(best_row["threshold"]),
            "precision": float(best_row["precision"]),
            "recall": float(best_row["recall"]),
            "f1": float(best_row["f1"]),
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    return metrics
