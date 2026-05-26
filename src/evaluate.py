from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_and_save_outputs(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_dir: Path,
) -> dict[str, float]:
    """Calculate metrics and save evaluation artifacts to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
    }

    metrics_frame = pd.DataFrame([metrics])
    metrics_frame.to_csv(output_dir / "validation_metrics.csv", index=False)
    summary = {
        "metrics": metrics,
        "prediction_summary": {
            "mean_actual": float(np.mean(y_true)),
            "mean_predicted": float(np.mean(y_pred)),
            "std_actual": float(np.std(y_true)),
            "std_predicted": float(np.std(y_pred)),
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    return metrics
