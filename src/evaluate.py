from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _validate_predictions(
    y_true: pd.Series,
    predictions: dict[str, np.ndarray],
) -> None:
    if not predictions:
        raise ValueError("predictions must include at least one model output.")

    expected_length = len(y_true)
    if expected_length == 0:
        raise ValueError("y_true must not be empty.")

    for model_name, y_pred in predictions.items():
        if len(y_pred) != expected_length:
            raise ValueError(
                f"Prediction length mismatch for model '{model_name}': "
                f"expected {expected_length}, got {len(y_pred)}."
            )
        if not np.isfinite(y_pred).all():
            raise ValueError(f"Predictions for model '{model_name}' contain non-finite values.")


def calculate_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate standard regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_models_and_save_outputs(
    y_true: pd.Series,
    predictions: dict[str, np.ndarray],
    output_dir: Path,
) -> pd.DataFrame:
    """Evaluate multiple regression models and save metrics + summary artifacts."""
    _validate_predictions(y_true, predictions)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, float | str]] = []
    prediction_summary: dict[str, dict[str, float]] = {}

    for model_name, y_pred in predictions.items():
        model_metrics = calculate_regression_metrics(y_true, y_pred)
        metrics_rows.append({"model": model_name, **model_metrics})
        prediction_summary[model_name] = {
            "mean_predicted": float(np.mean(y_pred)),
            "std_predicted": float(np.std(y_pred)),
        }

    metrics_frame = pd.DataFrame(metrics_rows).sort_values("rmse", ascending=True).reset_index(drop=True)
    metrics_frame.to_csv(output_dir / "validation_metrics.csv", index=False)

    best_row = metrics_frame.iloc[0].to_dict()
    summary = {
        "models_evaluated": list(predictions.keys()),
        "best_model_by_rmse": {
            "model": str(best_row["model"]),
            "mae": float(best_row["mae"]),
            "rmse": float(best_row["rmse"]),
            "r2": float(best_row["r2"]),
        },
        "actual_summary": {
            "mean_actual": float(np.mean(y_true)),
            "std_actual": float(np.std(y_true)),
        },
        "prediction_summary": prediction_summary,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    return metrics_frame


def evaluate_and_save_outputs(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_dir: Path,
) -> dict[str, float]:
    """Backward-compatible single-model evaluator and artifact writer."""
    metrics = calculate_regression_metrics(y_true, y_pred)
    evaluate_models_and_save_outputs(
        y_true=y_true,
        predictions={"model": y_pred},
        output_dir=output_dir,
    )
    return metrics
