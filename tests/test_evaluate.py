from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import calculate_regression_metrics, evaluate_models_and_save_outputs


class EvaluateTests(unittest.TestCase):
    def test_calculate_regression_metrics_zero_error(self) -> None:
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        metrics = calculate_regression_metrics(y_true, y_pred)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["r2"], 1.0)

    def test_evaluate_models_writes_expected_artifacts(self) -> None:
        y_true = pd.Series([10.0, 12.0, 14.0, 16.0])
        predictions = {
            "mean_baseline": np.array([13.0, 13.0, 13.0, 13.0]),
            "linear_regression": np.array([10.1, 11.9, 14.0, 16.1]),
            "xgboost": np.array([10.5, 12.2, 13.8, 15.8]),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            metrics_frame = evaluate_models_and_save_outputs(y_true, predictions, output_dir)

            self.assertTrue((output_dir / "validation_metrics.csv").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertEqual(set(metrics_frame["model"]), set(predictions.keys()))

            with (output_dir / "summary.json").open("r", encoding="utf-8") as summary_file:
                summary = json.load(summary_file)

            self.assertEqual(set(summary["models_evaluated"]), set(predictions.keys()))
            self.assertIn("best_model_by_rmse", summary)


if __name__ == "__main__":
    unittest.main()