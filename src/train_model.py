from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> LinearRegression:
    """Backward-compatible single-model trainer (linear regression)."""
    return train_models(x_train, y_train, random_state=random_state)["linear_regression"]


def train_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train regression models used in the project."""
    linear_regression = LinearRegression()
    linear_regression.fit(x_train, y_train)

    xgboost_regressor = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    xgboost_regressor.fit(x_train, y_train)

    return {
        "linear_regression": linear_regression,
        "xgboost": xgboost_regressor,
    }


def predict_mean_baseline(y_train: pd.Series, sample_count: int) -> np.ndarray:
    """Return constant predictions using the train-set mean."""
    baseline_value = float(y_train.mean())
    return np.full(sample_count, baseline_value, dtype=float)


def build_feature_importance(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Backward-compatible single-model feature importance export."""
    return build_feature_importance_for_models(
        {
            "linear_regression": model,
        },
        feature_names,
    )


def build_feature_importance_for_models(
    models: dict[str, Any],
    feature_names: list[str],
) -> pd.DataFrame:
    """Build a combined feature-importance table across supported models."""
    rows: list[pd.DataFrame] = []

    linear_model = models.get("linear_regression")
    if linear_model is not None and hasattr(linear_model, "coef_"):
        coefficients = pd.Series(linear_model.coef_, index=feature_names)
        rows.append(
            pd.DataFrame(
                {
                    "model": "linear_regression",
                    "feature": feature_names,
                    "importance": coefficients.abs().values,
                    "coefficient": coefficients.values,
                }
            )
        )

    xgboost_model = models.get("xgboost")
    if xgboost_model is not None and hasattr(xgboost_model, "feature_importances_"):
        rows.append(
            pd.DataFrame(
                {
                    "model": "xgboost",
                    "feature": feature_names,
                    "importance": xgboost_model.feature_importances_,
                    "coefficient": np.nan,
                }
            )
        )

    if not rows:
        return pd.DataFrame(columns=["model", "feature", "importance", "coefficient"])

    combined = pd.concat(rows, ignore_index=True)
    return combined.sort_values(["model", "importance"], ascending=[True, False]).reset_index(drop=True)
