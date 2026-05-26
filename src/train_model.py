from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train a baseline tree model."""
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def build_feature_importance(
    model: RandomForestRegressor,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return feature importances sorted from highest to lowest."""
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)
