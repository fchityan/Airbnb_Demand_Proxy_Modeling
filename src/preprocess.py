from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_and_scale(
    dataframe: pd.DataFrame,
    target_column: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data and scale features for model training."""
    if target_column not in dataframe.columns:
        raise ValueError(f"target_column '{target_column}' is not present in the dataframe.")
    if len(dataframe) < 2:
        raise ValueError("dataframe must contain at least 2 rows.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]

    if features.empty:
        raise ValueError("dataframe must include at least one feature column.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=x_train.columns,
        index=x_train.index,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=x_test.columns,
        index=x_test.index,
    )

    return x_train_scaled, x_test_scaled, y_train, y_test
