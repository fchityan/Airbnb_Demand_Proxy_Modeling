from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_regression


def load_data(random_state: int = 42, n_samples: int = 2000) -> pd.DataFrame:
    """Create a synthetic regression dataset for model training."""
    features, target = make_regression(
        n_samples=n_samples,
        n_features=12,
        n_informative=8,
        n_targets=1,
        noise=12.0,
        random_state=random_state,
    )

    columns = [f"feature_{index}" for index in range(features.shape[1])]
    dataframe = pd.DataFrame(features, columns=columns)
    dataframe["target"] = target
    return dataframe
