from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_classification


def load_data(random_state: int = 42, n_samples: int = 2000) -> pd.DataFrame:
    """Create a synthetic binary-classification dataset for model training."""
    features, target = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=random_state,
    )

    columns = [f"feature_{index}" for index in range(features.shape[1])]
    dataframe = pd.DataFrame(features, columns=columns)
    dataframe["target"] = target
    return dataframe
