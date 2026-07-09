from __future__ import annotations

from pathlib import Path

from src.data_loader import load_data
from src.evaluate import evaluate_models_and_save_outputs
from src.preprocess import split_and_scale
from src.train_model import (
    build_feature_importance_for_models,
    predict_mean_baseline,
    train_models,
)


def run_pipeline(
    output_dir: Path,
    random_state: int = 42,
    n_samples: int = 2000,
    test_size: float = 0.2,
) -> dict[str, str]:
    """Train models, evaluate them, and write output artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = load_data(random_state=random_state, n_samples=n_samples)
    x_train, x_test, y_train, y_test = split_and_scale(
        dataframe,
        test_size=test_size,
        random_state=random_state,
    )
    models = train_models(x_train, y_train, random_state=random_state)

    predictions = {
        "mean_baseline": predict_mean_baseline(y_train, len(x_test)),
        "linear_regression": models["linear_regression"].predict(x_test),
        "xgboost": models["xgboost"].predict(x_test),
    }

    feature_importance = build_feature_importance_for_models(models, list(x_train.columns))
    feature_importance_path = output_dir / "feature_importance.csv"
    feature_importance.to_csv(feature_importance_path, index=False)

    evaluate_models_and_save_outputs(y_test, predictions, output_dir)

    return {
        "feature_importance": str(feature_importance_path),
        "validation_metrics": str(output_dir / "validation_metrics.csv"),
        "summary": str(output_dir / "summary.json"),
    }


if __name__ == "__main__":
    generated = run_pipeline(output_dir=Path("output"))
    for name, path in generated.items():
        print(f"{name}: {path}")