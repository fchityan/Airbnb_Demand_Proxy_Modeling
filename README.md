# Airbnb Demand Proxy Modeling

## Executive Summary
This repository contains a reproducible machine learning workflow for modeling Airbnb demand proxy behavior as a regression task. It combines exploratory analysis in a notebook with a modular `src/` pipeline and automated unit tests.

## Project Objectives
- Build baseline and model-based demand proxy predictions.
- Compare model quality using MAE, RMSE, and R2.
- Export explainability artifacts for both linear and tree-based models.
- Keep the project reproducible via scriptable pipeline steps and tests.

## Analytical Scope
The primary notebook is [Airbnb Demand Proxy Modeling.ipynb](Airbnb%20Demand%20Proxy%20Modeling.ipynb).

Pipeline modules:
- [src/data_loader.py](src/data_loader.py): synthetic regression data generation.
- [src/preprocess.py](src/preprocess.py): train/test split and feature scaling.
- [src/train_model.py](src/train_model.py): baseline and model training utilities.
- [src/evaluate.py](src/evaluate.py): evaluation, metrics export, and summary artifact writing.
- [src/run_pipeline.py](src/run_pipeline.py): end-to-end pipeline orchestration.

Test modules:
- [tests/test_data_loader.py](tests/test_data_loader.py)
- [tests/test_preprocess.py](tests/test_preprocess.py)
- [tests/test_train_model.py](tests/test_train_model.py)
- [tests/test_evaluate.py](tests/test_evaluate.py)

## Methodology
- Data preparation: train and test split with scaled numerical features.
- Models: mean baseline, linear regression, and XGBoost regressor.
- Evaluation: MAE, RMSE, and R2 on holdout data.
- Explainability: linear coefficients and XGBoost feature importances.

## Outputs
- `output/feature_importance.csv`: combined importance table across supported models.
- `output/validation_metrics.csv`: model-wise regression metrics sorted by RMSE.
- `output/summary.json`: evaluated model names, best model by RMSE, and prediction summaries.

## Quick Run
- Install dependencies: `pip install -r requirements.txt`
- Run full pipeline: `python -m src.run_pipeline`
- Run tests: `python -m unittest discover -s tests -v`

## Validation and Error Handling
- `load_data` validates that `n_samples >= 2`.
- `split_and_scale` validates target-column existence, non-empty features, row count, and `test_size` bounds.
- `evaluate_models_and_save_outputs` validates that predictions are non-empty, length-aligned with ground truth, and finite.

## Limitations
- Current pipeline uses a synthetic dataset scaffold for deterministic demonstration.
- Production deployment requires validation against real market data and drift monitoring.
- Additional model comparison and hyperparameter optimization can improve robustness.