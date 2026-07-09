# Airbnb Demand Proxy Modeling

## Executive Summary
This repository demonstrates a production-minded machine learning workflow for demand proxy modeling using Airbnb-style data structures. It combines exploratory analysis in a notebook with a modular Python pipeline, automated evaluation outputs, and unit tests.

This project highlights:
- End-to-end ML pipeline design with clear module boundaries.
- Baseline and model-based regression benchmarking.
- Reproducibility through scripted execution and tests.
- Model interpretability artifacts for both linear and tree-based approaches.

## Project Objectives
- Build demand proxy predictions using baseline and ML models.
- Compare model quality using MAE, RMSE, and R2.
- Export explainability artifacts for model interpretation.
- Keep the workflow reproducible and testable.

## Analytical Scope
This project focuses on the machine learning workflow itself, from dataset creation through evaluation and artifact generation.

In scope:
- Exploratory notebook analysis in [Airbnb Demand Proxy Modeling.ipynb](Airbnb%20Demand%20Proxy%20Modeling.ipynb).
- Reproducible pipeline modules:
  - [src/data_loader.py](src/data_loader.py): synthetic regression data generation.
  - [src/preprocess.py](src/preprocess.py): train/test split and feature scaling.
  - [src/train_model.py](src/train_model.py): baseline and model training utilities.
  - [src/evaluate.py](src/evaluate.py): metric computation and output persistence.
  - [src/run_pipeline.py](src/run_pipeline.py): end-to-end orchestration.
- Test coverage for data loading, preprocessing, training, and evaluation:
  - [tests/test_data_loader.py](tests/test_data_loader.py)
  - [tests/test_preprocess.py](tests/test_preprocess.py)
  - [tests/test_train_model.py](tests/test_train_model.py)
  - [tests/test_evaluate.py](tests/test_evaluate.py)

## Methodology
- Data preparation: train/test split with standardized numerical features.
- Models trained: linear regression and XGBoost regressor.
- Prediction approaches evaluated: mean baseline, linear regression, and XGBoost.
- Evaluation metrics: MAE, RMSE, and R2 on holdout data.
- Explainability: linear coefficients and XGBoost feature importances.

## Outputs
- `output/feature_importance.csv`
  - Purpose: interpret which features drive predictions across models.
  - Columns: `model`, `feature`, `importance`, `coefficient`.
  - Notes: `coefficient` is populated for linear regression and `NaN` for XGBoost rows.
- `output/validation_metrics.csv`
  - Purpose: compare model quality on holdout data.
  - Columns: `model`, `mae`, `rmse`, `r2`.
  - Notes: rows are sorted by ascending `rmse`, so the first row is the top performer by RMSE.
- `output/summary.json`
  - Purpose: provide a quick, machine-readable report for downstream use.
  - Includes: evaluated models list, best model by RMSE, actual-target summary statistics, and per-model prediction summaries.

## Quick Run
- Install dependencies:
  - `pip install -r requirements.txt`
- Run full pipeline:
  - `python -m src.run_pipeline`
- Run tests:
  - `python -m unittest discover -s tests -v`

## Validation and Error Handling
- `load_data` validates that `n_samples >= 2`.
- `split_and_scale` validates target-column existence, non-empty features, minimum row count, and `test_size` bounds.
- `evaluate_models_and_save_outputs` validates that predictions are non-empty, aligned to ground-truth length, and finite.

## Limitations
- Current pipeline uses synthetic data as a deterministic scaffold.
- Production deployment requires real data ingestion, quality checks, and monitoring.
- Additional model comparison and hyperparameter optimization can improve robustness.