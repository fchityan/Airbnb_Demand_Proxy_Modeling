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

## Methodology
- Data preparation: train/test split with standardized numerical features.
- Models trained: linear regression and XGBoost regressor.
- Prediction approaches evaluated: mean baseline, linear regression, and XGBoost.
- Evaluation metrics: MAE, RMSE, and R2 on holdout data.
- Explainability: linear coefficients and XGBoost feature importances.

## Pipeline Architecture
The workflow is orchestrated through [src/run_pipeline.py](src/run_pipeline.py), which chains together modular components:
- **data_loader**: generates synthetic regression data with 12 features.
- **preprocess**: splits train/test and standardizes features.
- **train_model**: trains linear regression and XGBoost models.
- **evaluate**: computes metrics and persists artifacts.

Run the complete pipeline with `python -m src.run_pipeline` or call `run_pipeline()` directly from Python with custom parameters (output directory, sample count, test split ratio, random seed).

## Project Structure
- `src/` contains the implementation code used by the pipeline.
- `tests/` contains unit tests that verify the behavior of the matching modules in `src/`.
- `outputs/` contains generated artifacts produced by the pipeline.

The similar filenames in `src/` and `tests/` are intentional. For example, `tests/test_preprocess.py` validates the behavior in `src/preprocess.py`. This is a standard layout that keeps module ownership and test coverage easy to follow.

## Outputs
- `outputs/feature_importance.csv`
  - Purpose: interpret which features drive predictions across models.
  - Columns: `model`, `feature`, `importance`, `coefficient`.
  - Notes: `coefficient` is populated for linear regression and `NaN` for XGBoost rows.
- `outputs/validation_metrics.csv`
  - Purpose: compare model quality on holdout data.
  - Columns: `model`, `mae`, `rmse`, `r2`.
  - Notes: rows are sorted by ascending `rmse`, so the first row is the top performer by RMSE.
- `outputs/summary.json`
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