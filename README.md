# Airbnb Market Analysis

## Executive Summary
This repository presents a machine learning workflow for analyzing Airbnb demand behavior and predicting guest count as a regression problem. The project combines exploratory analysis with a reproducible training and evaluation pipeline to support data-driven market insights.

## Project Objectives
- Build a reliable baseline for guest count prediction.
- Quantify model performance with standard regression metrics.
- Surface the most influential predictors through feature importance.
- Provide a clean, reproducible workflow for reruns and iteration.

## Analytical Scope
The primary notebook is [Airbnb Market Analysis.ipynb](Airbnb Market Analysis.ipynb). Reproducible pipeline components are implemented in [src/data_loader.py](src/data_loader.py), [src/preprocess.py](src/preprocess.py), [src/train_model.py](src/train_model.py), and [src/evaluate.py](src/evaluate.py).

## Methodology
- Data preparation: train and test split with feature scaling.
- Models: mean-line baseline, linear regression, and XGBoost regressor.
- Evaluation: MAE, RMSE, and R2 on holdout data.
- Explainability: coefficient-based influence (linear regression) and feature importances (XGBoost).

## Outputs
- output/feature_importance.csv: combined feature importance table for linear regression and XGBoost.
- output/validation_metrics.csv: regression metrics comparison across mean baseline, linear regression, and XGBoost.
- output/summary.json: summary statistics and best-model snapshot by RMSE.

## Quick Run
- Install dependencies: `pip install -r requirements.txt`
- Run full pipeline: `python -m src.run_pipeline`
- Run tests: `python -m unittest discover -s tests -v`

## Limitations
- Current pipeline uses a synthetic dataset scaffold for deterministic demonstration.
- Production deployment requires validation against real market data and drift monitoring.
- Additional model comparison and hyperparameter optimization can improve robustness.