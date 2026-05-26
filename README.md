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
- Model: random forest regression baseline.
- Evaluation: MAE, RMSE, and R2 on holdout data.
- Explainability: ranked feature importance from trained model outputs.

## Limitations
- Current pipeline uses a synthetic dataset scaffold for deterministic demonstration.
- Production deployment requires validation against real market data and drift monitoring.
- Additional model comparison and hyperparameter optimization can improve robustness.