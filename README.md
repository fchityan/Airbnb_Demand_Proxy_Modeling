# Airbnb Market Analysis in Dublin

This project analyzes Airbnb demand patterns in Dublin and builds machine learning models to predict guest count as a proxy for market demand.

The core analysis is in:
- `Airbnb Market Analysis.ipynb`

## Repository Structure

- `Airbnb Market Analysis.ipynb`: End-to-end exploratory and modeling notebook.
- `src/data_loader.py`: Builds a synthetic binary-classification dataset.
- `src/preprocess.py`: Splits train/test sets and scales features.
- `src/train_model.py`: Trains a Random Forest model and computes feature importance.
- `src/evaluate.py`: Computes validation metrics, threshold tuning table, and summary JSON.
- `output/`: Stores generated model artifacts.

## Generated Output Files

The scripted pipeline writes the following files to `output/`:

- `feature_importance.csv`
- `validation_metrics.csv`
- `threshold_tuning.csv`
- `summary.json`

## Generate Output Artifacts

From the project root, run:

```bash
python - <<'PY'
from pathlib import Path

from src.data_loader import load_data
from src.preprocess import split_and_scale
from src.train_model import train_model, build_feature_importance
from src.evaluate import evaluate_and_save_outputs

output_dir = Path('output')
output_dir.mkdir(parents=True, exist_ok=True)

df = load_data(random_state=42, n_samples=2000)
x_train, x_test, y_train, y_test = split_and_scale(df)
model = train_model(x_train, y_train)

feature_importance = build_feature_importance(model, list(x_train.columns))
feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False)

y_proba = model.predict_proba(x_test)[:, 1]
evaluate_and_save_outputs(y_test, y_proba, output_dir)

print('Generated files in output/:')
for p in sorted(output_dir.glob('*')):
	print('-', p.name)
PY
```

## Project Goal

The notebook explores whether listing characteristics, room-type preferences, neighbourhood interests, and trip behavior can explain demand.

Because a clean nightly listing price variable is not available, the analysis is centered on:
- demand exploration (EDA)
- guest count prediction
- model comparison and error analysis

## Methods Used

- Data merge from guest contact and search datasets
- Data quality checks (missing values, duplicates)
- Feature engineering (one-hot encoding for room types)
- Feature engineering (one-hot encoding for neighbourhood selections)
- Exploratory analysis with summary statistics and plots
- Modeling (Mean baseline)
- Modeling (Linear Regression)
- Modeling (XGBoost Regressor)
- Evaluation metric (MAE)
- Evaluation metric (RMSE)
- Evaluation metric ($R^2$)

## Notebook Workflow

1. Problem framing and scope
2. Data loading and merge
3. Data cleaning and imputation
4. Exploratory data analysis
5. Experiment 1: minimum price-filter prediction
6. Experiment 2: guest-count prediction (main target)
7. Baseline and benchmark comparisons
8. Feature importance and error analysis
9. Final takeaways

## Main Insights

- Demand concentration appears strongest in central Dublin neighbourhoods.
- Entire home/apartment and private room preferences dominate search behavior.
- Smaller groups and shorter stays are more common than large-group/long-stay bookings.
- XGBoost performs slightly better than baseline and linear regression for guest-count prediction.
- Predictive signal is present but modest (limited explanatory power).

## Notes and Limitations

- Results depend on the quality/completeness of the source files.
- Price-filter fields are user-entered search preferences, not observed market transaction prices.
- The notebook is most useful for directional market insight, not high-precision forecasting.

## Future Improvements

- Add cross-validation and hyperparameter tuning.
- Add residual plots and model calibration checks.
- Engineer temporal features (seasonality, holidays, weekdays/weekends).
- Add robust handling for extreme outliers in guest counts.