# AEP Electricity Load Forecasting

A machine learning project comparing four models for hourly electricity load forecasting using historical AEP data. The project evaluates **1-hour-ahead** and **24-hour-ahead** predictions.

## Dataset

- **Source:** AEP hourly electricity consumption data from Kaggle’s *Hourly Energy Consumption* dataset by Rob Mulla
- **Period:** 2004–2018
- **Target:** Electricity load in MW

## Approach

1. Cleaned and organized the hourly time series, including duplicate and missing timestamps.
2. Created calendar and historical load features for forecasting.
3. Trained and evaluated Linear Regression, Random Forest, XGBoost, and LSTM.
4. Compared the models at both forecast horizons using RMSE, MAE, MAPE, and R².
5. Examined feature importance to help explain the tree-based models.

The 24-hour-ahead experiment predicts **one load reading 24 hours after the forecast origin**; it does not produce every intervening hourly reading.

## Results and Dashboard

The Streamlit dashboard displays the saved test-set results, model comparisons, historical actual-versus-predicted plots, and feature-importance charts. The performance tables are calculated from the prediction CSV files, so refer to the dashboard or notebook for the current numerical results.

The date selector displays **historical test predictions**. It does not generate forecasts for arbitrary future dates.

## Project Files

The files are in `Desktop/load-forecasting-power-systems/`:

- `load_forecasting.ipynb` — data preparation, training, evaluation, and analysis
- `dashboard.py` — Streamlit dashboard
- `test_predictions_1h.csv` — saved 1-hour-ahead test predictions
- `test_predictions_24h.csv` — saved 24-hour-ahead test predictions
- `feature_importances.csv` — feature importance data for the dashboard
- `requirements.txt` — libraries needed to run the dashboard

## Run the Dashboard Locally

From the repository root:

```bash
python -m pip install -r Desktop/load-forecasting-power-systems/requirements.txt
python -m streamlit run Desktop/load-forecasting-power-systems/dashboard.py
```

The dashboard uses the saved CSV results, so running it does **not** retrain the models. To reproduce the training experiments, download `AEP_hourly.csv` from the Kaggle dataset, place it where the notebook expects it, install the notebook's additional ML libraries, and run the notebook cells.
