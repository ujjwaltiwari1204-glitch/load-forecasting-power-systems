# Electrical Load Forecasting using XGBoost

## Overview
A machine learning project that forecasts hourly electricity demand for the AEP 
power grid using historical consumption data and time-series feature engineering.

## Dataset
- **Source:** Kaggle — Hourly Energy Consumption by Rob Mulla
- **Size:** 121,273 hourly records from 2004 to 2018
- **Features:** Datetime-derived features + lag features

## Approach
1. Loaded and explored 14 years of hourly energy consumption data
2. Performed EDA — identified daily, weekly and seasonal patterns
3. Engineered time features (hour, month, day of week, quarter)
4. Added lag features (24hr, 48hr, 1 week) to capture recent consumption patterns
5. Trained XGBoost regression model on 2004-2014 data
6. Tested on 2015-2018 data (no data leakage)

## Results
| Metric | Score |
|--------|-------|
| RMSE | 980 MW |
| MAE | 728 MW |

## Key Findings
- Daily consumption peaks at 4-6pm when commercial and residential demand overlap
- Summer and winter show highest consumption due to AC and heating
- Lag features (previous 24hr, 48hr, weekly consumption) were critical — 
  reduced RMSE from 1808 to 980 MW
- Time-based train/test split used to prevent data leakage

## Technologies Used
- Python, Pandas, NumPy
- XGBoost
- Matplotlib
- VSCode + Jupyter Notebook

## How to Run
1. Download AEP_hourly.csv from Kaggle (robikscube/hourly-energy-consumption)
2. Place it in the project folder
3. Open load_forecasting.ipynb in VSCode or Jupyter
4. Run all cells in order
