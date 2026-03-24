# Retail Demand Forecasting Dashboard

This project forecasts retail product demand using **PySpark + XGBoost** and presents the results in an interactive **Streamlit dashboard**.

The workflow has two main parts:

1. **Jupyter notebook / modeling pipeline**
   - load retail sales data
   - clean and aggregate data
   - engineer time series features
   - train an XGBoost model
   - evaluate forecasting performance
   - generate 365-day forecasts
   - save output files for the dashboard

2. **Streamlit dashboard**
   - load saved historical and forecast outputs
   - filter by date, store, product ID, and product name
   - visualize historical data and forecasted demand
   - display model metrics and feature importance

---

## Project Structure

```text
retail_forecast_project/
├── app/
│   └── streamlit_app.py
├── artifacts/
│   ├── historical.parquet
│   ├── forecast_next_365_days.parquet
│   ├── metrics.json
│   └── feature_importance.csv
├── notebook/
│   └── retail_forecast.ipynb
├── data/
│   └── sales.csv
├── requirements.txt
└── README.md