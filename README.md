# SuperFresh Big Data - Demand forecasting

Project prepared for the Big Data Systems focus assignment. It implements a sales forecasting system for SuperFresh with simulated data, a Machine Learning model, an API and a dashboard.

## Structure

- `data_generator.py`: generates simulated data for sales, products, stores, promotions, weather and stock.
- `features.py`: builds time variables, lag features and rolling averages.
- `train.py`: trains a `RandomForestRegressor` model, computes MAE, RMSE and R², and generates restocking recommendations.
- `spark_processing.py`: batch pipeline in PySpark (aggregates by store and product, window functions) over the CSV files.
- `storage.py`: schema and loading into PostgreSQL with SQLAlchemy, plus summary queries for sales and stock at risk.
- `api.py`: FastAPI API to query status, products, stores, metrics and predictions.
- `dashboard.py`: dashboard in Streamlit.
- `requirements.txt`: dependencies.

## Stack

Python · pandas · scikit-learn · PySpark · FastAPI · Streamlit · PostgreSQL (SQLAlchemy)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Full run

```bash
python data_generator.py
python train.py
```

## Launch the API

```bash
uvicorn api:app --reload --port 8001
```

Quick test:

```bash
curl http://127.0.0.1:8001/health
```

Example prediction:

```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"store_id":"T001","product_id":"P009","date":"2025-08-30","promotion":0,"temperature":28,"rainfall_mm":0,"special_event":0,"stock_available":40}'
```

## Launch the dashboard

```bash
streamlit run dashboard.py
```

## Expected results

Training saves the results to `artifacts/`:

- `superfresh_model.joblib`: trained model and auxiliary data.
- `metrics.json`: MAE, RMSE, R² and training/test size.
- `test_predictions.csv`: comparison between actual and predicted sales.
- `stock_recommendations.csv`: stock recommendation by store and product.
