from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import build_model_matrix, build_training_frame

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SALES_PATH = DATA_DIR / "sales.csv"
CATALOG_PATH = DATA_DIR / "catalog.csv"
MODEL_PATH = ARTIFACTS_DIR / "superfresh_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"
RECOMMENDATIONS_PATH = ARTIFACTS_DIR / "stock_recommendations.csv"


def train_model() -> None:
    sales = pd.read_csv(SALES_PATH)
    catalog = pd.read_csv(CATALOG_PATH)

    frame = build_training_frame(sales)
    frame = frame.sort_values("date")

    cutoff_date = pd.to_datetime(frame["date"]).max() - pd.Timedelta(days=30)
    train_frame = frame[pd.to_datetime(frame["date"]) <= cutoff_date].copy()
    test_frame = frame[pd.to_datetime(frame["date"]) > cutoff_date].copy()

    X_train, feature_columns = build_model_matrix(train_frame)
    y_train = train_frame["units_sold"]
    X_test, _ = build_model_matrix(test_frame, feature_columns)
    y_test = test_frame["units_sold"]

    model = RandomForestRegressor(
        n_estimators=220,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    metrics = {
        "model": "RandomForestRegressor",
        "rows_total": int(len(frame)),
        "rows_train": int(len(train_frame)),
        "rows_test": int(len(test_frame)),
        "products": int(catalog["product_id"].nunique()),
        "stores": int(sales["store_id"].nunique()),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2_score(y_test, predictions)), 3),
        "cutoff_date": str(cutoff_date.date()),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "catalog": catalog,
        "historical_sales": sales,
        "metrics": metrics,
    }
    joblib.dump(bundle, MODEL_PATH)

    output = test_frame[["date", "store_id", "product_id", "product_name", "category", "units_sold", "stock_available"]].copy()
    output["predicted_units"] = np.round(predictions, 1)
    output.to_csv(PREDICTIONS_PATH, index=False)

    recommendations = build_stock_recommendations(model, feature_columns, frame, catalog)
    recommendations.to_csv(RECOMMENDATIONS_PATH, index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Entrenamiento completado")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Predicciones guardadas en: {PREDICTIONS_PATH}")
    print(f"Recomendaciones de stock guardadas en: {RECOMMENDATIONS_PATH}")


def build_stock_recommendations(model, feature_columns: list[str], frame: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    latest_date = pd.to_datetime(frame["date"]).max()
    next_day = latest_date + pd.Timedelta(days=1)
    latest_rows = frame[pd.to_datetime(frame["date"]) == latest_date].copy()
    future_rows = latest_rows.copy()
    future_rows["date"] = next_day
    future_rows["promotion"] = 0
    future_rows["special_event"] = 0
    future_rows["stock_available"] = latest_rows["stock_available"].values


    future_rows["lag_1"] = latest_rows["units_sold"].values
    future_rows["lag_7"] = latest_rows["lag_7"].values
    future_rows["rolling_7"] = latest_rows["rolling_7"].values
    future_rows["rolling_14"] = latest_rows["rolling_14"].values

    from features import add_date_features

    future_rows = add_date_features(future_rows)
    X_future, _ = build_model_matrix(future_rows, feature_columns)
    future_rows["predicted_demand"] = np.round(model.predict(X_future), 1)
    future_rows["recommended_stock"] = np.ceil(future_rows["predicted_demand"] * 1.15).astype(int)
    future_rows["replenishment_units"] = (
        future_rows["recommended_stock"] - future_rows["stock_available"]
    ).clip(lower=0).astype(int)
    future_rows["risk_level"] = np.where(
        future_rows["stock_available"] < future_rows["predicted_demand"],
        "Riesgo de rotura",
        "Stock suficiente",
    )
    columns = [
        "date",
        "store_id",
        "product_id",
        "product_name",
        "category",
        "stock_available",
        "predicted_demand",
        "recommended_stock",
        "replenishment_units",
        "risk_level",
    ]
    return future_rows[columns].sort_values(["risk_level", "replenishment_units"], ascending=[False, False])


if __name__ == "__main__":
    train_model()
