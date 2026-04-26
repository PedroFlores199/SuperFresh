from __future__ import annotations

import pandas as pd

CATEGORICAL_COLUMNS = ["store_id", "product_id", "category"]
NUMERIC_COLUMNS = [
    "price",
    "promotion",
    "temperature",
    "rainfall_mm",
    "special_event",
    "stock_available",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_7",
    "rolling_14",
]
TARGET_COLUMN = "units_sold"


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["day_of_week"] = result["date"].dt.weekday
    result["month"] = result["date"].dt.month
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    return result


def add_demand_history_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values(["store_id", "product_id", "date"]).copy()
    group = result.groupby(["store_id", "product_id"])[TARGET_COLUMN]
    result["lag_1"] = group.shift(1)
    result["lag_7"] = group.shift(7)
    result["rolling_7"] = group.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    result["rolling_14"] = group.transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())


    for column in ["lag_1", "lag_7", "rolling_7", "rolling_14"]:
        result[column] = result[column].fillna(result.groupby(["store_id", "product_id"])[TARGET_COLUMN].transform("median"))
        result[column] = result[column].fillna(result[TARGET_COLUMN].median())
    return result


def build_training_frame(sales: pd.DataFrame) -> pd.DataFrame:
    with_dates = add_date_features(sales)
    return add_demand_history_features(with_dates)


def build_model_matrix(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    base = frame[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS].copy()
    encoded = pd.get_dummies(base, columns=CATEGORICAL_COLUMNS, drop_first=False)
    if feature_columns is None:
        feature_columns = list(encoded.columns)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded, feature_columns


def build_inference_frame(payload: dict, catalog: pd.DataFrame, historical_sales: pd.DataFrame) -> pd.DataFrame:

    product_id = payload["product_id"]
    store_id = payload["store_id"]
    date = pd.to_datetime(payload["date"])

    product = catalog.loc[catalog["product_id"] == product_id]
    if product.empty:
        raise ValueError(f"Producto no encontrado: {product_id}")
    product_row = product.iloc[0]

    history = historical_sales.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history[(history["store_id"] == store_id) & (history["product_id"] == product_id)]
    history = history[history["date"] < date].sort_values("date")

    if history.empty:
        lag_1 = lag_7 = rolling_7 = rolling_14 = float(historical_sales["units_sold"].median())
    else:
        values = history["units_sold"].tolist()
        lag_1 = float(values[-1])
        lag_7 = float(values[-7]) if len(values) >= 7 else float(pd.Series(values).median())
        rolling_7 = float(pd.Series(values[-7:]).mean())
        rolling_14 = float(pd.Series(values[-14:]).mean())

    row = {
        "date": date,
        "store_id": store_id,
        "product_id": product_id,
        "product_name": product_row["product_name"],
        "category": product_row["category"],
        "price": float(product_row["base_price"] if payload.get("price") is None else payload.get("price")),
        "promotion": int(payload.get("promotion", 0)),
        "temperature": float(payload.get("temperature", 20.0)),
        "rainfall_mm": float(payload.get("rainfall_mm", 0.0)),
        "special_event": int(payload.get("special_event", 0)),
        "stock_available": int(payload.get("stock_available", 0)),
        "units_sold": 0,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "rolling_7": rolling_7,
        "rolling_14": rolling_14,
    }
    frame = pd.DataFrame([row])
    frame = add_date_features(frame)
    return frame
