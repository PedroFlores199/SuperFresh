from __future__ import annotations

from pathlib import Path

import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from features import build_inference_frame, build_model_matrix

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "superfresh_model.joblib"
RECOMMENDATIONS_PATH = ARTIFACTS_DIR / "stock_recommendations.csv"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]
catalog = bundle["catalog"]
historical_sales = bundle["historical_sales"]
metrics = bundle["metrics"]

app = FastAPI(title="SuperFresh Demand API", version="1.0")


class PredictionRequest(BaseModel):
    store_id: str = Field(..., examples=["T001"])
    product_id: str = Field(..., examples=["P009"])
    date: str = Field(..., examples=["2025-08-30"])
    price: float | None = Field(None, examples=[0.70])
    promotion: int = Field(0, ge=0, le=1)
    temperature: float = Field(22.0)
    rainfall_mm: float = Field(0.0)
    special_event: int = Field(0, ge=0, le=1)
    stock_available: int = Field(50, ge=0)


class PredictionResponse(BaseModel):
    store_id: str
    product_id: str
    product_name: str
    predicted_demand: float
    recommended_stock: int
    replenishment_units: int
    message: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": metrics["model"], "mae": metrics["mae"], "rmse": metrics["rmse"], "r2": metrics["r2"]}


@app.get("/products")
def products() -> dict:
    return {"products": catalog[["product_id", "product_name", "category", "base_price"]].to_dict(orient="records")}


@app.get("/stores")
def stores() -> dict:
    return {"stores": sorted(historical_sales["store_id"].unique().tolist())}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> dict:
    if payload.store_id not in set(historical_sales["store_id"].unique()):
        raise HTTPException(status_code=404, detail="Tienda no encontrada")
    if payload.product_id not in set(catalog["product_id"].unique()):
        raise HTTPException(status_code=404, detail="Producto no encontrado")

    try:
        frame = build_inference_frame(payload.model_dump(), catalog, historical_sales)
        X, _ = build_model_matrix(frame, feature_columns)
        predicted = float(model.predict(X)[0])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    product_name = catalog.loc[catalog["product_id"] == payload.product_id, "product_name"].iloc[0]
    recommended_stock = int(round(predicted * 1.15 + 0.5))
    replenishment_units = max(0, recommended_stock - payload.stock_available)
    message = "Reponer producto" if replenishment_units > 0 else "Stock suficiente"

    return {
        "store_id": payload.store_id,
        "product_id": payload.product_id,
        "product_name": product_name,
        "predicted_demand": round(predicted, 1),
        "recommended_stock": recommended_stock,
        "replenishment_units": replenishment_units,
        "message": message,
    }


@app.get("/metrics")
def get_metrics() -> dict:
    return metrics


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)
