from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"
RECOMMENDATIONS_PATH = ARTIFACTS_DIR / "stock_recommendations.csv"
SALES_PATH = DATA_DIR / "sales.csv"

st.set_page_config(page_title="SuperFresh Big Data", page_icon="🛒", layout="wide")
st.title("SuperFresh - Predicción de demanda y gestión de inventario")

if not METRICS_PATH.exists():
    st.error("No se encuentran los artefactos. Ejecuta primero: python data_generator.py && python train.py")
    st.stop()

metrics = pd.read_json(METRICS_PATH, typ="series")
predictions = pd.read_csv(PREDICTIONS_PATH)
recommendations = pd.read_csv(RECOMMENDATIONS_PATH)
sales = pd.read_csv(SALES_PATH)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Filas analizadas", f"{int(metrics['rows_total']):,}".replace(",", "."))
col2.metric("MAE", f"{float(metrics['mae']):.2f}")
col3.metric("RMSE", f"{float(metrics['rmse']):.2f}")
col4.metric("R²", f"{float(metrics['r2']):.3f}")

st.subheader("Evolución de ventas reales")
store_options = sorted(sales["store_id"].unique())
product_options = sorted(sales["product_name"].unique())
left, right = st.columns(2)
selected_store = left.selectbox("Tienda", store_options)
selected_product = right.selectbox("Producto", product_options, index=product_options.index("Agua mineral") if "Agua mineral" in product_options else 0)

sales_filtered = sales[(sales["store_id"] == selected_store) & (sales["product_name"] == selected_product)].copy()
sales_filtered["date"] = pd.to_datetime(sales_filtered["date"])
sales_filtered = sales_filtered.set_index("date")[["units_sold", "stock_available"]]
st.line_chart(sales_filtered)

st.subheader("Validación del modelo: ventas reales frente a predicción")
pred_filtered = predictions[(predictions["store_id"] == selected_store) & (predictions["product_name"] == selected_product)].copy()
if pred_filtered.empty:
    st.info("No hay registros de validación para esta combinación.")
else:
    pred_filtered["date"] = pd.to_datetime(pred_filtered["date"])
    pred_filtered = pred_filtered.set_index("date")[["units_sold", "predicted_units"]]
    st.line_chart(pred_filtered)

st.subheader("Recomendaciones de reposición para el siguiente día")
rec_store = st.selectbox("Filtrar recomendaciones por tienda", store_options, key="rec_store")
rec_filtered = recommendations[recommendations["store_id"] == rec_store].sort_values("replenishment_units", ascending=False)
st.dataframe(
    rec_filtered[[
        "date",
        "store_id",
        "product_name",
        "category",
        "stock_available",
        "predicted_demand",
        "recommended_stock",
        "replenishment_units",
        "risk_level",
    ]],
    use_container_width=True,
)

st.caption("Sistema demostrativo con datos simulados. En un entorno real se sustituiría por datos históricos de ventas, clima, promociones e inventario de SuperFresh.")
