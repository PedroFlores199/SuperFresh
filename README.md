# SuperFresh Big Data - Predicción de demanda

Proyecto preparado para el trabajo de enfoque de Sistemas de Big Data. Implementa un sistema de predicción de ventas para SuperFresh con datos simulados, modelo de Machine Learning, API y cuadro de mando.

## Estructura

- `data_generator.py`: genera datos simulados de ventas, productos, tiendas, promociones, clima y stock.
- `features.py`: crea variables temporales, retardos y medias móviles.
- `train.py`: entrena un modelo `RandomForestRegressor`, calcula MAE, RMSE y R², y genera recomendaciones de reposición.
- `api.py`: API FastAPI para consultar estado, productos, tiendas, métricas y predicciones.
- `dashboard.py`: cuadro de mando en Streamlit.
- `requirements.txt`: dependencias.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución completa

```bash
python data_generator.py
python train.py
```

## Lanzar la API

```bash
uvicorn api:app --reload --port 8001
```

Prueba rápida:

```bash
curl http://127.0.0.1:8001/health
```

Predicción ejemplo:

```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"store_id":"T001","product_id":"P009","date":"2025-08-30","promotion":0,"temperature":28,"rainfall_mm":0,"special_event":0,"stock_available":40}'
```

## Lanzar dashboard

```bash
streamlit run dashboard.py
```

## Resultados esperados

El entrenamiento guarda los resultados en `artifacts/`:

- `superfresh_model.joblib`: modelo entrenado y datos auxiliares.
- `metrics.json`: MAE, RMSE, R² y tamaño de entrenamiento/prueba.
- `test_predictions.csv`: comparación entre ventas reales y predichas.
- `stock_recommendations.csv`: recomendación de stock por tienda y producto.
