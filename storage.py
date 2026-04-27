from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

SALES_PATH = DATA_DIR / "sales.csv"
CATALOG_PATH = DATA_DIR / "catalog.csv"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
RECOMMENDATIONS_PATH = ARTIFACTS_DIR / "stock_recommendations.csv"

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/superfresh")


def get_engine():
    return create_engine(DB_URL)


def create_schema(engine) -> None:
    schema_sql = text("""
                      CREATE TABLE IF NOT EXISTS catalog
                      (
                          product_id
                          TEXT
                          PRIMARY
                          KEY,
                          product_name
                          TEXT
                          NOT
                          NULL,
                          category
                          TEXT
                          NOT
                          NULL,
                          base_price
                          REAL
                          NOT
                          NULL
                      );

                      CREATE TABLE IF NOT EXISTS sales
                      (
                          id
                          SERIAL
                          PRIMARY
                          KEY,
                          date
                          TEXT
                          NOT
                          NULL,
                          store_id
                          TEXT
                          NOT
                          NULL,
                          product_id
                          TEXT
                          NOT
                          NULL
                          REFERENCES
                          catalog
                      (
                          product_id
                      ),
                          price REAL,
                          promotion INTEGER,
                          temperature REAL,
                          rainfall_mm REAL,
                          special_event INTEGER,
                          stock_available INTEGER,
                          units_sold INTEGER NOT NULL
                          );

                      CREATE INDEX IF NOT EXISTS idx_sales_store_product_date
                          ON sales(store_id, product_id, date);

                      CREATE TABLE IF NOT EXISTS model_metrics
                      (
                          run_id
                          SERIAL
                          PRIMARY
                          KEY,
                          run_date
                          TIMESTAMP
                          NOT
                          NULL
                          DEFAULT
                          CURRENT_TIMESTAMP,
                          model
                          TEXT,
                          mae
                          REAL,
                          rmse
                          REAL,
                          r2
                          REAL,
                          rows_train
                          INTEGER,
                          rows_test
                          INTEGER
                      );

                      CREATE TABLE IF NOT EXISTS stock_recommendations
                      (
                          id
                          SERIAL
                          PRIMARY
                          KEY,
                          date
                          TEXT,
                          store_id
                          TEXT,
                          product_id
                          TEXT,
                          product_name
                          TEXT,
                          category
                          TEXT,
                          stock_available
                          INTEGER,
                          predicted_demand
                          REAL,
                          recommended_stock
                          INTEGER,
                          replenishment_units
                          INTEGER,
                          risk_level
                          TEXT
                      );
                      """)

    with engine.begin() as conn:
        conn.execute(schema_sql)


def load_data_to_db() -> None:
    if not SALES_PATH.exists() or not CATALOG_PATH.exists():
        print("Ejecuta primero: python data_generator.py")
        return

    engine = get_engine()
    create_schema(engine)

    catalog = pd.read_csv(CATALOG_PATH)[["product_id", "product_name", "category", "base_price"]]
    catalog.to_sql("catalog", engine, if_exists="replace", index=False)
    print(f"Catálogo cargado: {len(catalog)} productos")

    sales = pd.read_csv(SALES_PATH)
    sales_cols = [
        "date",
        "store_id",
        "product_id",
        "price",
        "promotion",
        "temperature",
        "rainfall_mm",
        "special_event",
        "stock_available",
        "units_sold",
    ]
    sales[sales_cols].to_sql("sales", engine, if_exists="replace", index=False, chunksize=5000)
    print(f"Ventas cargadas: {len(sales):,} filas")

    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            m = json.load(f)

        insert_metric_sql = text("""
                                 INSERT INTO model_metrics (model, mae, rmse, r2, rows_train, rows_test)
                                 VALUES (:model, :mae, :rmse, :r2, :rows_train, :rows_test)
                                 """)

        with engine.begin() as conn:
            conn.execute(insert_metric_sql, {
                "model": m.get("model"),
                "mae": m.get("mae"),
                "rmse": m.get("rmse"),
                "r2": m.get("r2"),
                "rows_train": m.get("rows_train"),
                "rows_test": m.get("rows_test")
            })
        print("Métricas del modelo almacenadas")

    if RECOMMENDATIONS_PATH.exists():
        recs = pd.read_csv(RECOMMENDATIONS_PATH)
        recs.to_sql("stock_recommendations", engine, if_exists="replace", index=False)
        print(f"Recomendaciones cargadas: {len(recs)} filas")

    engine.dispose()
    print("Datos cargados correctamente en PostgreSQL.")


def query_sales_summary() -> pd.DataFrame:
    engine = get_engine()
    query = """
            SELECT s.store_id,
                   c.category,
                   COUNT(*)                    AS num_records,
                   SUM(s.units_sold)           AS total_units,
                   AVG(s.units_sold)           AS avg_units,
                   SUM(s.units_sold * s.price) AS total_revenue
            FROM sales s
                     JOIN catalog c ON s.product_id = c.product_id
            GROUP BY s.store_id, c.category
            ORDER BY total_revenue DESC \
            """
    df = pd.read_sql_query(query, engine)
    engine.dispose()
    return df


def query_stock_at_risk() -> pd.DataFrame:
    engine = get_engine()
    query = """
            SELECT store_id, \
                   product_name, \
                   category,
                   stock_available, \
                   predicted_demand,
                   replenishment_units, \
                   risk_level
            FROM stock_recommendations
            WHERE risk_level = 'Riesgo de rotura'
            ORDER BY replenishment_units DESC \
            """
    df = pd.read_sql_query(query, engine)
    engine.dispose()
    return df


if __name__ == "__main__":
    load_data_to_db()
    print("\n── Resumen de ventas por tienda y categoría ──")
    print(query_sales_summary().to_string(index=False))
    risk = query_stock_at_risk()
    if not risk.empty:
        print("\n── Productos con riesgo de rotura ──")
        print(risk.to_string(index=False))