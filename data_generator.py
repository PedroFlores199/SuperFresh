from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SALES_PATH = DATA_DIR / "sales.csv"
CATALOG_PATH = DATA_DIR / "catalog.csv"

STORES = ["T001", "T002", "T003", "T004", "T005"]

PRODUCTS = [
    {"product_id": "P001", "product_name": "Leche entera", "category": "Lacteos", "base_price": 1.25, "base_demand": 55},
    {"product_id": "P002", "product_name": "Yogur natural", "category": "Lacteos", "base_price": 1.10, "base_demand": 44},
    {"product_id": "P003", "product_name": "Queso fresco", "category": "Lacteos", "base_price": 2.60, "base_demand": 30},
    {"product_id": "P004", "product_name": "Pan de molde", "category": "Panaderia", "base_price": 1.55, "base_demand": 60},
    {"product_id": "P005", "product_name": "Bolleria", "category": "Panaderia", "base_price": 1.80, "base_demand": 35},
    {"product_id": "P006", "product_name": "Manzanas", "category": "Fruta", "base_price": 2.20, "base_demand": 48},
    {"product_id": "P007", "product_name": "Platanos", "category": "Fruta", "base_price": 1.95, "base_demand": 52},
    {"product_id": "P008", "product_name": "Ensalada preparada", "category": "Fresco", "base_price": 2.75, "base_demand": 28},
    {"product_id": "P009", "product_name": "Agua mineral", "category": "Bebidas", "base_price": 0.70, "base_demand": 75},
    {"product_id": "P010", "product_name": "Refresco cola", "category": "Bebidas", "base_price": 1.20, "base_demand": 66},
    {"product_id": "P011", "product_name": "Cafe molido", "category": "Despensa", "base_price": 3.40, "base_demand": 24},
    {"product_id": "P012", "product_name": "Arroz", "category": "Despensa", "base_price": 1.35, "base_demand": 42},
    {"product_id": "P013", "product_name": "Pasta", "category": "Despensa", "base_price": 1.15, "base_demand": 40},
    {"product_id": "P014", "product_name": "Helado", "category": "Congelado", "base_price": 2.90, "base_demand": 26},
    {"product_id": "P015", "product_name": "Pizza congelada", "category": "Congelado", "base_price": 3.10, "base_demand": 34},
]

STORE_FACTOR = {
    "T001": 1.05,
    "T002": 0.92,
    "T003": 1.18,
    "T004": 0.85,
    "T005": 1.00,
}

CATEGORY_SEASONALITY = {
    "Bebidas": 0.75,
    "Congelado": 0.45,
    "Fruta": 0.25,
    "Fresco": 0.20,
    "Panaderia": 0.10,
    "Lacteos": 0.05,
    "Despensa": -0.05,
}


def generate_catalog() -> pd.DataFrame:
    return pd.DataFrame(PRODUCTS)


def weather_for_day(day_of_year: int) -> tuple[float, float]:

    temp = 17 + 9 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 2.5)
    rain_probability = 0.20 + 0.15 * np.cos(2 * np.pi * day_of_year / 365)
    rainfall = np.random.gamma(1.5, 3.0) if random.random() < rain_probability else 0.0
    return round(float(temp), 1), round(float(rainfall), 1)


def generate_sales(start_date: str = "2025-01-01", periods: int = 240) -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)

    dates = pd.date_range(start_date, periods=periods, freq="D")
    rows: list[dict] = []
    catalog = generate_catalog()

    for date in dates:
        day_of_year = int(date.dayofyear)
        weekday = int(date.weekday())
        is_weekend = 1 if weekday >= 5 else 0
        special_event = 1 if date.day in {1, 15, 28} or (date.month == 12 and date.day >= 20) else 0
        temperature, rainfall = weather_for_day(day_of_year)

        for store_id in STORES:
            for _, product in catalog.iterrows():
                category = product["category"]
                base_price = float(product["base_price"])
                base_demand = float(product["base_demand"])

                promotion = 1 if random.random() < 0.17 else 0
                price = base_price * (0.90 if promotion else 1.0)

                seasonal = 1 + CATEGORY_SEASONALITY[category] * np.sin(2 * np.pi * (day_of_year - 165) / 365)
                weekend_boost = 1.18 if is_weekend and category in {"Bebidas", "Panaderia", "Congelado"} else 1.0
                promo_boost = 1.28 if promotion else 1.0
                event_boost = 1.17 if special_event else 1.0
                rain_effect = 1.10 if rainfall > 6 and category in {"Despensa", "Panaderia"} else 1.0
                heat_effect = 1.25 if temperature > 27 and category in {"Bebidas", "Congelado"} else 1.0

                expected = (
                    base_demand
                    * STORE_FACTOR[store_id]
                    * seasonal
                    * weekend_boost
                    * promo_boost
                    * event_boost
                    * rain_effect
                    * heat_effect
                )
                noise = np.random.normal(0, expected * 0.10)
                units_sold = max(0, int(round(expected + noise)))


                stock_available = max(0, int(round(expected * random.uniform(0.75, 1.55))))
                if random.random() < 0.06:
                    stock_available = max(0, int(round(expected * random.uniform(0.20, 0.65))))
                    units_sold = min(units_sold, stock_available)

                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "store_id": store_id,
                        "product_id": product["product_id"],
                        "product_name": product["product_name"],
                        "category": category,
                        "price": round(price, 2),
                        "promotion": promotion,
                        "temperature": temperature,
                        "rainfall_mm": rainfall,
                        "special_event": special_event,
                        "stock_available": stock_available,
                        "units_sold": units_sold,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    catalog = generate_catalog()
    sales = generate_sales()
    catalog.to_csv(CATALOG_PATH, index=False)
    sales.to_csv(SALES_PATH, index=False)
    print(f"Catalogo generado: {CATALOG_PATH} ({len(catalog)} productos)")
    print(f"Ventas generadas: {SALES_PATH} ({len(sales)} filas)")


if __name__ == "__main__":
    main()
