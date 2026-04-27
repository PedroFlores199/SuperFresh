from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import requests

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
    """Generador sintético de clima usado como respaldo."""
    temp = 17 + 9 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 2.5)
    rain_probability = 0.20 + 0.15 * np.cos(2 * np.pi * day_of_year / 365)
    rainfall = np.random.gamma(1.5, 3.0) if random.random() < rain_probability else 0.0
    return round(float(temp), 1), round(float(rainfall), 1)


def fetch_real_weather(start_date: str = "2025-01-01", periods: int = 240) -> dict:
  
    end_date = pd.date_range(start_date, periods=periods, freq="D")[-1].strftime("%Y-%m-%d")
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude=40.4168&longitude=-3.7038"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,precipitation_sum"
        f"&timezone=Europe/Madrid"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()["daily"]
        weather = {}
        for date_str, temp, rain in zip(
            data["time"],
            data["temperature_2m_max"],
            data["precipitation_sum"],
        ):
            weather[date_str] = (
                round(float(temp if temp is not None else 20.0), 1),
                round(float(rain if rain is not None else 0.0), 1),
            )
        print(f"[Open-Meteo] Datos meteorológicos reales obtenidos: {len(weather)} días (Madrid)")
        return weather
    except Exception as exc:
        print(f"[Open-Meteo] No se pudo obtener el clima real, usando generador sintético: {exc}")
        return {}


def generate_sales(start_date: str = "2025-01-01", periods: int = 240) -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)

    dates = pd.date_range(start_date, periods=periods, freq="D")

    real_weather = fetch_real_weather(start_date, periods)

    rows: list[dict] = []
    catalog = generate_catalog()

    for date in dates:
        day_of_year = int(date.dayofyear)
        date_str = date.strftime("%Y-%m-%d")
        weekday = int(date.weekday())
        is_weekend = 1 if weekday >= 5 else 0
        special_event = 1 if date.day in {1, 15, 28} or (date.month == 12 and date.day >= 20) else 0

        if date_str in real_weather:
            temperature, rainfall = real_weather[date_str]
        else:
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
                        "weather_source": "real" if date_str in real_weather else "synthetic",
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
    real_count = (sales["weather_source"] == "real").sum() if "weather_source" in sales.columns else 0
    print(f"Registros con clima real: {real_count:,} | Sintético: {len(sales) - real_count:,}")


if __name__ == "__main__":
    main()
