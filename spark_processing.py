from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

SALES_PATH = str(DATA_DIR / "sales.csv")
CATALOG_PATH = str(DATA_DIR / "catalog.csv")
SPARK_OUTPUT_DIR = str(ARTIFACTS_DIR / "spark_output")


def get_spark():
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("SuperFresh-BigData")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def run_spark_pipeline() -> None:
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    spark = get_spark()
    print("Spark UI:", spark.sparkContext.uiWebUrl)

    sales_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(SALES_PATH)
    )
    catalog_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(CATALOG_PATH)
    )

    print(f"Filas cargadas (ventas): {sales_df.count():,}")
    print(f"Filas cargadas (catálogo): {catalog_df.count():,}")
    sales_df.printSchema()

    sales_df = sales_df.withColumn("date", F.to_date("date"))

    sales_df = (
        sales_df
        .withColumn("day_of_week", F.dayofweek("date"))
        .withColumn("month", F.month("date"))
        .withColumn("is_weekend", F.when(F.dayofweek("date").isin(1, 7), 1).otherwise(0))
    )

    window_store_product = (
        Window
        .partitionBy("store_id", "product_id")
        .orderBy("date")
    )

    sales_df = (
        sales_df
        .withColumn("lag_1", F.lag("units_sold", 1).over(window_store_product))
        .withColumn("lag_7", F.lag("units_sold", 7).over(window_store_product))
        .withColumn(
            "rolling_7",
            F.avg("units_sold").over(
                window_store_product.rowsBetween(-7, -1)
            ),
        )
        .withColumn(
            "rolling_14",
            F.avg("units_sold").over(
                window_store_product.rowsBetween(-14, -1)
            ),
        )
    )

    median_val = sales_df.approxQuantile("units_sold", [0.5], 0.01)[0]

    sales_df = (
        sales_df
        .fillna({
            "lag_1": median_val,
            "lag_7": median_val,
            "rolling_7": median_val,
            "rolling_14": median_val,
        })
    )

    by_store = (
        sales_df
        .groupBy("store_id")
        .agg(
            F.sum("units_sold").alias("total_units"),
            F.avg("units_sold").alias("avg_units"),
            F.sum(F.col("units_sold") * F.col("price")).alias("total_revenue"),
        )
        .orderBy(F.desc("total_revenue"))
    )

    by_category_month = (
        sales_df
        .groupBy("category", "month")
        .agg(
            F.sum("units_sold").alias("total_units"),
            F.avg("temperature").alias("avg_temp"),
        )
        .orderBy("category", "month")
    )

    stock_risk = (
        sales_df
        .withColumn("stock_ratio", F.col("stock_available") / (F.col("units_sold") + F.lit(1)))
        .groupBy("store_id", "product_name", "category")
        .agg(F.avg("stock_ratio").alias("avg_stock_ratio"))
        .filter(F.col("avg_stock_ratio") < 1.2)
        .orderBy("avg_stock_ratio")
    )

    print("\n── Ventas por tienda ──")
    by_store.show()
    print("\n── Top productos con riesgo de rotura ──")
    stock_risk.show(10)

    output = Path(SPARK_OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)

    (
        sales_df
        .write
        .mode("overwrite")
        .partitionBy("store_id")
        .parquet(str(output / "sales_features"))
    )

    (
        by_store
        .write
        .mode("overwrite")
        .parquet(str(output / "store_summary"))
    )

    (
        by_category_month
        .write
        .mode("overwrite")
        .parquet(str(output / "category_month_summary"))
    )

    print(f"\nResultados guardados en: {SPARK_OUTPUT_DIR}")

    spark.stop()
    print("Pipeline Spark completado correctamente.")


if __name__ == "__main__":
    run_spark_pipeline()