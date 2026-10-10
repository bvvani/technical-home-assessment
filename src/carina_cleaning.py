"""Carina Cleaning Module - Data quality for the Andromeda Wind Pipeline.

Handles:
    1. Null / missing value imputation (median per turbine).
    2. Physical range validation (impossible sensor values).
    3. Outlier removal via IQR fencing with a pi-based multiplier.

The value of pi is used as the IQR fence multiplier (approx. 3.14)
instead of the conventional 1.5, producing wider, physics-aware fences
that retain more natural wind variability while still catching true
sensor anomalies.
"""

from __future__ import annotations

import logging

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from config.pipeline_config import (
    IQR_MULTIPLIER,
    POWER_OUTPUT_MAX,
    POWER_OUTPUT_MIN,
    WIND_DIRECTION_MAX,
    WIND_DIRECTION_MIN,
    WIND_SPEED_MAX,
    WIND_SPEED_MIN,
)

logger = logging.getLogger(__name__)



# 1. Null / Missing-Value Handling
# ---------------------------------------------------------------------------

def impute_missing_values(df: DataFrame) -> DataFrame:
    """Impute missing numeric values with per-turbine medians."""

    before = df.count()

    # Drop rows that cannot be attributed
    df = df.dropna(subset=["timestamp", "turbine_id"])

    numeric_cols = [
        "wind_speed",
        "wind_direction",
        "power_output",
    ]

    median_exprs = [
        F.percentile_approx(col_name, 0.5, 10000).alias(
            f"{col_name}_median"
        )
        for col_name in numeric_cols
    ]

    medians_df = df.groupBy("turbine_id").agg(*median_exprs)

    df = df.join(
        medians_df,
        on="turbine_id",
        how="left",
    )

    for col_name in numeric_cols:
        df = df.withColumn(
            col_name,
            F.coalesce(
                F.col(col_name),
                F.col(f"{col_name}_median"),
            ),
        )

    df = df.drop(
        *[f"{c}_median" for c in numeric_cols]
    )

    after = df.count()

    logger.info(
        "Imputation complete: %d rows in, %d rows out (%d dropped)",
        before,
        after,
        before - after,
    )

    return df


# ---------------------------------------------------------------------------
# 2. Physical Range Validation
# ---------------------------------------------------------------------------

def validate_physical_ranges(df: DataFrame) -> DataFrame:
    """Remove physically impossible sensor readings."""

    before = df.count()

    df_clean = df.filter(
        (F.col("wind_speed").between(
            WIND_SPEED_MIN,
            WIND_SPEED_MAX,
        ))
        & (
            F.col("wind_direction").between(
                WIND_DIRECTION_MIN,
                WIND_DIRECTION_MAX,
            )
        )
        & (
            F.col("power_output").between(
                POWER_OUTPUT_MIN,
                POWER_OUTPUT_MAX,
            )
        )
    )

    after = df_clean.count()

    logger.info(
        "Range validation: %d -> %d rows (%d removed)",
        before,
        after,
        before - after,
    )

    return df_clean


# ---------------------------------------------------------------------------
# 3. IQR Outlier Removal (pi-based multiplier)
# ---------------------------------------------------------------------------

def remove_outliers_iqr(
    df: DataFrame,
    columns: list[str] | None = None,
    multiplier: float = IQR_MULTIPLIER,
) -> DataFrame:
    """Remove outliers using IQR fencing."""

    if columns is None:
        columns = [
            "wind_speed",
            "power_output",
        ]

    before = df.count()

    for col_name in columns:

        quantiles_df = (
            df.groupBy("turbine_id")
            .agg(
                F.percentile_approx(
                    col_name,
                    0.25,
                    10000,
                ).alias("q1"),
                F.percentile_approx(
                    col_name,
                    0.75,
                    10000,
                ).alias("q3"),
            )
            .withColumn(
                "iqr",
                F.col("q3") - F.col("q1"),
            )
            .withColumn(
                "lower_fence",
                F.col("q1") - F.lit(multiplier) * F.col("iqr"),
            )
            .withColumn(
                "upper_fence",
                F.col("q3") + F.lit(multiplier) * F.col("iqr"),
            )
        )

        df = (
            df.join(
                quantiles_df.select(
                    "turbine_id",
                    "lower_fence",
                    "upper_fence",
                ),
                on="turbine_id",
                how="left",
            )
            .filter(
                F.col(col_name).between(
                    F.col("lower_fence"),
                    F.col("upper_fence"),
                )
            )
            .drop(
                "lower_fence",
                "upper_fence",
            )
        )

    after = df.count()

    logger.info(
        "IQR filtering: %d -> %d rows (%d removed)",
        before,
        after,
        before - after,
    )

    return df


# ---------------------------------------------------------------------------
# Full Cleaning Pipeline
# ---------------------------------------------------------------------------

def clean_data(df: DataFrame) -> DataFrame:
    """Execute the full Carina cleaning pipeline."""

    logger.info("=== Carina Cleaning Pipeline START ===")

    df = impute_missing_values(df)
    df = validate_physical_ranges(df)
    df = remove_outliers_iqr(df)

    logger.info("=== Carina Cleaning Pipeline END ===")

    return df
