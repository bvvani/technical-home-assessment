"""Orion Statistics Module - Summary statistics for the Andromeda Wind Pipeline."""

from __future__ import annotations

import logging

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from config.pipeline_config import (
    AGGREGATION_WINDOW_HOURS,
    AIR_DENSITY_KG_M3,
    SWEPT_AREA_M2,
    TURBINE_EFFICIENCY,
)

logger = logging.getLogger(__name__)


def add_date_window(
    df: DataFrame,
    window_hours: int = AGGREGATION_WINDOW_HOURS,
) -> DataFrame:
    """Add window_start and window_end columns."""

    if window_hours == 24:
        return (
            df
            .withColumn(
                "window_start",
                F.date_trunc("day", F.col("timestamp"))
            )
            .withColumn(
                "window_end",
                F.date_trunc("day", F.col("timestamp"))
                + F.expr("INTERVAL 1 DAY")
            )
        )

    window_spec = f"{window_hours} hours"

    return (
        df
        .withColumn(
            "_window",
            F.window(F.col("timestamp"), window_spec)
        )
        .withColumn(
            "window_start",
            F.col("_window.start")
        )
        .withColumn(
            "window_end",
            F.col("_window.end")
        )
        .drop("_window")
    )


def compute_summary_statistics(
    df: DataFrame,
    window_hours: int = AGGREGATION_WINDOW_HOURS,
) -> DataFrame:
    """Compute per-turbine summary statistics."""

    logger.info(
        "Computing summary statistics (window=%s)",
        window_hours,
    )

    df = add_date_window(df, window_hours)

    group_cols = [
        "turbine_id",
        "window_start",
        "window_end",
    ]

    summary = (
        df.groupBy(*group_cols)
        .agg(
            F.min("power_output").alias("power_min_mw"),
            F.max("power_output").alias("power_max_mw"),
            F.avg("power_output").alias("power_mean_mw"),
            F.stddev("power_output").alias("power_std_mw"),
            F.min("wind_speed").alias("wind_speed_min"),
            F.max("wind_speed").alias("wind_speed_max"),
            F.avg("wind_speed").alias("wind_speed_mean"),
            F.count("*").alias("record_count"),
        )
    )

    # Fix floating-point precision issue
    summary = summary.withColumn(
        "power_mean_mw",
        F.round(F.col("power_mean_mw"), 6)
    )

    summary = summary.withColumn(
        "theoretical_power_mw",
        (
            F.lit(0.5)
            * F.lit(AIR_DENSITY_KG_M3)
            * F.lit(SWEPT_AREA_M2)
            * F.pow(F.col("wind_speed_mean"), 3)
            * F.lit(TURBINE_EFFICIENCY)
        ) / F.lit(1_000_000)
    )

    summary = summary.withColumn(
        "capacity_factor",
        F.when(
            F.col("theoretical_power_mw") > 0,
            F.col("power_mean_mw")
            / F.col("theoretical_power_mw")
        ).otherwise(F.lit(0.0))
    )

    summary = summary.withColumn(
        "missing_entries",
        F.lit(window_hours) - F.col("record_count")
    )

    logger.info(
        "Summary statistics computed: %d rows",
        summary.count(),
    )

    return summary
