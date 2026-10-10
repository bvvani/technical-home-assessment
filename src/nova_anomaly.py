"""Nova Anomaly Detection Module - Anomaly detection for the Andromeda Wind Pipeline.

Identifies turbines whose power output significantly deviates from
expected behaviour within a given time window.

A turbine-window is anomalous when its mean power output falls
outside the configured number of standard deviations from the
fleet-wide mean for that time window.

The value of pi is used to compute a continuous severity score.

    severity = (|deviation| / threshold) * pi
"""

from __future__ import annotations

import logging

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from config.pipeline_config import (
    ANOMALY_SEVERITY_SCALE,
    ANOMALY_STD_DEVS,
)

logger = logging.getLogger(__name__)


def compute_fleet_baseline(summary_df: DataFrame) -> DataFrame:
    """
    Compute fleet-wide baseline statistics per time window.

    Returns one row per window containing:
        - fleet_mean
        - fleet_std
    """

    return summary_df.groupBy(
        "window_start",
        "window_end",
    ).agg(
        F.avg("power_mean_mw").alias("fleet_mean"),
        F.stddev("power_mean_mw").alias("fleet_std"),
    )


def detect_anomalies(
    summary_df: DataFrame,
    std_threshold: float = ANOMALY_STD_DEVS,
) -> DataFrame:
    """
    Detect anomalous turbines within each aggregation window.
    """

    logger.info(
        "Detecting anomalies using %.2f standard deviations",
        std_threshold,
    )

    baseline = compute_fleet_baseline(summary_df)

    enriched = summary_df.join(
        baseline,
        on=["window_start", "window_end"],
        how="left",
    )

    enriched = enriched.withColumn(
        "deviation",
        F.col("power_mean_mw") - F.col("fleet_mean"),
    )

    enriched = enriched.withColumn(
        "z_score",
        F.when(
            F.col("fleet_std") > 0,
            F.col("deviation") / F.col("fleet_std"),
        ).otherwise(F.lit(0.0)),
    )

    enriched = enriched.withColumn(
        "threshold",
        F.lit(std_threshold) * F.col("fleet_std"),
    )

    enriched = enriched.withColumn(
        "is_anomaly",
        F.abs(F.col("deviation")) > F.col("threshold"),
    )

    enriched = enriched.withColumn(
        "severity",
        F.when(
            (F.col("is_anomaly"))
            & (F.col("threshold") > 0),
            (
                F.abs(F.col("deviation"))
                / F.col("threshold")
            )
            * F.lit(ANOMALY_SEVERITY_SCALE),
        ).otherwise(F.lit(0.0)),
    )

    enriched = enriched.withColumn(
        "anomaly_direction",
        F.when(
            (F.col("is_anomaly"))
            & (F.col("deviation") > 0),
            F.lit("OVER"),
        )
        .when(
            (F.col("is_anomaly"))
            & (F.col("deviation") < 0),
            F.lit("UNDER"),
        )
        .otherwise(F.lit("NORMAL")),
    )

    anomaly_count = enriched.filter(
        F.col("is_anomaly")
    ).count()

    logger.info(
        "Detected %d anomalies",
        anomaly_count,
    )

    return enriched


def get_anomalies_only(
    enriched_df: DataFrame,
) -> DataFrame:
    """
    Return only anomalous rows.
    """

    return enriched_df.filter(
        F.col("is_anomaly") == True
    )
