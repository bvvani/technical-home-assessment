"""Nebula Storage Module - Delta table persistence for the Andromeda Wind Pipeline.

Writes processed DataFrames to Delta tables in Unity Catalog (or local
Delta format for standalone / test runs). Supports both batch writes
and incremental merge-upserts.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from pyspark.sql import DataFrame, SparkSession

from config.pipeline_config import (
    DELTA_CATALOG,
    DELTA_SCHEMA,
    TABLE_ANOMALIES,
    TABLE_CLEANED,
    TABLE_RAW,
    TABLE_SUMMARY,
    get_fqn,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema / Catalog Initialisation
# ---------------------------------------------------------------------------

def ensure_schema_exists(
    spark: SparkSession,
    catalog: str = DELTA_CATALOG,
    schema: str = DELTA_SCHEMA,
) -> None:
    """
    Create catalog and schema if they do not already exist.
    """

    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

    logger.info(
        "Ensured catalog=%s schema=%s",
        catalog,
        schema,
    )


# ---------------------------------------------------------------------------
# Generic Delta Writers
# ---------------------------------------------------------------------------

def write_delta(
    df: DataFrame,
    table_name: str,
    mode: str = "overwrite",
    partition_by: Optional[List[str]] = None,
    catalog: str = DELTA_CATALOG,
    schema: str = DELTA_SCHEMA,
) -> str:
    """
    Write a DataFrame to a Delta table.
    """

    fqn = get_fqn(
        table_name,
        catalog,
        schema,
    )

    writer = (
        df.write
        .format("delta")
        .mode(mode)
    )

    if partition_by:
        writer = writer.partitionBy(*partition_by)

    writer.saveAsTable(fqn)

    row_count = df.count()

    logger.info(
        "Wrote %d rows to %s (mode=%s)",
        row_count,
        fqn,
        mode,
    )

    return fqn


def write_delta_path(
    df: DataFrame,
    path: str,
    mode: str = "overwrite",
    partition_by: Optional[List[str]] = None,
) -> str:
    """
    Write a DataFrame to a Delta path.
    """

    writer = (
        df.write
        .format("delta")
        .mode(mode)
    )

    if partition_by:
        writer = writer.partitionBy(*partition_by)

    writer.save(path)

    logger.info(
        "Wrote Delta to path: %s (mode=%s)",
        path,
        mode,
    )

    return path


# ---------------------------------------------------------------------------
# Pipeline-Specific Writers
# ---------------------------------------------------------------------------

def store_raw_data(
    df: DataFrame,
    **kwargs,
) -> str:
    """
    Persist raw turbine readings.
    """

    return write_delta(
        df,
        TABLE_RAW,
        partition_by=["source_file"],
        **kwargs,
    )


def store_cleaned_data(
    df: DataFrame,
    **kwargs,
) -> str:
    """
    Persist cleaned turbine readings.
    """

    return write_delta(
        df,
        TABLE_CLEANED,
        partition_by=["source_file"],
        **kwargs,
    )


def store_summary_statistics(
    df: DataFrame,
    **kwargs,
) -> str:
    """
    Persist summary statistics.
    """

    return write_delta(
        df,
        TABLE_SUMMARY,
        **kwargs,
    )


def store_anomalies(
    df: DataFrame,
    **kwargs,
) -> str:
    """
    Persist anomaly records.
    """

    return write_delta(
        df,
        TABLE_ANOMALIES,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Store Entire Pipeline Output
# ---------------------------------------------------------------------------

def store_all(
    raw_df: DataFrame,
    cleaned_df: DataFrame,
    summary_df: DataFrame,
    anomalies_df: DataFrame,
    catalog: str = DELTA_CATALOG,
    schema: str = DELTA_SCHEMA,
) -> dict:
    """
    Persist all pipeline outputs.
    """

    kwargs = {
        "catalog": catalog,
        "schema": schema,
    }

    return {
        "raw": store_raw_data(raw_df, **kwargs),
        "cleaned": store_cleaned_data(cleaned_df, **kwargs),
        "summary": store_summary_statistics(summary_df, **kwargs),
        "anomalies": store_anomalies(anomalies_df, **kwargs),
    }
