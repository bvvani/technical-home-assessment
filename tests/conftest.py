"""Shared pytest fixtures for the Andromeda Wind Pipeline test suite."""

from __future__ import annotations

from datetime import datetime

import pytest
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for the test suite."""

    builder = (
        SparkSession.builder
        .master("local[*]")
        .appName("Andromeda-Tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    yield spark

    spark.stop()


TURBINE_SCHEMA = StructType(
    [
        StructField("timestamp", TimestampType(), nullable=True),
        StructField("turbine_id", IntegerType(), nullable=True),
        StructField("wind_speed", DoubleType(), nullable=True),
        StructField("wind_direction", DoubleType(), nullable=True),
        StructField("power_output", DoubleType(), nullable=True),
        StructField("source_file", StringType(), nullable=True),
    ]
)


@pytest.fixture
def sample_raw_df(spark):
    """Small raw DataFrame with known nulls, outliers, and clean rows."""

    data = [
        # Clean rows
        (datetime(2022, 3, 1, 0), 1, 12.0, 180.0, 2.5, "data_group_1.csv"),
        (datetime(2022, 3, 1, 1), 1, 11.5, 175.0, 2.3, "data_group_1.csv"),
        (datetime(2022, 3, 1, 2), 1, 13.0, 190.0, 2.8, "data_group_1.csv"),
        (datetime(2022, 3, 1, 0), 2, 10.0, 200.0, 3.0, "data_group_1.csv"),
        (datetime(2022, 3, 1, 1), 2, 9.5, 210.0, 2.9, "data_group_1.csv"),
        (datetime(2022, 3, 1, 2), 2, 10.5, 195.0, 3.1, "data_group_1.csv"),

        # Row with null wind_speed
        (datetime(2022, 3, 1, 3), 1, None, 170.0, 2.4, "data_group_1.csv"),

        # Row with null power_output
        (datetime(2022, 3, 1, 3), 2, 11.0, 205.0, None, "data_group_1.csv"),

        # Row with null timestamp (should be dropped)
        (None, 1, 12.0, 180.0, 2.5, "data_group_1.csv"),

        # Row with impossible values
        (datetime(2022, 3, 1, 4), 1, -5.0, 180.0, 2.5, "data_group_1.csv"),

        # Row with extreme power output
        (datetime(2022, 3, 1, 5), 2, 10.0, 200.0, 99.0, "data_group_1.csv"),
    ]

    return spark.createDataFrame(data, schema=TURBINE_SCHEMA)


@pytest.fixture
def sample_clean_df(spark):
    """
    Small cleaned DataFrame for statistics and anomaly tests.

    Contains 2 turbines x 24 hours to simulate a full daily window.
    Turbine 1 has consistent output.
    Turbine 2 has slightly higher output.
    Turbine 3 contains anomalous low output.
    """

    rows = []

    for hour in range(24):
        ts = datetime(2022, 3, 1, hour)

        rows.append(
            (
                ts,
                1,
                12.0,
                180.0,
                2.5 + (hour % 3) * 0.1,
                "data_group_1.csv",
            )
        )

        rows.append(
            (
                ts,
                2,
                10.0,
                200.0,
                3.0 + (hour % 3) * 0.1,
                "data_group_1.csv",
            )
        )

    for hour in range(24):
        ts = datetime(2022, 3, 1, hour)

        rows.append(
            (
                ts,
                3,
                11.0,
                190.0,
                0.3,
                "data_group_1.csv",
            )
        )

    return spark.createDataFrame(rows, schema=TURBINE_SCHEMA)
