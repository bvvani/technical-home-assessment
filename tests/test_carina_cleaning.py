"""Tests for the Carina Cleaning Module."""

from __future__ import annotations

import math
from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from config.pipeline_config import PI
from src.carina_cleaning import (
    clean_data,
    impute_missing_values,
    remove_outliers_iqr,
    validate_physical_ranges,
)


class TestImputeMissingValues:
    """Tests for null handling and median imputation."""

    def test_drops_null_timestamp(self, sample_raw_df):
        """Rows with null timestamp should be removed."""
        result = impute_missing_values(sample_raw_df)

        null_ts = result.filter(
            F.col("timestamp").isNull()
        ).count()

        assert null_ts == 0

    def test_imputes_null_wind_speed(self, sample_raw_df):
        """Null wind_speed should be replaced with turbine median."""
        result = impute_missing_values(sample_raw_df)

        null_ws = result.filter(
            F.col("wind_speed").isNull()
        ).count()

        assert null_ws == 0

    def test_imputes_null_power_output(self, sample_raw_df):
        """Null power_output should be replaced with turbine median."""
        result = impute_missing_values(sample_raw_df)

        null_po = result.filter(
            F.col("power_output").isNull()
        ).count()

        assert null_po == 0

    def test_preserves_clean_rows(self, sample_raw_df):
        """Clean rows should survive imputation unchanged."""

        result = impute_missing_values(sample_raw_df)

        row = result.filter(
            (F.col("turbine_id") == 1)
            & (F.hour(F.col("timestamp")) == 0)
        ).first()

        assert row is not None
        assert row["wind_speed"] == 12.0


class TestValidatePhysicalRanges:
    """Tests for physical range filtering."""

    def test_removes_negative_wind_speed(self, spark):
        """Negative wind speed should be removed."""

        schema = StructType(
            [
                StructField("timestamp", TimestampType()),
                StructField("turbine_id", IntegerType()),
                StructField("wind_speed", DoubleType()),
                StructField("wind_direction", DoubleType()),
                StructField("power_output", DoubleType()),
                StructField("source_file", StringType()),
            ]
        )

        data = [
            (datetime(2022, 3, 1), 1, -5.0, 180.0, 2.5, "g1.csv"),
            (datetime(2022, 3, 1), 1, 12.0, 180.0, 2.5, "g1.csv"),
        ]

        df = spark.createDataFrame(data, schema)

        result = validate_physical_ranges(df)

        assert result.count() == 1

    def test_removes_extreme_power(self, spark):
        """Power output above 15 MW should be removed."""

        schema = StructType(
            [
                StructField("timestamp", TimestampType()),
                StructField("turbine_id", IntegerType()),
                StructField("wind_speed", DoubleType()),
                StructField("wind_direction", DoubleType()),
                StructField("power_output", DoubleType()),
                StructField("source_file", StringType()),
            ]
        )

        data = [
            (datetime(2022, 3, 1), 1, 12.0, 180.0, 99.0, "g1.csv"),
            (datetime(2022, 3, 1), 1, 12.0, 180.0, 3.0, "g1.csv"),
        ]

        df = spark.createDataFrame(data, schema)

        result = validate_physical_ranges(df)

        assert result.count() == 1
        assert result.first()["power_output"] == 3.0


class TestRemoveOutliersIQR:
    """Tests for IQR-based outlier removal."""

    def test_uses_pi_as_multiplier(self):
        """Verify the default multiplier is pi."""

        from config.pipeline_config import IQR_MULTIPLIER

        assert abs(IQR_MULTIPLIER - math.pi) < 1e-10

    def test_retains_normal_values(self, sample_clean_df):
        """Normal values should survive IQR filtering."""

        result = remove_outliers_iqr(sample_clean_df)

        assert result.count() > 0


class TestFullCleanPipeline:
    """Integration test for the full cleaning pipeline."""

    def test_end_to_end(self, sample_raw_df):
        """Full pipeline should produce fewer rows with no nulls."""

        result = clean_data(sample_raw_df)

        for col in [
            "timestamp",
            "turbine_id",
            "wind_speed",
            "power_output",
        ]:
            assert (
                result.filter(F.col(col).isNull()).count() == 0
            )

        assert result.count() < sample_raw_df.count()
