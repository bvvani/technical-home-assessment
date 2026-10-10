"""Tests for the Orion Statistics Module."""

from __future__ import annotations

from datetime import datetime

from pyspark.sql import functions as F

from config.pipeline_config import PI, SWEPT_AREA_M2
from src.orion_statistics import (
    add_date_window,
    compute_summary_statistics,
)


class TestAddDateWindow:
    """Tests for time-window assignment."""

    def test_24h_window_creates_date_columns(self, sample_clean_df):
        """24-hour window should produce date-truncated window columns."""
        result = add_date_window(sample_clean_df, 24)

        assert "window_start" in result.columns
        assert "window_end" in result.columns

    def test_24h_window_start_matches_date(self, sample_clean_df):
        """window_start should be the calendar date."""

        result = add_date_window(sample_clean_df, 24)

        starts = [
            r[0]
            for r in result.select("window_start")
            .distinct()
            .collect()
        ]

        assert len(starts) == 1
        assert starts[0].date() == datetime(2022, 3, 1).date()


class TestComputeSummaryStatistics:
    """Tests for per-turbine summary computation."""

    def test_output_has_expected_columns(self, sample_clean_df):
        """Summary should contain all expected metric columns."""

        result = compute_summary_statistics(sample_clean_df)

        expected = {
            "turbine_id",
            "window_start",
            "window_end",
            "power_min_mw",
            "power_max_mw",
            "power_mean_mw",
            "power_std_mw",
            "wind_speed_min",
            "wind_speed_max",
            "wind_speed_mean",
            "record_count",
            "theoretical_power_mw",
            "capacity_factor",
            "missing_entries",
        }

        assert expected.issubset(set(result.columns))

    def test_one_row_per_turbine_per_window(self, sample_clean_df):
        """Should produce exactly one row per turbine per 24-hour window."""

        result = compute_summary_statistics(sample_clean_df)

        # sample_clean_df contains 3 turbines and 1 day
        assert result.count() == 3

    def test_record_count_matches(self, sample_clean_df):
        """Record count should equal hourly entries per turbine."""

        result = compute_summary_statistics(sample_clean_df)

        turbine_1 = (
            result.filter(F.col("turbine_id") == 1)
            .first()
        )

        assert turbine_1["record_count"] == 24

    def test_power_min_max_ordering(self, sample_clean_df):
        """min <= mean <= max for power output."""

        result = compute_summary_statistics(sample_clean_df)

        for row in result.collect():
            assert row["power_min_mw"] <= row["power_mean_mw"]
            assert row["power_mean_mw"] <= row["power_max_mw"]

    def test_theoretical_power_uses_pi(self, sample_clean_df):
        """Theoretical power should incorporate pi via swept area."""

        assert SWEPT_AREA_M2 > 0

        # radius = 40m -> r² = 1600
        assert abs(SWEPT_AREA_M2 / PI - 1600.0) < 1e-6

    def test_missing_entries_computed(self, sample_clean_df):
        """Missing entries should be zero for a complete 24-hour window."""

        result = compute_summary_statistics(sample_clean_df)

        turbine_1 = (
            result.filter(F.col("turbine_id") == 1)
            .first()
        )

        assert turbine_1["missing_entries"] == 0
