"""Tests for the Nova Anomaly Detection Module."""

from __future__ import annotations

import math

import pytest
from pyspark.sql import functions as F

from config.pipeline_config import (
    ANOMALY_SEVERITY_SCALE,
    PI,
)
from src.nova_anomaly import (
    compute_fleet_baseline,
    detect_anomalies,
    get_anomalies_only,
)
from src.orion_statistics import compute_summary_statistics


@pytest.fixture
def summary_with_anomaly(sample_clean_df):
    """Summary stats from sample data that includes an anomalous turbine."""
    return compute_summary_statistics(sample_clean_df)


class TestComputeFleetBaseline:
    """Tests for fleet-wide baseline computation."""

    def test_baseline_has_expected_columns(self, summary_with_anomaly):
        """Baseline should have fleet_mean and fleet_std."""
        result = compute_fleet_baseline(summary_with_anomaly)

        assert "fleet_mean" in result.columns
        assert "fleet_std" in result.columns

    def test_one_baseline_per_window(self, summary_with_anomaly):
        """Should produce one baseline row per time window."""
        result = compute_fleet_baseline(summary_with_anomaly)

        # one day in the sample data
        assert result.count() == 1


class TestDetectAnomalies:
    """Tests for anomaly detection."""

    def test_enriched_has_anomaly_columns(self, summary_with_anomaly):
        """Enriched DataFrame should contain anomaly metadata."""
        result = detect_anomalies(summary_with_anomaly)

        expected = {
            "deviation",
            "z_score",
            "threshold",
            "is_anomaly",
            "severity",
            "anomaly_direction",
        }

        assert expected.issubset(set(result.columns))

    def test_detects_anomalous_turbine(self, summary_with_anomaly):
        """Turbine 3 should be flagged as anomalous."""

        result = detect_anomalies(summary_with_anomaly)

        anomalies = get_anomalies_only(result)

        anomaly_ids = [
            row["turbine_id"]
            for row in anomalies.collect()
        ]

        assert 3 in anomaly_ids

    def test_normal_turbines_not_flagged(self, summary_with_anomaly):
        """Turbines 1 and 2 should not be anomalous."""

        result = detect_anomalies(summary_with_anomaly)

        normal = result.filter(
            (F.col("turbine_id").isin([1, 2]))
            & (F.col("is_anomaly"))
        )

        assert normal.count() == 0

    def test_severity_uses_pi_scaling(self, summary_with_anomaly):
        """Severity score should use pi scaling."""

        assert abs(
            ANOMALY_SEVERITY_SCALE - math.pi
        ) < 1e-10

        result = detect_anomalies(summary_with_anomaly)

        anomalies = get_anomalies_only(result)

        if anomalies.count() > 0:
            row = anomalies.first()

            assert row["severity"] > 0

    def test_anomaly_direction_labels(self, summary_with_anomaly):
        """Anomalies should be labelled OVER or UNDER."""

        result = detect_anomalies(summary_with_anomaly)

        anomalies = get_anomalies_only(result)

        for row in anomalies.collect():
            assert row["anomaly_direction"] in (
                "OVER",
                "UNDER",
            )

    def test_normal_direction_label(self, summary_with_anomaly):
        """Non-anomalous rows should be labelled NORMAL."""

        result = detect_anomalies(summary_with_anomaly)

        normals = result.filter(
            ~F.col("is_anomaly")
        )

        for row in normals.collect():
            assert row["anomaly_direction"] == "NORMAL"


class TestGetAnomaliesOnly:
    """Tests for anomaly filtering utility."""

    def test_returns_only_flagged_rows(self, summary_with_anomaly):
        """All returned rows should have is_anomaly=True."""

        enriched = detect_anomalies(summary_with_anomaly)

        result = get_anomalies_only(enriched)

        for row in result.collect():
            assert row["is_anomaly"] is True
