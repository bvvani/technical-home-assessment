"""Andromeda Wind Pipeline - Main Orchestrator.

Project Codename: Andromeda

Phase Codenames:
    Lyra    -> Ingestion
    Carina  -> Cleaning
    Orion   -> Statistics
    Nova    -> Anomaly Detection
    Nebula  -> Storage

This script orchestrates the full pipeline end-to-end:

    1. Ingest raw CSV data from turbine sensor groups
    2. Clean and validate the data
    3. Compute summary statistics per turbine per time window
    4. Detect anomalous turbine behaviour
    5. Persist all artefacts

Usage:

    python main.py \
        --data-dir data/raw \
        --no-delta \
        --output-path output
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

from pyspark.sql import SparkSession

from src.lyra_ingestion import ingest_all_groups
from src.carina_cleaning import clean_data
from src.orion_statistics import compute_summary_statistics
from src.nova_anomaly import (
    detect_anomalies,
    get_anomalies_only,
)
from src.nebula_storage import (
    ensure_schema_exists,
    store_all,
    write_delta_path,
)

from config.pipeline_config import (
    AGGREGATION_WINDOW_HOURS,
    DELTA_CATALOG,
    DELTA_SCHEMA,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger("andromeda")


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------

def run_pipeline(
    spark: SparkSession,
    data_dir: str,
    catalog: str = DELTA_CATALOG,
    schema: str = DELTA_SCHEMA,
    window_hours: int = AGGREGATION_WINDOW_HOURS,
    write_to_delta: bool = True,
    delta_output_path: Optional[str] = None,
) -> dict:
    """
    Execute the full pipeline.
    """

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("ANDROMEDA WIND PIPELINE - START")
    logger.info("Data directory : %s", data_dir)
    logger.info("Window hours   : %s", window_hours)
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Phase 1 - Lyra
    # -----------------------------------------------------------------------

    logger.info("[1/5] LYRA - Ingesting data")

    raw_df = ingest_all_groups(
        spark=spark,
        data_dir=data_dir,
    )

    raw_count = raw_df.count()

    logger.info(
        "Raw records ingested: %d",
        raw_count,
    )

    # -----------------------------------------------------------------------
    # Phase 2 - Carina
    # -----------------------------------------------------------------------

    logger.info("[2/5] CARINA - Cleaning data")

    cleaned_df = clean_data(raw_df)

    cleaned_count = cleaned_df.count()

    logger.info(
        "Cleaned records: %d",
        cleaned_count,
    )

    # -----------------------------------------------------------------------
    # Phase 3 - Orion
    # -----------------------------------------------------------------------

    logger.info("[3/5] ORION - Computing statistics")

    summary_df = compute_summary_statistics(
        cleaned_df,
        window_hours,
    )

    summary_count = summary_df.count()

    logger.info(
        "Summary rows: %d",
        summary_count,
    )

    # -----------------------------------------------------------------------
    # Phase 4 - Nova
    # -----------------------------------------------------------------------

    logger.info("[4/5] NOVA - Detecting anomalies")

    enriched_df = detect_anomalies(summary_df)

    anomalies_df = get_anomalies_only(enriched_df)

    anomaly_count = anomalies_df.count()

    logger.info(
        "Anomalies detected: %d",
        anomaly_count,
    )

    # -----------------------------------------------------------------------
    # Phase 5 - Nebula
    # -----------------------------------------------------------------------

    logger.info("[5/5] NEBULA - Persisting results")

    outputs = {}

    if write_to_delta:
        ensure_schema_exists(
            spark=spark,
            catalog=catalog,
            schema=schema,
        )

        outputs = store_all(
            raw_df=raw_df,
            cleaned_df=cleaned_df,
            summary_df=summary_df,
            anomalies_df=anomalies_df,
            catalog=catalog,
            schema=schema,
        )

    elif delta_output_path:
        logger.info(
            "Writing pipeline outputs as CSV files to %s",
            delta_output_path,
        )

        raw_df.write \
            .mode("overwrite") \
            .option("header", "true") \
            .csv(f"{delta_output_path}/raw")

        cleaned_df.write \
            .mode("overwrite") \
            .option("header", "true") \
            .csv(f"{delta_output_path}/cleaned")

        summary_df.write \
            .mode("overwrite") \
            .option("header", "true") \
            .csv(f"{delta_output_path}/summary")

        anomalies_df.write \
            .mode("overwrite") \
            .option("header", "true") \
            .csv(f"{delta_output_path}/anomalies")

        outputs = {
            "raw": f"{delta_output_path}/raw",
            "cleaned": f"{delta_output_path}/cleaned",
            "summary": f"{delta_output_path}/summary",
            "anomalies": f"{delta_output_path}/anomalies",
        }

        logger.info(
            "CSV output successfully written."
        )

       """ # write_delta_path(
        #     raw_df,
        #     f"{delta_output_path}/raw",
        # )

        # write_delta_path(
        #     cleaned_df,
        #     f"{delta_output_path}/cleaned",
        # )

        # write_delta_path(
        #     summary_df,
        #     f"{delta_output_path}/summary",
        # )

        # write_delta_path(
        #     anomalies_df,
        #     f"{delta_output_path}/anomalies",
        # )

        # outputs = {
        #     "raw": f"{delta_output_path}/raw",
        #     "cleaned": f"{delta_output_path}/cleaned",
        #     "summary": f"{delta_output_path}/summary",
        #     "anomalies": f"{delta_output_path}/anomalies",
        # }"""

    else:
        logger.warning(
            "Storage skipped because Delta and output path are disabled."
        )

    elapsed_seconds = round(
        time.time() - start_time,
        2,
    )

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("Raw Rows      : %d", raw_count)
    logger.info("Cleaned Rows  : %d", cleaned_count)
    logger.info("Summary Rows  : %d", summary_count)
    logger.info("Anomalies     : %d", anomaly_count)
    logger.info("Duration (s)  : %.2f", elapsed_seconds)
    logger.info("=" * 60)

    return {
        "raw_count": raw_count,
        "cleaned_count": cleaned_count,
        "summary_count": summary_count,
        "anomaly_count": anomaly_count,
        "elapsed_seconds": elapsed_seconds,
        "outputs": outputs,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(args=None):
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Andromeda Wind Pipeline",
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing data_group_*.csv files",
    )

    parser.add_argument(
        "--catalog",
        default=DELTA_CATALOG,
    )

    parser.add_argument(
        "--schema",
        default=DELTA_SCHEMA,
    )

    parser.add_argument(
        "--window-hours",
        type=int,
        default=AGGREGATION_WINDOW_HOURS,
    )

    parser.add_argument(
        "--no-delta",
        action="store_true",
        help="Skip Unity Catalog Delta writes",
    )

    parser.add_argument(
        "--output-path",
        default=None,
        help="Write Delta output to local filesystem",
    )

    return parser.parse_args(args)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    """
    Application entry point.
    """

    args = parse_args()

    logger.info("Data Directory : %s", args.data_dir)
    logger.info("Window Hours   : %s", args.window_hours)

    spark = (
        SparkSession.builder
        .appName("Andromeda Wind Pipeline")
        .getOrCreate()
    )

    try:
        run_pipeline(
            spark=spark,
            data_dir=args.data_dir,
            catalog=args.catalog,
            schema=args.schema,
            window_hours=args.window_hours,
            write_to_delta=not args.no_delta,
            delta_output_path=args.output_path,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
