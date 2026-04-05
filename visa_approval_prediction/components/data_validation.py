import os
import sys

import yaml
import pandas as pd
from scipy.stats import ks_2samp

from visa_approval_prediction.entity.config_entity import DataValidationConfig
from visa_approval_prediction.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from visa_approval_prediction.exception import visaException
from visa_approval_prediction.logger import logging


class DataValidation:
    def __init__(
        self,
        config: DataValidationConfig,
        ingestion_artifact: DataIngestionArtifact,
    ):
        self.config = config
        self.ingestion_artifact = ingestion_artifact

    def _read_schema(self) -> dict:
        with open(self.config.schema_file_path, "r") as f:
            return yaml.safe_load(f)

    def _validate_columns(self, df: pd.DataFrame, schema: dict) -> bool:
        """Check all expected columns are present after feature engineering."""
        expected = set()
        for col_def in schema["columns"]:
            if isinstance(col_def, dict):
                col_name = list(col_def.keys())[0]
            else:
                col_name = col_def
            expected.add(col_name)

        # Adjust for ingestion-stage feature engineering
        expected.discard("case_id")
        expected.discard("yr_of_estab")
        expected.add("company_age")

        actual = set(df.columns)
        missing = expected - actual

        if missing:
            logging.warning(f"Missing columns: {missing}")
            return False
        return True

    def _detect_drift(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> dict:
        """KS test on numerical columns to detect distribution drift."""
        report = {}
        numerical_cols = train_df.select_dtypes(include=["int64", "float64"]).columns

        for col in numerical_cols:
            stat, p_value = ks_2samp(train_df[col], test_df[col])
            is_drifted = p_value < 0.05
            report[col] = {
                "ks_statistic": float(round(stat, 4)),
                "p_value": float(round(p_value, 4)),
                "drift_detected": is_drifted,
            }
            if is_drifted:
                logging.warning(f"Drift in '{col}' (p={p_value:.4f})")

        return report

    def initiate_data_validation(self) -> DataValidationArtifact:
        logging.info("Starting data validation")
        try:
            train_df = pd.read_csv(self.ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.ingestion_artifact.test_file_path)
            schema = self._read_schema()

            # Column validation
            train_valid = self._validate_columns(train_df, schema)
            test_valid = self._validate_columns(test_df, schema)

            if not (train_valid and test_valid):
                message = "Column validation failed"
                logging.error(message)
                validation_status = False
            else:
                message = "Validation passed"
                validation_status = True

            # Drift detection (warning only, does not fail validation)
            drift_report = self._detect_drift(train_df, test_df)
            drifted_cols = [c for c, r in drift_report.items() if r["drift_detected"]]
            if drifted_cols:
                message += f" | Drift detected in: {drifted_cols}"
                logging.warning(f"Drift found in {len(drifted_cols)} columns")

            # Save drift report
            os.makedirs(os.path.dirname(self.config.drift_report_file_path), exist_ok=True)
            with open(self.config.drift_report_file_path, "w") as f:
                yaml.dump(drift_report, f, default_flow_style=False)
            logging.info(f"Drift report saved to {self.config.drift_report_file_path}")

            return DataValidationArtifact(
                validation_status=validation_status,
                message=message,
                drift_report_file_path=self.config.drift_report_file_path,
            )

        except Exception as e:
            raise visaException(e, sys) from e
