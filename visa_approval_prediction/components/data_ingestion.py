import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from visa_approval_prediction.constants import CURRENT_YEAR, TARGET_COLUMN
from visa_approval_prediction.entity.config_entity import DataIngestionConfig
from visa_approval_prediction.entity.artifact_entity import DataIngestionArtifact
from visa_approval_prediction.exception import visaException
from visa_approval_prediction.logger import logging


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Starting data ingestion")
        try:
            df = pd.read_csv(self.config.data_source_path)
            logging.info(f"Loaded dataset: {df.shape}")

            # Feature engineering
            df["company_age"] = CURRENT_YEAR - df["yr_of_estab"]
            df.drop(columns=["case_id", "yr_of_estab"], inplace=True)

            # Encode target: Certified=0, Denied=1
            df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Certified": 0, "Denied": 1})

            # Stratified train/test split
            train_set, test_set = train_test_split(
                df,
                test_size=self.config.split_ratio,
                random_state=42,
                stratify=df[TARGET_COLUMN],
            )
            logging.info(f"Train: {train_set.shape}, Test: {test_set.shape}")

            # Save splits
            os.makedirs(os.path.dirname(self.config.train_file_path), exist_ok=True)
            train_set.to_csv(self.config.train_file_path, index=False)
            test_set.to_csv(self.config.test_file_path, index=False)
            logging.info(f"Train saved to {self.config.train_file_path}")
            logging.info(f"Test saved to {self.config.test_file_path}")

            return DataIngestionArtifact(
                train_file_path=self.config.train_file_path,
                test_file_path=self.config.test_file_path,
            )

        except Exception as e:
            raise visaException(e, sys) from e
