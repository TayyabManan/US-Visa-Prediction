import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)
from visa_approval_prediction.constants import TARGET_COLUMN
from visa_approval_prediction.entity.config_entity import DataTransformationConfig
from visa_approval_prediction.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from visa_approval_prediction.exception import visaException
from visa_approval_prediction.logger import logging


class DataTransformation:
    def __init__(
        self,
        config: DataTransformationConfig,
        ingestion_artifact: DataIngestionArtifact,
    ):
        self.config = config
        self.ingestion_artifact = ingestion_artifact

    @staticmethod
    def _build_preprocessor() -> ColumnTransformer:
        """Build the ColumnTransformer (must match prediction pipeline expectations)."""
        onehot_cols = ["continent", "unit_of_wage", "region_of_employment"]
        ordinal_cols = [
            "has_job_experience",
            "requires_job_training",
            "full_time_position",
            "education_of_employee",
        ]
        ordinal_categories = [
            ["N", "Y"],
            ["N", "Y"],
            ["N", "Y"],
            ["High School", "Bachelor's", "Master's", "Doctorate"],
        ]
        power_scale_cols = ["no_of_employees", "company_age"]
        scale_only_cols = ["prevailing_wage"]

        return ColumnTransformer(
            transformers=[
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    onehot_cols,
                ),
                (
                    "ordinal",
                    OrdinalEncoder(categories=ordinal_categories),
                    ordinal_cols,
                ),
                (
                    "power_scale",
                    SkPipeline(
                        [
                            ("power", PowerTransformer(method="yeo-johnson")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    power_scale_cols,
                ),
                ("scale", StandardScaler(), scale_only_cols),
            ],
            remainder="drop",
        )

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Starting data transformation")
        try:
            train_df = pd.read_csv(self.ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.ingestion_artifact.test_file_path)

            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]
            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            # Fit preprocessor on training data only (no data leakage)
            preprocessor = self._build_preprocessor()
            logging.info("Fitting preprocessor on training data")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            logging.info(
                f"Transformed — Train: {X_train_transformed.shape}, "
                f"Test: {X_test_transformed.shape}"
            )

            # Train on natural distribution (no resampling)
            logging.info(
                f"Training on natural distribution: {X_train_transformed.shape[0]} samples"
            )

            # Save transformed arrays
            for path in [
                self.config.transformed_train_file_path,
                self.config.preprocessor_object_file_path,
            ]:
                os.makedirs(os.path.dirname(path), exist_ok=True)

            np.save(self.config.transformed_train_file_path, X_train_transformed)
            np.save(self.config.transformed_test_file_path, X_test_transformed)
            np.save(self.config.transformed_train_target_path, np.array(y_train))
            np.save(self.config.transformed_test_target_path, np.array(y_test))

            # Save preprocessor
            with open(self.config.preprocessor_object_file_path, "wb") as f:
                pickle.dump(preprocessor, f)
            logging.info(
                f"Preprocessor saved to {self.config.preprocessor_object_file_path}"
            )

            return DataTransformationArtifact(
                transformed_train_file_path=self.config.transformed_train_file_path,
                transformed_test_file_path=self.config.transformed_test_file_path,
                transformed_train_target_path=self.config.transformed_train_target_path,
                transformed_test_target_path=self.config.transformed_test_target_path,
                preprocessor_object_file_path=self.config.preprocessor_object_file_path,
            )

        except Exception as e:
            raise visaException(e, sys) from e
