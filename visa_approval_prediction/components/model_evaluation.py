import os
import sys
import pickle
import shutil

import pandas as pd
from sklearn.metrics import f1_score

from visa_approval_prediction.constants import TARGET_COLUMN
from visa_approval_prediction.entity.config_entity import ModelEvaluationConfig
from visa_approval_prediction.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)
from visa_approval_prediction.exception import visaException
from visa_approval_prediction.logger import logging


class ModelEvaluation:
    def __init__(
        self,
        config: ModelEvaluationConfig,
        trainer_artifact: ModelTrainerArtifact,
        ingestion_artifact: DataIngestionArtifact,
    ):
        self.config = config
        self.trainer_artifact = trainer_artifact
        self.ingestion_artifact = ingestion_artifact

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        logging.info("Starting model evaluation")
        try:
            new_model_f1 = self.trainer_artifact.test_f1_score
            best_model_f1 = 0.0

            # Compare with existing production model if it exists
            if os.path.exists(self.config.best_model_path):
                logging.info(
                    f"Existing model found at {self.config.best_model_path}"
                )
                with open(self.config.best_model_path, "rb") as f:
                    existing_model = pickle.load(f)

                # Use raw test data — each visaModel has its own preprocessor
                # so we call predict() on raw DataFrame, not pre-transformed arrays
                test_df = pd.read_csv(self.ingestion_artifact.test_file_path)
                X_test = test_df.drop(columns=[TARGET_COLUMN])
                y_test = test_df[TARGET_COLUMN]

                y_pred = existing_model.predict(X_test)
                best_model_f1 = f1_score(y_test, y_pred)
                logging.info(f"Existing model F1: {best_model_f1:.4f}")
                logging.info(f"New model F1:      {new_model_f1:.4f}")
            else:
                logging.info("No existing model found, new model will be accepted")

            improved = new_model_f1 - best_model_f1
            is_accepted = (
                improved >= self.config.changed_threshold_score
                or best_model_f1 == 0.0
            )

            if is_accepted:
                os.makedirs(
                    os.path.dirname(self.config.best_model_path), exist_ok=True
                )
                shutil.copy2(
                    self.trainer_artifact.trained_model_file_path,
                    self.config.best_model_path,
                )
                logging.info(
                    f"New model promoted to {self.config.best_model_path}"
                )
            else:
                logging.info(
                    f"New model rejected (improvement {improved:.4f} "
                    f"< threshold {self.config.changed_threshold_score})"
                )

            return ModelEvaluationArtifact(
                is_model_accepted=is_accepted,
                best_model_path=self.config.best_model_path,
                trained_model_f1_score=new_model_f1,
                best_model_f1_score=(
                    max(best_model_f1, new_model_f1) if is_accepted else best_model_f1
                ),
            )

        except Exception as e:
            raise visaException(e, sys) from e
