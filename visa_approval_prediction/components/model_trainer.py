import os
import sys
import pickle

import yaml
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

from visa_approval_prediction.entity.config_entity import ModelTrainerConfig
from visa_approval_prediction.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from visa_approval_prediction.entity.estimator import visaModel
from visa_approval_prediction.exception import visaException
from visa_approval_prediction.logger import logging

MODEL_CLASSES = {
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "XGBClassifier": XGBClassifier,
}


class ModelTrainer:
    def __init__(
        self,
        config: ModelTrainerConfig,
        transformation_artifact: DataTransformationArtifact,
    ):
        self.config = config
        self.transformation_artifact = transformation_artifact

    def initiate_model_training(self) -> ModelTrainerArtifact:
        logging.info("Starting model training")
        try:
            # Load transformed + resampled training data
            X_train = np.load(
                self.transformation_artifact.transformed_train_file_path
            )
            y_train = np.load(
                self.transformation_artifact.transformed_train_target_path
            )
            X_test = np.load(
                self.transformation_artifact.transformed_test_file_path
            )
            y_test = np.load(
                self.transformation_artifact.transformed_test_target_path
            )
            logging.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

            # Load preprocessor (needed to bundle into visaModel)
            with open(
                self.transformation_artifact.preprocessor_object_file_path, "rb"
            ) as f:
                preprocessor = pickle.load(f)

            # Load model config
            with open(self.config.model_config_file_path, "r") as f:
                config = yaml.safe_load(f)

            gs_cfg = config["grid_search"]["params"]
            models_cfg = config["model_selection"]

            # Plain GridSearchCV for each model on pre-resampled data
            results = []
            for key in sorted(models_cfg.keys()):
                mcfg = models_cfg[key]
                cls_name = mcfg["class"]
                cls = MODEL_CLASSES[cls_name]
                base_params = dict(mcfg.get("params", {}))
                param_grid = mcfg.get("search_param_grid", {})

                logging.info(f"Training {cls_name} ...")
                estimator = cls(**base_params)

                gs = GridSearchCV(
                    estimator,
                    param_grid=param_grid,
                    cv=gs_cfg["cv"],
                    scoring=gs_cfg["scoring"],
                    verbose=gs_cfg.get("verbose", 0),
                    n_jobs=-1,
                )
                gs.fit(X_train, y_train)

                # Evaluate on unresampled test data
                y_pred_train = gs.best_estimator_.predict(X_train)
                y_pred_test = gs.best_estimator_.predict(X_test)
                train_f1 = f1_score(y_train, y_pred_train)
                test_f1 = f1_score(y_test, y_pred_test)
                test_acc = accuracy_score(y_test, y_pred_test)

                results.append(
                    {
                        "name": cls_name,
                        "best_params": gs.best_params_,
                        "cv_f1": gs.best_score_,
                        "train_f1": train_f1,
                        "test_f1": test_f1,
                        "test_acc": test_acc,
                        "model": gs.best_estimator_,
                    }
                )
                logging.info(
                    f"  {cls_name}: CV F1={gs.best_score_:.4f}, "
                    f"Test F1={test_f1:.4f}, Test Acc={test_acc:.4f}"
                )

            # Select best model by test F1
            best = max(results, key=lambda r: r["test_f1"])
            logging.info(
                f"Best model: {best['name']} (Test F1={best['test_f1']:.4f})"
            )

            # Log classification report
            y_pred_best = best["model"].predict(X_test)
            report = classification_report(
                y_test, y_pred_best, target_names=["Certified", "Denied"]
            )
            logging.info(f"Classification report:\n{report}")

            if best["test_f1"] < self.config.expected_accuracy:
                logging.warning(
                    f"Best F1 ({best['test_f1']:.4f}) below threshold "
                    f"({self.config.expected_accuracy})"
                )

            # Bundle preprocessor + best classifier into visaModel
            visa_model = visaModel(
                preprocessing_object=preprocessor,
                trained_model_object=best["model"],
            )

            os.makedirs(
                os.path.dirname(self.config.trained_model_file_path), exist_ok=True
            )
            with open(self.config.trained_model_file_path, "wb") as f:
                pickle.dump(visa_model, f)
            logging.info(f"Model saved to {self.config.trained_model_file_path}")

            return ModelTrainerArtifact(
                trained_model_file_path=self.config.trained_model_file_path,
                train_f1_score=best["train_f1"],
                test_f1_score=best["test_f1"],
                test_accuracy=best["test_acc"],
                best_model_name=best["name"],
            )

        except Exception as e:
            raise visaException(e, sys) from e
