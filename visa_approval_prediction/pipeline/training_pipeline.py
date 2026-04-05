import sys

from visa_approval_prediction.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)
from visa_approval_prediction.components.data_ingestion import DataIngestion
from visa_approval_prediction.components.data_validation import DataValidation
from visa_approval_prediction.components.data_transformation import DataTransformation
from visa_approval_prediction.components.model_trainer import ModelTrainer
from visa_approval_prediction.components.model_evaluation import ModelEvaluation
from visa_approval_prediction.exception import visaException
from visa_approval_prediction.logger import logging


class TrainingPipeline:
    def __init__(self):
        self.pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self):
        config = DataIngestionConfig(self.pipeline_config)
        component = DataIngestion(config)
        return component.initiate_data_ingestion()

    def start_data_validation(self, ingestion_artifact):
        config = DataValidationConfig(self.pipeline_config)
        component = DataValidation(config, ingestion_artifact)
        return component.initiate_data_validation()

    def start_data_transformation(self, ingestion_artifact):
        config = DataTransformationConfig(self.pipeline_config)
        component = DataTransformation(config, ingestion_artifact)
        return component.initiate_data_transformation()

    def start_model_training(self, transformation_artifact):
        config = ModelTrainerConfig(self.pipeline_config)
        component = ModelTrainer(config, transformation_artifact)
        return component.initiate_model_training()

    def start_model_evaluation(self, trainer_artifact, ingestion_artifact):
        config = ModelEvaluationConfig(self.pipeline_config)
        component = ModelEvaluation(config, trainer_artifact, ingestion_artifact)
        return component.initiate_model_evaluation()

    def run(self):
        try:
            print("=" * 60)
            print("VISA APPROVAL PREDICTION - TRAINING PIPELINE")
            print("=" * 60)

            # Stage 1: Data Ingestion
            print("\n[1/5] Data Ingestion ...")
            logging.info(">>> Stage 1: Data Ingestion")
            ingestion_artifact = self.start_data_ingestion()
            print(f"      Train: {ingestion_artifact.train_file_path}")
            print(f"      Test:  {ingestion_artifact.test_file_path}")

            # Stage 2: Data Validation
            print("\n[2/5] Data Validation ...")
            logging.info(">>> Stage 2: Data Validation")
            validation_artifact = self.start_data_validation(ingestion_artifact)
            print(f"      Status: {validation_artifact.message}")
            if not validation_artifact.validation_status:
                raise Exception(
                    f"Data validation failed: {validation_artifact.message}"
                )

            # Stage 3: Data Transformation
            print("\n[3/5] Data Transformation ...")
            logging.info(">>> Stage 3: Data Transformation")
            transformation_artifact = self.start_data_transformation(
                ingestion_artifact
            )
            print(f"      Preprocessor: {transformation_artifact.preprocessor_object_file_path}")

            # Stage 4: Model Training
            print("\n[4/5] Model Training (this may take a while) ...")
            logging.info(">>> Stage 4: Model Training")
            trainer_artifact = self.start_model_training(transformation_artifact)
            print(f"      Best model:  {trainer_artifact.best_model_name}")
            print(f"      Test Acc:    {trainer_artifact.test_accuracy:.4f}")
            print(f"      Test F1:     {trainer_artifact.test_f1_score:.4f}")

            # Stage 5: Model Evaluation
            print("\n[5/5] Model Evaluation ...")
            logging.info(">>> Stage 5: Model Evaluation")
            evaluation_artifact = self.start_model_evaluation(
                trainer_artifact, ingestion_artifact
            )
            print(f"      New model F1:      {evaluation_artifact.trained_model_f1_score:.4f}")
            print(f"      Existing model F1: {evaluation_artifact.best_model_f1_score:.4f}")
            print(f"      Accepted: {evaluation_artifact.is_model_accepted}")
            print(f"      Model:    {evaluation_artifact.best_model_path}")

            print("\n" + "=" * 60)
            print("PIPELINE COMPLETE")
            print("=" * 60)

            logging.info("Training pipeline finished successfully")
            return evaluation_artifact

        except Exception as e:
            logging.error(f"Training pipeline failed: {e}")
            raise visaException(e, sys) from e


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
