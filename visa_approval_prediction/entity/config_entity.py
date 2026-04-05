import os
from datetime import datetime
from visa_approval_prediction.constants import (
    ARTIFACT_DIR,
    DATA_INGESTION_DIR_NAME,
    DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO,
    TRAIN_FILE_NAME,
    TEST_FILE_NAME,
    DATA_VALIDATION_DIR_NAME,
    DATA_VALIDATION_DRIFT_REPORT_DIR,
    DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
    SCHEMA_FILE_PATH,
    DATA_TRANSFORMATION_DIR_NAME,
    DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
    DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
    PREPROCSSING_OBJECT_FILE_NAME,
    MODEL_TRAINER_DIR_NAME,
    MODEL_TRAINER_TRAINED_MODEL_DIR,
    MODEL_TRAINER_TRAINED_MODEL_NAME,
    MODEL_TRAINER_EXPECTED_SCORE,
    MODEL_TRAINER_MODEL_CONFIG_FILE_PATH,
    MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE,
    MODEL_FILE_NAME,
)


class TrainingPipelineConfig:
    def __init__(self, timestamp=None):
        self.timestamp = timestamp or datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.artifact_dir = os.path.join(ARTIFACT_DIR, self.timestamp)


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_source_path = "EasyVisa.csv"
        self.split_ratio = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME
        )
        self.train_file_path = os.path.join(ingestion_dir, TRAIN_FILE_NAME)
        self.test_file_path = os.path.join(ingestion_dir, TEST_FILE_NAME)


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        validation_dir = os.path.join(
            training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME
        )
        self.schema_file_path = SCHEMA_FILE_PATH
        self.drift_report_file_path = os.path.join(
            validation_dir,
            DATA_VALIDATION_DRIFT_REPORT_DIR,
            DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME
        )
        data_dir = os.path.join(
            transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR
        )
        object_dir = os.path.join(
            transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR
        )
        self.transformed_train_file_path = os.path.join(data_dir, "train.npy")
        self.transformed_test_file_path = os.path.join(data_dir, "test.npy")
        self.transformed_train_target_path = os.path.join(data_dir, "train_target.npy")
        self.transformed_test_target_path = os.path.join(data_dir, "test_target.npy")
        self.preprocessor_object_file_path = os.path.join(
            object_dir, PREPROCSSING_OBJECT_FILE_NAME
        )


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        trainer_dir = os.path.join(
            training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path = os.path.join(
            trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_TRAINER_TRAINED_MODEL_NAME
        )
        self.expected_accuracy = MODEL_TRAINER_EXPECTED_SCORE
        self.model_config_file_path = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH


class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.changed_threshold_score = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
        self.best_model_path = os.path.join(ARTIFACT_DIR, MODEL_FILE_NAME)
