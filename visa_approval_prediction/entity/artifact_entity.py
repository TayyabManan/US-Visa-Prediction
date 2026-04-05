from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_train_target_path: str
    transformed_test_target_path: str
    preprocessor_object_file_path: str


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_f1_score: float
    test_f1_score: float
    test_accuracy: float
    best_model_name: str


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    best_model_path: str
    trained_model_f1_score: float
    best_model_f1_score: float
