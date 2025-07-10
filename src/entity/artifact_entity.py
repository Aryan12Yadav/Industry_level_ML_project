from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    validation_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float
    accuracy: float  # ✅ Added accuracy here

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metric_artifact: ClassificationMetricArtifact

    @property
    def model_accuracy(self) -> float:
        """
        Returns the accuracy from the metric artifact.
        This allows using `artifact.model_accuracy` safely.
        """
        return self.metric_artifact.accuracy  # ✅ Now accessible

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    changed_accuracy: float
    s3_model_path: str
    trained_model_path: str

@dataclass
class ModelPusherArtifact:
    bucket_name: str
    s3_model_path: str
