import sys
from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact
)


class TrainPipeline:
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion...")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Train path: {artifact.trained_file_path}, Test path: {artifact.test_file_path}")
            return artifact
        except Exception as e:
            raise MyException(e)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation...")
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )
            artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation status: {artifact.validation_status} | Message: {artifact.message}")
            return artifact
        except Exception as e:
            raise MyException(e)

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation...")
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact
            )
            artifact = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed.")
            return artifact
        except Exception as e:
            raise MyException(e)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training...")
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )
            artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model training completed. Accuracy: {artifact.model_accuracy}")
            return artifact
        except Exception as e:
            raise MyException(e)

    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting model evaluation...")
            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            artifact = model_evaluation.initiate_model_evaluation()
            logging.info(f"Model evaluation completed. Model accepted: {artifact.is_model_accepted}")
            return artifact
        except Exception as e:
            raise MyException(e)

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        try:
            logging.info("Starting model pushing...")
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config
            )
            artifact = model_pusher.initiate_model_pusher()
            logging.info(f"Model successfully pushed to {artifact.saved_model_path}")
            return artifact
        except Exception as e:
            raise MyException(e)

    def run_pipeline(self) -> None:
        try:
            logging.info("***** Pipeline execution started *****")

            ingestion_artifact = self.start_data_ingestion()
            validation_artifact = self.start_data_validation(ingestion_artifact)
            transformation_artifact = self.start_data_transformation(ingestion_artifact, validation_artifact)
            trainer_artifact = self.start_model_trainer(transformation_artifact)
            evaluation_artifact = self.start_model_evaluation(ingestion_artifact, trainer_artifact)

            if not evaluation_artifact.is_model_accepted:
                logging.warning("Model rejected during evaluation. Pipeline stopped.")
                return

            self.start_model_pusher(evaluation_artifact)

            logging.info("***** Pipeline execution completed successfully *****")

        except Exception as e:
            logging.error("Pipeline execution failed.")
            raise MyException(e)
