import sys

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import Proj1Estimator


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        Initializes ModelPusher with evaluation artifact and configuration.

        :param model_evaluation_artifact: Output from model evaluation stage
        :param model_pusher_config: Configuration for pushing the model to S3
        """
        try:
            self.s3 = SimpleStorageService()
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
            self.proj1_estimator = Proj1Estimator(
                bucket_name=model_pusher_config.bucket_name,
                model_path=model_pusher_config.s3_model_key_path
            )
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Pushes the newly trained model to S3 if accepted by evaluation.

        :return: ModelPusherArtifact containing bucket and model path info
        :raises: MyException on failure
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            logging.info("Uploading new trained model to S3 bucket...")

            # Push model to S3
            self.proj1_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path
            )

            logging.info(f"Model successfully uploaded to S3 at {model_pusher_artifact.s3_model_path}")
            logging.info(f"ModelPusherArtifact created: {model_pusher_artifact}")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys) from e
