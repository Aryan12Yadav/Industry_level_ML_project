import sys
from typing import Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)
from src.entity.estimator import MyModel


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):
        """
        Initialize the ModelTrainer with transformation artifact and trainer config.
        """
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise MyException(e, sys) from e

    def get_model_object_and_report(
        self, train: np.ndarray, test: np.ndarray
    ) -> Tuple[RandomForestClassifier, ClassificationMetricArtifact]:
        """
        Trains RandomForestClassifier and returns the model and evaluation metrics.

        :param train: Transformed training data
        :param test: Transformed test data
        :return: Trained model and evaluation metrics
        """
        try:
            logging.info("Splitting datasets into features and targets.")
            x_train, y_train = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]

            logging.info("Initializing RandomForestClassifier with config hyperparameters...")
            model = RandomForestClassifier(
                n_estimators=self.model_trainer_config._n_estimators,
                min_samples_split=self.model_trainer_config._min_samples_split,
                min_samples_leaf=self.model_trainer_config._min_samples_leaf,
                max_depth=self.model_trainer_config._max_depth,
                criterion=self.model_trainer_config._criterion,
                random_state=self.model_trainer_config._random_state,
            )

            logging.info("Training RandomForestClassifier...")
            model.fit(x_train, y_train)

            logging.info("Predicting on test data...")
            y_pred = model.predict(x_test)

            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            logging.info(f"Test metrics — Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

            metric_artifact = ClassificationMetricArtifact(
                accuracy=accuracy,
                f1_score=f1,
                precision_score=precision,
                recall_score=recall
            )

            return model, metric_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Orchestrates loading data, training model, evaluating metrics,
        validating performance, and saving the model pipeline.

        :return: ModelTrainerArtifact with model path and metrics
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            print("🟡 Starting Model Trainer Component")

            logging.info("Loading transformed train and test data...")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            logging.info("Training model and computing evaluation metrics...")
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)

            logging.info("Loading preprocessing object...")
            preprocessing_obj = load_object(self.data_transformation_artifact.transformed_object_file_path)

            logging.info("Calculating training accuracy...")
            train_accuracy = accuracy_score(train_arr[:, -1], trained_model.predict(train_arr[:, :-1]))
            logging.info(f"Training accuracy: {train_accuracy:.4f}")

            if train_accuracy < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"Model training accuracy {train_accuracy:.4f} is below expected threshold "
                    f"{self.model_trainer_config.expected_accuracy}"
                )

            logging.info("Saving complete model pipeline (preprocessing + model)...")
            my_model = MyModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=trained_model
            )
            save_object(self.model_trainer_config.trained_model_file_path, my_model)

            logging.info("Creating ModelTrainerArtifact...")
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )

            logging.info(f"ModelTrainerArtifact created: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e
