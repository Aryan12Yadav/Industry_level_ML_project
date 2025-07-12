import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact
)
from src.exception import MyException
from src.logger import logging
from src.constants import TARGET_COLUMN
from src.utils.main_utils import load_object
from src.entity.s3_estimator import Proj1Estimator


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys.exc_info()[2])

    def get_best_model(self) -> Optional[Proj1Estimator]:
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name, model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise MyException(e, sys.exc_info()[2])

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if "Gender" in df.columns:
                df["Gender"] = df["Gender"].map({'Female': 0, 'Male': 1}).astype(int)

            if "_id" in df.columns:
                df.drop("_id", axis=1, inplace=True)

            # One-hot encode with rename for consistent model compatibility
            df = pd.get_dummies(df, drop_first=True)

            df.rename(columns={
                "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
                "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
            }, inplace=True)

            for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
                if col in df.columns:
                    df[col] = df[col].astype(int)

            return df
        except Exception as e:
            raise MyException(e, sys.exc_info()[2])

    def evaluate_model(self) -> EvaluateModelResponse:
        try:
            logging.info("Loading test data for evaluation...")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            x = test_df.drop(TARGET_COLUMN, axis=1)
            y = test_df[TARGET_COLUMN]
            x = self._preprocess(x)

            logging.info("Loading trained model...")
            trained_model = load_object(self.model_trainer_artifact.trained_model_file_path)
            y_pred_trained = trained_model.predict(x)
            trained_model_f1 = f1_score(y, y_pred_trained)
            logging.info(f"Trained model F1 score: {trained_model_f1}")

            best_model_f1 = None
            best_model = self.get_best_model()

            if best_model:
                logging.info("Evaluating existing production (best) model...")
                y_pred_best = best_model.predict(x)
                best_model_f1 = f1_score(y, y_pred_best)
                logging.info(f"Best model F1 score: {best_model_f1}")

            best_score = best_model_f1 if best_model_f1 is not None else 0.0
            is_accepted = trained_model_f1 > best_score
            difference = trained_model_f1 - best_score

            return EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1,
                best_model_f1_score=best_model_f1,
                is_model_accepted=is_accepted,
                difference=difference
            )
        except Exception as e:
            raise MyException(e, sys.exc_info()[2])

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("----- Model Evaluation Started -----")
            evaluation_result = self.evaluate_model()

            return ModelEvaluationArtifact(
                is_model_accepted=evaluation_result.is_model_accepted,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluation_result.difference
            )
        except Exception as e:
            raise MyException(e, sys.exc_info()[2])
