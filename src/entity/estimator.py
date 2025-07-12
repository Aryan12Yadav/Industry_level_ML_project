import sys
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging


class TargetValueMapping:
    def __init__(self):
        self.yes: int = 0
        self.no: int = 1

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        """
        Returns a reversed dictionary of the target mapping (e.g., {0: 'yes', 1: 'no'}).
        """
        mapping_response = self._asdict()
        return {v: k for k, v in mapping_response.items()}


class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Sklearn Pipeline or any transformer for preprocessing.
        :param trained_model_object: Trained model object (like RandomForestClassifier, etc.).
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> DataFrame:
        """
        Applies preprocessing and returns model predictions for the given DataFrame.

        :param dataframe: Input DataFrame to make predictions on.
        :return: Array-like predictions from the trained model.
        """
        try:
            logging.info("Starting prediction process.")

            # Step 1: Apply preprocessing
            transformed_feature = self.preprocessing_object.transform(dataframe)

            # Step 2: Make predictions
            logging.info("Generating predictions using the trained model.")
            predictions = self.trained_model_object.predict(transformed_feature)

            return predictions

        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys.exc_info()[2]) from e

    def __repr__(self):
        return f"{self.__class__.__name__}(model={type(self.trained_model_object).__name__})"

    def __str__(self):
        return self.__repr__()
