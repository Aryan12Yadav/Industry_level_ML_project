import json
import sys
import os
import pandas as pd
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    def read_data(self, file_path) -> DataFrame:
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            for col in df.select_dtypes(include='object').columns:
                df[col] = df[col].str.strip()

            # Drop both 'id' (user-defined) and '_id' (from MongoDB)
            drop_cols = [self._schema_config.get("drop_columns", ""), "_id"]
            for col in drop_cols:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
                    logging.info(f"Dropped column: {col}")

            return df
        except Exception as e:
            raise MyException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        try:
            expected_columns = set(self._schema_config["columns"].keys()) - {self._schema_config["drop_columns"]}
            actual_columns = set(dataframe.columns)

            missing = expected_columns - actual_columns
            extra = actual_columns - expected_columns

            if missing:
                logging.warning(f"Missing columns: {missing}")
            if extra:
                logging.warning(f"Extra columns: {extra}")

            return not (missing or extra)
        except Exception as e:
            raise MyException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = [
                col for col in self._schema_config["numerical_columns"]
                if col not in dataframe_columns
            ]
            missing_categorical_columns = [
                col for col in self._schema_config["categorical_columns"]
                if col not in dataframe_columns
            ]

            if missing_numerical_columns:
                logging.warning(f"Missing numerical columns: {missing_numerical_columns}")
            if missing_categorical_columns:
                logging.warning(f"Missing categorical columns: {missing_categorical_columns}")

            return not (missing_numerical_columns or missing_categorical_columns)
        except Exception as e:
            raise MyException(e, sys)

    def is_train_test_column_match(self, train_df: DataFrame, test_df: DataFrame) -> bool:
        try:
            if list(train_df.columns) != list(test_df.columns):
                logging.warning("Train and test column names/order do not match.")
                return False
            return True
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            validation_error_msg = ""

            if not self.validate_number_of_columns(train_df):
                validation_error_msg += "Training data column count mismatch. "
            if not self.is_column_exist(train_df):
                validation_error_msg += "Training data column names/types mismatch. "

            if not self.validate_number_of_columns(test_df):
                validation_error_msg += "Testing data column count mismatch. "
            if not self.is_column_exist(test_df):
                validation_error_msg += "Testing data column names/types mismatch. "

            if not self.is_train_test_column_match(train_df, test_df):
                validation_error_msg += "Train and test column names/order mismatch. "

            validation_status = len(validation_error_msg) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg.strip(),
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            os.makedirs(os.path.dirname(self.data_validation_config.validation_report_file_path), exist_ok=True)

            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
                json.dump({
                    "validation_status": validation_status,
                    "message": validation_error_msg.strip()
                }, report_file, indent=4)

            logging.info("Data validation completed successfully.")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys)
