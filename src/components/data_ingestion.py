import os
import sys
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.proj1_data import Proj1Data


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Loads data from MongoDB, cleans it, and saves it to a CSV file (feature store).
        """
        try:
            logging.info("Exporting data from MongoDB...")
            my_data = Proj1Data()
            dataframe = my_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )

            logging.info(f"Original shape: {dataframe.shape}")

            # Clean column names and object-type string values
            dataframe.columns = dataframe.columns.str.strip()
            for col in dataframe.select_dtypes(include='object').columns:
                dataframe[col] = dataframe[col].str.strip()

            logging.info(f"Cleaned shape: {dataframe.shape}")

            # Save cleaned data to feature store
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False)

            logging.info(f"Saved feature store file at: {feature_store_file_path}")
            return dataframe

        except Exception as e:
            raise MyException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Splits the cleaned dataframe into training and testing sets, and saves them to CSV.
        """
        logging.info("Splitting dataset into train and test sets...")
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42  # Ensures reproducibility
            )

            # Save train and test datasets
            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False)

            logging.info(f"Train data saved at: {self.data_ingestion_config.training_file_path}")
            logging.info(f"Test data saved at: {self.data_ingestion_config.testing_file_path}")

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Orchestrates the data ingestion process: loading, cleaning, splitting, saving.
        """
        logging.info("Initiating data ingestion pipeline...")
        try:
            dataframe = self.export_data_into_feature_store()
            logging.info("Data fetched and cleaned successfully.")

            self.split_data_as_train_test(dataframe)
            logging.info("Train-test split completed.")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info(f"DataIngestionArtifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys)
