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
        try:
            logging.info("Exporting data from MongoDB")
            my_data = Proj1Data()
            dataframe = my_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )

            logging.info(f"Shape before cleaning: {dataframe.shape}")
            dataframe.columns = dataframe.columns.str.strip()

            for col in dataframe.select_dtypes(include='object').columns:
                dataframe[col] = dataframe[col].str.strip()

            logging.info(f"Shape after cleaning: {dataframe.shape}")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)

            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logging.info(f"Saved to feature store: {feature_store_file_path}")
            return dataframe

        except Exception as e:
            raise MyException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        logging.info("Entered split_data_as_train_test method of DataIngestion class")
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )

            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info("Exported train and test data successfully.")
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")
        try:
            dataframe = self.export_data_into_feature_store()
            logging.info("Got the data from MongoDB")

            self.split_data_as_train_test(dataframe)
            logging.info("Performed train-test split")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys)
