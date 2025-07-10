import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact
)
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file
)


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys.exc_info()[2])

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].str.strip()
            return df
        except Exception as e:
            raise MyException(e, sys.exc_info()[2])

    def get_data_transformer_object(self) -> Pipeline:
        try:
            logging.info("Preparing preprocessing pipeline...")

            num_features = self._schema_config['num_features']
            mm_columns = self._schema_config['mm_columns']

            numeric_transformer = StandardScaler()
            minmax_transformer = MinMaxScaler()

            preprocessor = ColumnTransformer(
                transformers=[
                    ('std_scaler', numeric_transformer, num_features),
                    ('minmax_scaler', minmax_transformer, mm_columns)
                ],
                remainder='passthrough'
            )

            return Pipeline(steps=[('preprocessor', preprocessor)])
        except Exception as e:
            raise MyException(e, sys.exc_info()[2])

    def _map_gender_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        return df

    def _drop_id_column(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["id", "_id"]:
            if col in df.columns:
                df.drop(columns=col, inplace=True)
        return df

    def _create_dummy_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(df, drop_first=True)

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype(int)
        return df

    def _prepare_features(self, df: pd.DataFrame, ref_columns: list = None) -> pd.DataFrame:
        df = self._map_gender_column(df)
        df = self._drop_id_column(df)
        df = self._create_dummy_columns(df)
        df = self._rename_columns(df)

        if ref_columns is not None:
            missing_cols = set(ref_columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0
            extra_cols = set(df.columns) - set(ref_columns)
            df.drop(columns=extra_cols, inplace=True)
            df = df[ref_columns]

        return df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started")

            if not self.data_validation_artifact.validation_status:
                raise MyException(self.data_validation_artifact.message, sys.exc_info()[2])

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            input_feature_train_df = train_df.drop(columns=TARGET_COLUMN)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=TARGET_COLUMN)
            target_feature_test_df = test_df[TARGET_COLUMN]

            input_feature_train_df = self._prepare_features(input_feature_train_df)
            input_feature_test_df = self._prepare_features(
                input_feature_test_df,
                ref_columns=input_feature_train_df.columns.tolist()
            )

            logging.info(f"Train shape after encoding: {input_feature_train_df.shape}")
            logging.info(f"Test shape after encoding: {input_feature_test_df.shape}")

            # Preprocessing
            preprocessor = self.get_data_transformer_object()

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # ✅ Handle NaNs before SMOTEENN
            logging.info("Imputing missing values before SMOTEENN...")
            imputer = SimpleImputer(strategy="mean")
            input_feature_train_arr = imputer.fit_transform(input_feature_train_arr)
            input_feature_test_arr = imputer.transform(input_feature_test_arr)

            # ✅ Apply SMOTEENN
            logging.info("Applying SMOTEENN on training set only...")
            smoteenn = SMOTEENN(sampling_strategy="minority", random_state=42)
            input_feature_train_final, target_feature_train_final = smoteenn.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )

            # Save final transformed arrays
            train_arr = np.c_[input_feature_train_final, target_feature_train_final]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            logging.info("Data Transformation Completed Successfully")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys.exc_info()[2])
