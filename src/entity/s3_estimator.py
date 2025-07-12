from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.entity.estimator import MyModel
import sys
from pandas import DataFrame


class Proj1Estimator:
    """
    This class is responsible for saving, retrieving, and using the model
    stored in an S3 bucket for predictions.
    """

    def __init__(self, bucket_name: str, model_path: str):
        """
        :param bucket_name: Name of the S3 bucket where the model is stored.
        :param model_path: Path (key) in the S3 bucket where the model is located.
        """
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.s3 = SimpleStorageService()
        self.loaded_model: MyModel = None

    def is_model_present(self, model_path: str) -> bool:
        """
        Check whether the model exists at the specified path in the S3 bucket.
        """
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
        except MyException as e:
            print(e)
            return False

    def load_model(self) -> MyModel:
        """
        Load the model from the specified S3 path.

        :return: Deserialized MyModel object
        """
        try:
            return self.s3.load_model(self.model_path, bucket_name=self.bucket_name)
        except Exception as e:
            raise MyException(e, sys.exc_info()[2])

    def save_model(self, from_file: str, remove: bool = False) -> None:
        """
        Save a local model file to the S3 bucket.

        :param from_file: Local path to the model file.
        :param remove: Whether to delete the local file after upload.
        """
        try:
            self.s3.upload_file(
                from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove
            )
        except Exception as e:
            raise MyException(e, sys.exc_info()[2])

    def predict(self, dataframe: DataFrame):
        """
        Run predictions using the loaded model.

        :param dataframe: Input features as a DataFrame.
        :return: Model predictions.
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise MyException(e, sys.exc_info()[2])
