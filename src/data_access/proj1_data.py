import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import MyException

class Proj1Data:
    """
    A class to export MongoDB records as a pandas DataFrame.
    """

    def __init__(self) -> None:
        """
        Initializes the MongoDB client connection.
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise MyException(e, sys.exc_info()[2])

    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        """
        Exports an entire MongoDB collection as a pandas DataFrame.

        Parameters:
        ----------
        collection_name : str
            The name of the MongoDB collection to export.
        database_name : Optional[str]
            Name of the database (optional). Defaults to DATABASE_NAME.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the collection data, with '_id' column removed and 'na' values replaced with NaN.
        """
        try:
            # Access specified collection from the default or specified database
            db = self.mongo_client[database_name] if database_name else self.mongo_client.database
            collection = db[collection_name]

            print("Fetching data from MongoDB...")
            records = list(collection.find())

            if not records:
                raise ValueError(f"No records found in collection: '{collection_name}'")

            df = pd.DataFrame(records)
            print(f"Data fetched successfully. Total records: {len(df)}")

            if "id" in df.columns:
                df.drop(columns=["id"], inplace=True)

            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise MyException(e, sys.exc_info()[2])
