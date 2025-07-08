import os
import sys
import pymongo
import certifi

from src.exception import MyException
from src.logger import logging
from src.constants import DATABASE_NAME,MONGODB_URL_KEY

# Load the certfificate authority file to avoid timout errrors when connecting to Mongodb
ca = certifi.where()

class MongoDBClient:

    """
    MongoDBClient is responsible for establishing a connection to teh MongoDB database
    
    Attributes:
    -----
    client: MonogoClient
         A shared MongoClient instance for the class.
    database : Database
    The specific database instance that MongoDBClient connects to.
    
    Methods:
    -----
    __init__(database_name:str) -> None
        Initializes the MongoDB connection using the given database name.
        """

    client = None # Shared MongoClient instance across all MongoDBClient instance

    def __init__(self,database_name:str = DATABASE_NAME) -> None:
        """
        Initialize a connection to the MongoDB database. If no existing connection is found, it estblishes a new one.
        
        """

        try:
            # Check if a MongoDB client connection has already been established,; if not create a new one
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)# Retrieve MongoDB URL from environment varaibles
                if mongo_db_url is None:
                    raise Exception(f"Environment variable '{MONGODB_URL_KEY}' is not set.") 

                # Establish a new MongoDB client connection
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url,tlsCAFile = ca)

                # Use the shared MongoClient for this instance
                self.client = MongoDBClient.client
                self.database = self.client[database_name]
                self.database_name = database_name
                logging.info("MongoDB connection successful.")
            
        except Exception as e:
            # Raise a custom exception with traceback details if connection fails
            raise MyException(e,sys)











































