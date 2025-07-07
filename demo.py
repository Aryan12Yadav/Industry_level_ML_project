# below code is to check the logging config

# from src.logger import logging

# logging.debug("this is a debug message.")
# logging.info("This is an info message")
# logging.warning("This is a warning message.")
# logging.error("this is an error message.")
# logging.critical("this is a critical message")


# below code is to check teh exception config

from src.logger import logging
from src.exception import MyException
import sys

try:
    a = 1+'Z'
except Exception as e:
    logging.info(e)
    raise MyException(e,sys) from e