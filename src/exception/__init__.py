import sys
import logging
import traceback


def error_message_detail(error: Exception, tb=None) -> str:
    """
    Constructs a detailed error message from the given exception and traceback.
    """
    if tb:
        file_name = tb.tb_frame.f_code.co_filename
        line_number = tb.tb_lineno
        error_message = f"Error occurred in script: [{file_name}] at line number [{line_number}]: {str(error)}"
    else:
        error_message = f"Error: {str(error)}"

    logging.error(error_message)
    return error_message


class MyException(Exception):
    """
    Custom exception class that captures detailed traceback information.
    """

    def __init__(self, error: Exception, error_detail=None):
        if error_detail is None:
            error_detail = sys.exc_info()[2]  # Correctly fetch traceback
        self.error_message = error_message_detail(error, error_detail)
        super().__init__(self.error_message)

    def __str__(self) -> str:
        return self.error_message
