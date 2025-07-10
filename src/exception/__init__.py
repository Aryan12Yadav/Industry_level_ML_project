import sys
import logging


def error_message_detail(error: Exception, tb=None) -> str:
    """
    Constructs a detailed error message from the given exception and traceback.

    Args:
        error (Exception): The actual exception object.
        tb (traceback or None): The traceback object from sys.exc_info()[2].

    Returns:
        str: A formatted string containing file name, line number, and the error message.
    """
    if tb:
        file_name = tb.tb_frame.f_code.co_filename
        line_number = tb.tb_lineno
        error_message = f"Error occurred in script: [{file_name}] at line number [{line_number}]: {str(error)}"
    else:
        error_message = str(error)

    logging.error(error_message)  # Optional: Log the error
    return error_message


class MyException(Exception):
    """
    Custom exception class that captures detailed traceback information.
    """

    def __init__(self, error: Exception, tb=None):
        """
        Initializes MyException with detailed error information.

        Args:
            error (Exception): The raised exception.
            tb (traceback or None): The traceback object (typically from sys.exc_info()[2]).
        """
        super().__init__(str(error))
        self.error_message = error_message_detail(error, tb)

    def __str__(self) -> str:
        return self.error_message
