import sys
from networksecurity.logging.logger import logger

class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_details: sys):
        """
        Custom exception class to capture and format error details.
        """
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()
        self.lineno = exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        """
        Returns a formatted error message string.
        """
        return (
            f"Error occurred in file: {self.filename} at line number: {self.lineno} "
            f"with error message: {self.error_message}"
        )
