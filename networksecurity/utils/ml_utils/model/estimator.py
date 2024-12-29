from networksecurity.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os,sys

class NetworkModel:
    def __init__(self, processor, model):
        try:
            self.processor = processor
            self.model = model
        except Exception as e:
            logging.error(f"Error in NetworkModel: {str(e)}")
            raise NetworkSecurityException(f"Error in NetworkModel: {str(e)}")
        
    def predict(self, x):
        try:
            x_transform = self.processor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            logging.error(f"Error in predict: {str(e)}")
            raise NetworkSecurityException(f"Error in predict: {str(e)}")