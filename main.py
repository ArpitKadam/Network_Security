from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.entity.config_entity import DataValidationConfig, DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
import sys 


if __name__ == "__main__":
    try:
        logging.info(">>>>>>>>>>>>>Data Ingestion Started<<<<<<<<<<<<<<")
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        logging.info(">>>>>>>>>>>>>Data Ingestion Completed<<<<<<<<<<<<<<")
        logging.info(f"Train File Path: {dataingestionartifact.trained_file_path}")
        logging.info(f"Test File Path: {dataingestionartifact.test_file_path}")
        print(dataingestionartifact)

        logging.info(">>>>>>>>>>>>>Data Validation Started<<<<<<<<<<<<<<")        
        data_validation_config = DataValidationConfig(trainingpipelineconfig)     
        data_validation = DataValidation(dataingestionartifact,data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info(">>>>>>>>>>>>>Data Validation Completed<<<<<<<<<<<<<<")
        print(data_validation_artifact)

        logging.info(">>>>>>>>>>>>>Data Transformation Started<<<<<<<<<<<<<<")
        data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
        data_transformation = DataTransformation(data_validation_artifact,data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info(">>>>>>>>>>>>>Data Transformation Completed<<<<<<<<<<<<<<")
        print(data_transformation_artifact)

        logging.info(">>>>>>>>>>>>>Model Trainer Started<<<<<<<<<<<<<<")
        model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
        model_trainer = ModelTrainer(data_transformation_artifact = data_transformation_artifact, model_trainer_config= model_trainer_config)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info(">>>>>>>>>>>>>Model Trainer Completed<<<<<<<<<<<<<<")
        print(model_trainer_artifact)
        
    except NetworkSecurityException as e:
        raise NetworkSecurityException(e,sys)
    


