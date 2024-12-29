import os, sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.constants.training_pipeline import TRAINING_BUCKET_NAME
from networksecurity.cloud.s3_syncer import S3Sync

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            logging.info(">>>>>>>>>>>>Data Ingestion Started<<<<<<<<<<<<<<")
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(">>>>>>>>>>>>Completed Data Ingestion<<<<<<<<<<<<<<")
            logging.info(f"Data Ingestion completed: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
            logging.info(">>>>>>>>>>>>>>Data Validation started<<<<<<<<<<<<<<")
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(">>>>>>>>>>>>>>Data Validation completed<<<<<<<<<<<<<<")
            logging.info(f"Data Validation completed: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):  
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,data_transformation_config=data_transformation_config)
            logging.info(">>>>>>>>>>>>>Data Transformation Started<<<<<<<<<<<<<<")
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(">>>>>>>>>>>>>Data Transformation completed<<<<<<<<<<<<<<")
            logging.info(f"Data Transformation completed: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact, model_trainer_config= self.model_trainer_config)
            logging.info(">>>>>>>>>>>>Model Trainer started<<<<<<<<<<<<<<")
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(">>>>>>>>>>>>>Model Trainer completed <<<<<<<<<<<<<<")
            logging.info(f"Model Trainer completed: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    def sync_artifact_dir_to_S3(self):
        try:
            logging.info("Syncing artifact to S3")
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_S3(folder = self.training_pipeline_config.artifact_dir, aws_bucket_url = aws_bucket_url)
            logging.info("Artifact Syncing completed")
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    def sync_saved_models_dir_to_S3(self):
        try:
            logging.info("Syncing models to S3")
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_models/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_S3(folder = self.training_pipeline_config.models_dir, aws_bucket_url = aws_bucket_url)
            logging.info("Models Syncing completed")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

        
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            self.sync_artifact_dir_to_S3()
            self.sync_saved_models_dir_to_S3()
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)