import os, sys
from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from networksecurity.logging.logger import logging

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        model_accuracy_score = accuracy_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_f1_score = f1_score(y_true, y_pred)

        classification_metric = ClassificationMetricArtifact(f1_score = model_f1_score,
            accuracy_score = model_accuracy_score, precision_score = model_precision_score,
            recall_score = model_recall_score)
        return classification_metric
    except Exception as e:
        raise NetworkSecurityException(e, sys)