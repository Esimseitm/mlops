"""
Единая точка настройки подключения к MLflow и MinIO.
Импортируй: from configs.env import setup_mlflow
"""

import os


def setup_mlflow():
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "password123"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"


EXPERIMENT_NAME = "road_detection_bdd100k"
