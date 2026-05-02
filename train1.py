import os
import mlflow
from ultralytics import YOLO

# 1. Настройки подключения к твоей инфраструктуре
os.environ['MLFLOW_TRACKING_URI'] = "http://localhost:5000"
os.environ['MLFLOW_EXPERIMENT_NAME'] = "Road_Object_Detection_Final"

# Настройки для сохранения весов в MinIO (S3)
os.environ['AWS_ACCESS_KEY_ID'] = "admin"
os.environ['AWS_SECRET_ACCESS_KEY'] = "password123"
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"
os.environ['MLFLOW_S3_IGNORE_TLS'] = "true" # Так как у нас локальный http

def run_train():
    print("--- Запуск чистого обучения ---")
    
    # Загружаем предобученную модель
    model = YOLO("yolov8n.pt")

    # Запускаем обучение
    # YOLOv8 сама найдет MLflow и отправит туда данные
    model.train(
        data="./data/data.yaml",   # Твой конфиг к coco8
        epochs=10,                 # 10 эпох — быстро и наглядно
        imgsz=640,
        batch=16,
        project="road_detection",
        name="persistent_mvp_run",
        device="cpu"               # Если хочешь GPU, напиши device=0
    )

    print("--- Обучение завершено! Проверяй MLflow. ---")

if __name__ == "__main__":
    run_train()