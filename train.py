import os
import mlflow
from ultralytics import YOLO


os.environ['MLFLOW_TRACKING_URI'] = "http://localhost:5000"

os.environ['AWS_ACCESS_KEY_ID'] = "admin"
os.environ['AWS_SECRET_ACCESS_KEY'] = "password123" 
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000" 
# ---

os.environ['MLFLOW_EXPERIMENT_NAME'] = "First test"

def run_train():
    print(f"--- Запуск обучения MVP (только трекинг) ---")
    
    # 2. Инициализация модели
    # Загружаем базовые веса YOLOv8 Nano
    model = YOLO("yolov8n.pt")

    # 3. Запуск обучения
    # Все метрики автоматически улетят в MLflow
    model.train(
        data="./data/data.yaml",   # Путь к нашему конфигу coco8
        epochs=10,                 # 10 проходов по данным
        imgsz=640,                 # Размер изображения
        batch=16,                  # Размер группы изображений
        project="road_detection",  # Папка для локальных сохранений
        name="mvp_run_coco8",      # Название заезда в MLflow
        device="cpu"               # Для теста на WSL лучше оставить cpu
    )

    print("--- Обучение завершено. Проверяй MLflow UI! ---")

if __name__ == "__main__":
    run_train()