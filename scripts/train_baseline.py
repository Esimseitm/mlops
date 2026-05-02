"""
Обучение baseline YOLOv8n на BDD100K (clear/daytime сцены).
Логирует в MLflow: параметры, метрики (overall + per-class), веса модели.

Запуск: python scripts/train_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
from ultralytics import YOLO

from configs.env import EXPERIMENT_NAME, setup_mlflow

setup_mlflow()

RUN_NAME = "baseline_clear_daytime"
CONFIG = "configs/bdd100k_clear.yaml"

CLASS_NAMES = [
    "car", "truck", "bus", "person", "rider",
    "traffic light", "traffic sign", "motor", "bike", "train",
]

TRAIN_PARAMS = {
    "data": CONFIG,
    "epochs": 30,
    "imgsz": 640,
    "batch": 8,
    "device": 0,
    "project": "runs/train",
    "name": RUN_NAME,
    "exist_ok": True,
    "verbose": True,
}


def log_per_class_metrics(results, prefix=""):
    """Логирует AP50 по каждому классу — нужно для Results таблицы в статье."""
    if hasattr(results, "ap50") and results.ap50 is not None:
        for i, ap in enumerate(results.ap50):
            if i < len(CLASS_NAMES):
                mlflow.log_metric(f"{prefix}AP50_{CLASS_NAMES[i]}", float(ap))


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        print(f"MLflow Run ID: {run.info.run_id}")

        mlflow.log_params({k: str(v) for k, v in TRAIN_PARAMS.items()})
        mlflow.set_tag("stage", "baseline")

        model = YOLO("yolov8n.pt")
        results = model.train(**TRAIN_PARAMS)

        metrics = {
            "mAP50":     results.results_dict.get("metrics/mAP50(B)", 0),
            "mAP50_95":  results.results_dict.get("metrics/mAP50-95(B)", 0),
            "precision": results.results_dict.get("metrics/precision(B)", 0),
            "recall":    results.results_dict.get("metrics/recall(B)", 0),
        }
        mlflow.log_metrics(metrics)

        # Per-class AP50
        if hasattr(results, "box"):
            log_per_class_metrics(results.box, prefix="baseline_")

        best_weights = Path(f"runs/train/{RUN_NAME}/weights/best.pt")
        if best_weights.exists():
            mlflow.log_artifact(str(best_weights), artifact_path="weights")

        print("\n=== Baseline результаты ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"\n>>> Run ID: {run.info.run_id}")
        print(">>> Скопируй Run ID — он нужен для следующего шага!")


if __name__ == "__main__":
    main()
