"""
Переобучение на смешанных данных (clear + OOD) после деградации.
Fine-tune с весов baseline модели.
После обучения автоматически валидирует на OOD — доказывает восстановление.

Запуск: python scripts/retrain.py --baseline_run_id <RUN_ID>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
from ultralytics import YOLO

from configs.env import EXPERIMENT_NAME, setup_mlflow

setup_mlflow()

RUN_NAME = "retrain_mixed_data"
MIXED_CONFIG = "configs/bdd100k_mixed.yaml"
OOD_CONFIG = "configs/bdd100k_ood.yaml"

CLASS_NAMES = [
    "car", "truck", "bus", "person", "rider",
    "traffic light", "traffic sign", "motor", "bike", "train",
]

TRAIN_PARAMS = {
    "data": MIXED_CONFIG,
    "epochs": 30,
    "imgsz": 640,
    "batch": 8,
    "device": 0,
    "project": "runs/train",
    "name": RUN_NAME,
    "exist_ok": True,
}


def load_baseline_weights(run_id: str) -> Path:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="weights/best.pt",
    )
    return Path(local_path)


def main(baseline_run_id: str):
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Загружаем baseline веса для fine-tune: {baseline_run_id}")
    weights_path = load_baseline_weights(baseline_run_id)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_param("baseline_run_id", baseline_run_id)
        mlflow.log_param("retrain_reason", "degradation_detected")
        mlflow.log_param("init_weights", "baseline_best.pt")
        mlflow.log_params({k: str(v) for k, v in TRAIN_PARAMS.items()})
        mlflow.set_tag("stage", "retrain")

        model = YOLO(str(weights_path))
        results = model.train(**TRAIN_PARAMS)

        metrics = {
            "mAP50_retrain":     results.results_dict.get("metrics/mAP50(B)", 0),
            "mAP50_95_retrain":  results.results_dict.get("metrics/mAP50-95(B)", 0),
            "precision_retrain": results.results_dict.get("metrics/precision(B)", 0),
            "recall_retrain":    results.results_dict.get("metrics/recall(B)", 0),
        }
        mlflow.log_metrics(metrics)

        # Per-class
        if hasattr(results, "box") and hasattr(results.box, "ap50") and results.box.ap50 is not None:
            for i, ap in enumerate(results.box.ap50):
                if i < len(CLASS_NAMES):
                    mlflow.log_metric(f"retrain_AP50_{CLASS_NAMES[i]}", float(ap))

        # Повторная валидация на OOD — доказываем восстановление
        print("\nВалидация retrained модели на OOD данных...")
        best_weights = Path(f"runs/train/{RUN_NAME}/weights/best.pt")
        retrained_model = YOLO(str(best_weights))
        ood_results = retrained_model.val(data=OOD_CONFIG, device=0, imgsz=640, batch=8)

        ood_metrics = {
            "mAP50_retrain_on_ood":     ood_results.results_dict.get("metrics/mAP50(B)", 0),
            "precision_retrain_on_ood": ood_results.results_dict.get("metrics/precision(B)", 0),
            "recall_retrain_on_ood":    ood_results.results_dict.get("metrics/recall(B)", 0),
        }
        mlflow.log_metrics(ood_metrics)

        # Сравниваем всё
        baseline_run = mlflow.get_run(baseline_run_id)
        baseline_map = (
            baseline_run.data.metrics.get("mAP50")
            or baseline_run.data.metrics.get("metrics/mAP50(B)")
            or 0
        )
        retrain_map = metrics["mAP50_retrain"]
        retrain_ood_map = ood_metrics["mAP50_retrain_on_ood"]

        mlflow.log_metric("mAP50_baseline_ref", baseline_map)
        mlflow.log_metric("mAP50_recovery_vs_baseline", retrain_map - baseline_map)

        if best_weights.exists():
            mlflow.log_artifact(str(best_weights), artifact_path="weights")

        print("\n=== Результаты переобучения ===")
        print(f"  Baseline mAP@0.5 (clear):         {baseline_map:.4f}")
        print(f"  Retrain mAP@0.5  (mixed val):     {retrain_map:.4f}")
        print(f"  Retrain mAP@0.5  (OOD val):       {retrain_ood_map:.4f}")
        print(f"  Recovery vs baseline:              {'+' if retrain_map >= baseline_map else ''}{retrain_map - baseline_map:.4f}")
        print(f"\n>>> Run ID: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_run_id", required=True)
    args = parser.parse_args()
    main(args.baseline_run_id)
