"""
Симуляция деградации: inference baseline-модели на OOD-сценах
(ночь, дождь, туман). Фиксирует падение mAP + собирает per-image данные
для Evidently (confidence scores + detection counts).

Запуск: python scripts/simulate_degradation.py --run_id <BASELINE_RUN_ID>
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
from ultralytics import YOLO

from configs.env import EXPERIMENT_NAME, setup_mlflow

setup_mlflow()

OOD_CONFIG = "configs/bdd100k_ood.yaml"
SCORES_FILE = Path("monitoring/ood_scores.json")

CLASS_NAMES = [
    "car", "truck", "bus", "person", "rider",
    "traffic light", "traffic sign", "motor", "bike", "train",
]


def load_baseline_weights(run_id: str) -> Path:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="weights/best.pt",
    )
    return Path(local_path)


def collect_per_image_data(model: YOLO, images_dir: Path, max_images: int = 500) -> list[dict]:
    """
    Per-image: средний confidence + кол-во детекций.
    Evidently'ю нужен массив значений (не одна цифра) для корректного drift detection.
    """
    records = []
    images = sorted(images_dir.glob("*.jpg"))[:max_images]
    if not images:
        images = sorted(images_dir.glob("*.png"))[:max_images]

    for img_path in images:
        preds = model.predict(str(img_path), verbose=False, device=0)
        if preds and len(preds[0].boxes) > 0:
            confs = preds[0].boxes.conf.cpu().tolist()
            records.append({
                "avg_confidence": sum(confs) / len(confs),
                "num_detections": len(confs),
            })
        else:
            records.append({"avg_confidence": 0.0, "num_detections": 0})

    return records


def main(run_id: str):
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Загружаем baseline веса из Run ID: {run_id}")
    weights_path = load_baseline_weights(run_id)
    model = YOLO(str(weights_path))

    with mlflow.start_run(run_name="degradation_ood_test") as run:
        mlflow.log_param("baseline_run_id", run_id)
        mlflow.log_param("test_type", "ood_degradation")
        mlflow.log_param("ood_conditions", "night, rainy, foggy, snowy")
        mlflow.set_tag("stage", "degradation")

        val_results = model.val(data=OOD_CONFIG, device=0, imgsz=640, batch=8)

        metrics = {
            "mAP50_ood":     val_results.results_dict.get("metrics/mAP50(B)", 0),
            "mAP50_95_ood":  val_results.results_dict.get("metrics/mAP50-95(B)", 0),
            "precision_ood": val_results.results_dict.get("metrics/precision(B)", 0),
            "recall_ood":    val_results.results_dict.get("metrics/recall(B)", 0),
        }
        mlflow.log_metrics(metrics)

        # Per-class AP50 на OOD
        if hasattr(val_results, "ap50") and val_results.ap50 is not None:
            for i, ap in enumerate(val_results.ap50):
                if i < len(CLASS_NAMES):
                    mlflow.log_metric(f"ood_AP50_{CLASS_NAMES[i]}", float(ap))

        # Сравниваем с baseline
        # YOLOv8 autolog пишет "metrics/mAP50(B)", наш скрипт пишет "mAP50"
        baseline_run = mlflow.get_run(run_id)
        baseline_map = (
            baseline_run.data.metrics.get("mAP50")
            or baseline_run.data.metrics.get("metrics/mAP50(B)")
            or 0
        )
        ood_map = metrics["mAP50_ood"]
        drop_abs = baseline_map - ood_map
        drop_pct = (drop_abs / baseline_map * 100) if baseline_map > 0 else 0

        mlflow.log_metric("mAP50_baseline_ref", baseline_map)
        mlflow.log_metric("mAP50_drop_absolute", drop_abs)
        mlflow.log_metric("mAP50_drop_percent", drop_pct)

        # Per-image data для Evidently
        ood_images_dir = Path("data/ood/val/images")
        clear_images_dir = Path("data/clear/val/images")

        if ood_images_dir.exists() and clear_images_dir.exists():
            print("Собираем per-image scores для Evidently...")
            ood_data = collect_per_image_data(model, ood_images_dir)
            clear_data = collect_per_image_data(model, clear_images_dir, max_images=len(ood_data))

            scores_payload = {
                "baseline_data": clear_data,
                "ood_data": ood_data,
                "baseline_run_id": run_id,
                "ood_run_id": run.info.run_id,
            }
            SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
            SCORES_FILE.write_text(json.dumps(scores_payload, indent=2))
            mlflow.log_artifact(str(SCORES_FILE), artifact_path="evidently_data")
            print(f"Scores сохранены: {SCORES_FILE}")

        degradation_detected = drop_pct >= 5
        mlflow.set_tag("degradation_detected", str(degradation_detected).lower())

        print("\n=== Результаты симуляции деградации ===")
        print(f"  Baseline mAP@0.5:  {baseline_map:.4f}")
        print(f"  OOD mAP@0.5:       {ood_map:.4f}")
        print(f"  Падение:           -{drop_abs:.4f} ({drop_pct:.1f}%)")
        print(f"  Деградация:        {'ДА' if degradation_detected else 'НЕТ (< 5%)'}")
        print(f"\n>>> Run ID деградации: {run.info.run_id}")
        print(">>> Скопируй — нужен для evidently_report.py!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True, help="MLflow Run ID baseline модели")
    args = parser.parse_args()
    main(args.run_id)
