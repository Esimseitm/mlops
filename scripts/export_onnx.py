"""
Экспорт лучшей модели в ONNX формат + бенчмарк скорости инференса.
Нужно для секции "Edge Deployment" в статье.

Запуск: python scripts/export_onnx.py --run_id <RETRAIN_RUN_ID>
        python scripts/export_onnx.py --weights runs/train/retrain_mixed_data/weights/best.pt
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
from ultralytics import YOLO

from configs.env import EXPERIMENT_NAME, setup_mlflow

setup_mlflow()


def load_weights(run_id: str = None, weights: str = None) -> Path:
    if weights:
        return Path(weights)
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="weights/best.pt",
    )
    return Path(local_path)


def benchmark(model_path: str, test_images_dir: Path, n_runs: int = 50) -> dict:
    model = YOLO(model_path)
    images = sorted(test_images_dir.glob("*.jpg"))[:n_runs]
    if not images:
        images = sorted(test_images_dir.glob("*.png"))[:n_runs]
    if not images:
        return {"fps": 0, "avg_ms": 0, "n_images": 0}

    # Warmup
    for img in images[:3]:
        model.predict(str(img), verbose=False, device=0)

    times = []
    for img in images:
        start = time.perf_counter()
        model.predict(str(img), verbose=False, device=0)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_ms = sum(times) / len(times)
    fps = 1000 / avg_ms if avg_ms > 0 else 0

    return {"fps": round(fps, 1), "avg_ms": round(avg_ms, 1), "n_images": len(times)}


def main(run_id: str = None, weights: str = None):
    weights_path = load_weights(run_id, weights)
    print(f"Веса: {weights_path}")

    model = YOLO(str(weights_path))

    # Экспорт в ONNX
    onnx_path = model.export(format="onnx", imgsz=640, simplify=True)
    print(f"ONNX экспортирован: {onnx_path}")

    # Бенчмарк: PyTorch vs ONNX
    test_dir = Path("data/clear/val/images")
    if not test_dir.exists():
        test_dir = Path("data/ood/val/images")

    if test_dir.exists():
        print("\nБенчмарк PyTorch (.pt)...")
        pt_bench = benchmark(str(weights_path), test_dir)
        print(f"  PyTorch: {pt_bench['fps']} FPS ({pt_bench['avg_ms']} ms/img)")

        print("Бенчмарк ONNX...")
        onnx_bench = benchmark(onnx_path, test_dir)
        print(f"  ONNX:    {onnx_bench['fps']} FPS ({onnx_bench['avg_ms']} ms/img)")
    else:
        pt_bench = {"fps": 0, "avg_ms": 0, "n_images": 0}
        onnx_bench = {"fps": 0, "avg_ms": 0, "n_images": 0}
        print("Нет данных для бенчмарка — пропускаем")

    # Логируем в MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="onnx_export"):
        mlflow.log_artifact(onnx_path, artifact_path="onnx")
        mlflow.log_param("source_run_id", run_id or "local")
        mlflow.log_param("onnx_path", onnx_path)
        mlflow.set_tag("stage", "export")

        mlflow.log_metric("pytorch_fps", pt_bench["fps"])
        mlflow.log_metric("pytorch_ms_per_img", pt_bench["avg_ms"])
        mlflow.log_metric("onnx_fps", onnx_bench["fps"])
        mlflow.log_metric("onnx_ms_per_img", onnx_bench["avg_ms"])

        if pt_bench["fps"] > 0:
            speedup = onnx_bench["fps"] / pt_bench["fps"]
            mlflow.log_metric("onnx_speedup", round(speedup, 2))
            print(f"\n  ONNX speedup: {speedup:.2f}x")

        print(f"\nГотово! ONNX модель: {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=None, help="MLflow Run ID модели")
    parser.add_argument("--weights", default=None, help="Путь к .pt файлу")
    args = parser.parse_args()

    if not args.run_id and not args.weights:
        print("ERROR: укажи --run_id или --weights")
        sys.exit(1)

    main(args.run_id, args.weights)
