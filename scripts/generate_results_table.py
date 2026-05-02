"""
Генерирует итоговую таблицу результатов для секции Results статьи.
Собирает метрики из всех MLflow runs в эксперименте.

Запуск: python scripts/generate_results_table.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import pandas as pd

from configs.env import EXPERIMENT_NAME, setup_mlflow

setup_mlflow()

OUTPUT_DIR = Path("results")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"ERROR: Эксперимент '{EXPERIMENT_NAME}' не найден в MLflow")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time ASC"],
    )

    # Собираем ключевые runs по тегу stage
    stages = {}
    for r in runs:
        stage = r.data.tags.get("stage", r.info.run_name)
        stages[stage] = r

    # Таблица 1: Overall metrics comparison
    rows = []

    baseline = stages.get("baseline")
    if baseline:
        rows.append({
            "Stage": "Baseline (clear/daytime)",
            "mAP@0.5": baseline.data.metrics.get("mAP50", "-"),
            "mAP@0.5:0.95": baseline.data.metrics.get("mAP50_95", "-"),
            "Precision": baseline.data.metrics.get("precision", "-"),
            "Recall": baseline.data.metrics.get("recall", "-"),
        })

    degradation = stages.get("degradation")
    if degradation:
        rows.append({
            "Stage": "Degradation (OOD test)",
            "mAP@0.5": degradation.data.metrics.get("mAP50_ood", "-"),
            "mAP@0.5:0.95": degradation.data.metrics.get("mAP50_95_ood", "-"),
            "Precision": degradation.data.metrics.get("precision_ood", "-"),
            "Recall": degradation.data.metrics.get("recall_ood", "-"),
        })

    retrain = stages.get("retrain")
    if retrain:
        rows.append({
            "Stage": "Retrain (mixed data)",
            "mAP@0.5": retrain.data.metrics.get("mAP50_retrain", "-"),
            "mAP@0.5:0.95": retrain.data.metrics.get("mAP50_95_retrain", "-"),
            "Precision": retrain.data.metrics.get("precision_retrain", "-"),
            "Recall": retrain.data.metrics.get("recall_retrain", "-"),
        })
        rows.append({
            "Stage": "Retrain → OOD validation",
            "mAP@0.5": retrain.data.metrics.get("mAP50_retrain_on_ood", "-"),
            "mAP@0.5:0.95": "-",
            "Precision": retrain.data.metrics.get("precision_retrain_on_ood", "-"),
            "Recall": retrain.data.metrics.get("recall_retrain_on_ood", "-"),
        })

    df = pd.DataFrame(rows)

    # Форматируем числа
    for col in ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"]:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)

    # Сохраняем
    csv_path = OUTPUT_DIR / "results_table.csv"
    df.to_csv(csv_path, index=False)

    print("=== Results Table (для статьи) ===\n")
    print(df.to_string(index=False))

    # Таблица 2: ONNX бенчмарк
    export = stages.get("export")
    if export:
        print("\n=== ONNX Benchmark ===\n")
        print(f"  PyTorch FPS: {export.data.metrics.get('pytorch_fps', '-')}")
        print(f"  ONNX FPS:    {export.data.metrics.get('onnx_fps', '-')}")
        print(f"  Speedup:     {export.data.metrics.get('onnx_speedup', '-')}x")

    print(f"\nCSV сохранён: {csv_path}")
    print("Используй эту таблицу в секции Results статьи!")


if __name__ == "__main__":
    main()
