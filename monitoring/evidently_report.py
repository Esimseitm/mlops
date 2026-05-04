"""
Evidently AI отчёт: сравнивает per-image confidence и detection count
между baseline и OOD данными. Генерирует HTML-отчёт + логирует в MLflow.

Запуск:
  python monitoring/evidently_report.py
  python monitoring/evidently_report.py --scores_file monitoring/ood_scores.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import pandas as pd
from evidently.report import Report
from evidently.presets import DataDriftPreset

from configs.env import EXPERIMENT_NAME, setup_mlflow

setup_mlflow()

REPORT_DIR = Path("monitoring/reports")
SCORES_FILE = Path("monitoring/ood_scores.json")


def load_scores(scores_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    data = json.loads(scores_file.read_text())
    ref_df = pd.DataFrame(data["baseline_data"])
    cur_df = pd.DataFrame(data["ood_data"])
    return ref_df, cur_df, data["baseline_run_id"], data["ood_run_id"]


def main(scores_file: Path):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not scores_file.exists():
        print(f"ERROR: Файл не найден: {scores_file}")
        print("Сначала запусти: python scripts/simulate_degradation.py --run_id <ID>")
        return

    ref_df, cur_df, baseline_run_id, ood_run_id = load_scores(scores_file)

    print(f"Reference (baseline): {len(ref_df)} images")
    print(f"Current (OOD):        {len(cur_df)} images")

    report = Report([DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)

    report_path = REPORT_DIR / "degradation_report.html"
    report.save_html(str(report_path))
    print(f"\nОтчёт сохранён: {report_path}")

    # Логируем в MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="evidently_drift_report"):
        mlflow.log_artifact(str(report_path), artifact_path="evidently")
        mlflow.log_param("baseline_run_id", baseline_run_id)
        mlflow.log_param("ood_run_id", ood_run_id)
        mlflow.set_tag("stage", "monitoring")

        bm = mlflow.get_run(baseline_run_id).data.metrics
        om = mlflow.get_run(ood_run_id).data.metrics
        mlflow.log_metric("mAP50_baseline", bm.get("mAP50") or bm.get("metrics/mAP50B", 0))
        mlflow.log_metric("mAP50_ood", om.get("mAP50_ood", 0))
        mlflow.log_metric("mAP50_drop_pct", om.get("mAP50_drop_percent", 0))

        print("Отчёт загружен в MLflow artifacts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_file", type=Path, default=SCORES_FILE)
    args = parser.parse_args()
    main(args.scores_file)
