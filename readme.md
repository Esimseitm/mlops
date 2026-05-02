# MLOps Platform — Road Object Detection

YOLOv8n + MLflow + DVC + Evidently AI на BDD100K.

## Установка

```bash
pip install -r requirements.txt
```

## Шаг 0 — Инфраструктура

```bash
docker compose up -d
# MLflow UI → http://localhost:5000
# MinIO UI  → http://localhost:9001  (admin / password123)
```

## Шаг 1 — Данные

BDD100K требует регистрации: https://bdd-data.berkeley.edu/

```bash
# Скачай и распакуй:
#   bdd100k_images_100k.zip → data/raw/bdd100k_images_100k/val/
#   bdd100k_labels_images_val.json → data/raw/

python scripts/prepare_bdd100k.py
```

## Шаг 2 — Baseline

```bash
python scripts/train_baseline.py
# >>> Run ID: <запиши>
```

## Шаг 3 — Симуляция деградации

```bash
python scripts/simulate_degradation.py --run_id <BASELINE_RUN_ID>
# >>> Run ID деградации: <запиши>
```

## Шаг 4 — Evidently отчёт

```bash
python monitoring/evidently_report.py
# Отчёт → monitoring/reports/degradation_report.html
```

## Шаг 5 — Переобучение

```bash
python scripts/build_mixed_dataset.py
python scripts/retrain.py --baseline_run_id <BASELINE_RUN_ID>
```

## Шаг 6 — ONNX экспорт + бенчмарк

```bash
python scripts/export_onnx.py --run_id <RETRAIN_RUN_ID>
```

## Шаг 7 — Итоговая таблица для статьи

```bash
python scripts/generate_results_table.py
# CSV → results/results_table.csv
```

## Структура

```
mlops/
├── docker-compose.yml
├── requirements.txt
├── configs/
│   ├── env.py                  # MLflow/MinIO credentials
│   ├── bdd100k_clear.yaml
│   ├── bdd100k_ood.yaml
│   └── bdd100k_mixed.yaml
├── scripts/
│   ├── prepare_bdd100k.py
│   ├── train_baseline.py
│   ├── simulate_degradation.py
│   ├── build_mixed_dataset.py
│   ├── retrain.py
│   ├── export_onnx.py
│   └── generate_results_table.py
├── monitoring/
│   └── evidently_report.py
└── data/                       (DVC, не в Git)
```

## GTX 1650 (4GB VRAM)

batch=8, imgsz=640. Если OOM → уменьши batch до 4.
