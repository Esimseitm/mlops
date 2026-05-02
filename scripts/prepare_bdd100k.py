"""
Подготавливает BDD100K validation subset для эксперимента.
Разделяет изображения на:
  - data/clear/train+val/  — дневные чёткие сцены (80/20 split)
  - data/ood/val/          — ночные/дождливые сцены (только тест, не обучаем)

Ожидает что файлы уже скачаны вручную:
  data/raw/bdd100k_labels_images_val.json
  data/raw/bdd100k_images_100k/val/*.jpg

Запуск: python scripts/prepare_bdd100k.py
"""

import json
import random
import shutil
from pathlib import Path

from tqdm import tqdm

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"

BDD_CLASSES = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "person": 3,
    "rider": 4,
    "traffic light": 5,
    "traffic sign": 6,
    "motor": 7,
    "bike": 8,
    "train": 9,
}

OOD_WEATHER = {"rainy", "snowy", "foggy"}
OOD_TIMEOFDAY = {"night", "dawn/dusk"}

VAL_RATIO = 0.2
RANDOM_SEED = 42


def convert_bbox_to_yolo(x1, y1, x2, y2, img_w=1280, img_h=720):
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def save_item(item: dict, images_src: Path, img_out: Path, lbl_out: Path) -> bool:
    img_name = item["name"]
    img_src = images_src / img_name
    if not img_src.exists():
        return False

    labels = item.get("labels") or []
    yolo_lines = []
    for label in labels:
        category = label.get("category", "")
        if category not in BDD_CLASSES:
            continue
        box2d = label.get("box2d")
        if box2d is None:
            continue
        class_id = BDD_CLASSES[category]
        cx, cy, w, h = convert_bbox_to_yolo(
            box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]
        )
        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    if not yolo_lines:
        return False

    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    shutil.copy(img_src, img_out / img_name)
    (lbl_out / (Path(img_name).stem + ".txt")).write_text("\n".join(yolo_lines))
    return True


def process_clear(annotations: list, images_src: Path):
    """Фильтрует дневные чёткие сцены и делает train/val split 80/20."""
    clear_items = []
    for item in annotations:
        attrs = item.get("attributes", {})
        weather = attrs.get("weather", "")
        timeofday = attrs.get("timeofday", "")
        is_ood = weather in OOD_WEATHER or timeofday in OOD_TIMEOFDAY
        if not is_ood:
            clear_items.append(item)

    random.seed(RANDOM_SEED)
    random.shuffle(clear_items)
    split_idx = int(len(clear_items) * (1 - VAL_RATIO))
    train_items = clear_items[:split_idx]
    val_items = clear_items[split_idx:]

    train_img = DATA_DIR / "clear" / "train" / "images"
    train_lbl = DATA_DIR / "clear" / "train" / "labels"
    val_img = DATA_DIR / "clear" / "val" / "images"
    val_lbl = DATA_DIR / "clear" / "val" / "labels"

    n_train = sum(save_item(it, images_src, train_img, train_lbl)
                  for it in tqdm(train_items, desc="clear/train"))
    n_val = sum(save_item(it, images_src, val_img, val_lbl)
                for it in tqdm(val_items, desc="clear/val"))

    print(f"  clear: {n_train} train + {n_val} val")
    return n_train, n_val


def process_ood(annotations: list, images_src: Path):
    """Фильтрует OOD сцены — все идут в val/ (только для тестирования)."""
    ood_items = []
    for item in annotations:
        attrs = item.get("attributes", {})
        weather = attrs.get("weather", "")
        timeofday = attrs.get("timeofday", "")
        is_ood = weather in OOD_WEATHER or timeofday in OOD_TIMEOFDAY
        if is_ood:
            ood_items.append(item)

    val_img = DATA_DIR / "ood" / "val" / "images"
    val_lbl = DATA_DIR / "ood" / "val" / "labels"

    n_ood = sum(save_item(it, images_src, val_img, val_lbl)
                for it in tqdm(ood_items, desc="ood/val"))

    print(f"  ood: {n_ood} val images")
    return n_ood


def main():
    labels_file = RAW_DIR / "bdd100k_labels_images_val.json"
    images_dir = RAW_DIR / "bdd100k_images_100k" / "val"

    if not labels_file.exists():
        print("ERROR: Файл меток не найден:", labels_file)
        print()
        print("Скачай BDD100K вручную с https://bdd-data.berkeley.edu/")
        print("Нужны два файла:")
        print("  1. bdd100k_labels_images_val.json  → data/raw/")
        print("  2. bdd100k_images_100k.zip → распакуй в data/raw/bdd100k_images_100k/")
        return

    if not images_dir.exists():
        print("ERROR: Папка с изображениями не найдена:", images_dir)
        return

    with open(labels_file) as f:
        annotations = json.load(f)

    print(f"Всего аннотаций в val set: {len(annotations)}")

    n_train, n_val = process_clear(annotations, images_dir)
    n_ood = process_ood(annotations, images_dir)

    print(f"\nГотово:")
    print(f"  data/clear/train/: {n_train} изображений")
    print(f"  data/clear/val/:   {n_val} изображений")
    print(f"  data/ood/val/:     {n_ood} изображений")
    print(f"\nСледующий шаг: python scripts/train_baseline.py")


if __name__ == "__main__":
    main()
