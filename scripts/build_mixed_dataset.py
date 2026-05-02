"""
Создаёт смешанный датасет для переобучения:
  data/mixed/train/ = data/clear/train/ + data/ood/val/ (80%)
  data/mixed/val/   = data/clear/val/   (оставшиеся 20% clear)

Запуск: python scripts/build_mixed_dataset.py
"""

import random
import shutil
from pathlib import Path

CLEAR_TRAIN = Path("data/clear/train")
CLEAR_VAL = Path("data/clear/val")
OOD_VAL = Path("data/ood/val")
MIXED_TRAIN = Path("data/mixed/train")
MIXED_VAL = Path("data/mixed/val")
RANDOM_SEED = 42


def copy_split(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path) -> int:
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    copied = 0
    if not src_img.exists():
        return 0
    for img in src_img.iterdir():
        lbl = src_lbl / (img.stem + ".txt")
        if not lbl.exists():
            continue
        dst_img_file = dst_img / img.name
        if not dst_img_file.exists():
            shutil.copy(img, dst_img_file)
            shutil.copy(lbl, dst_lbl / lbl.name)
            copied += 1
    return copied


def main():
    # train: clear/train + часть ood/val
    n1 = copy_split(
        CLEAR_TRAIN / "images", CLEAR_TRAIN / "labels",
        MIXED_TRAIN / "images", MIXED_TRAIN / "labels",
    )

    # Берём 80% OOD для train, 20% для val
    ood_images = sorted((OOD_VAL / "images").glob("*.jpg"))
    if not ood_images:
        ood_images = sorted((OOD_VAL / "images").glob("*.png"))

    random.seed(RANDOM_SEED)
    random.shuffle(ood_images)
    split = int(len(ood_images) * 0.8)
    ood_train_imgs = ood_images[:split]
    ood_val_imgs = ood_images[split:]

    n2 = 0
    for img in ood_train_imgs:
        lbl = OOD_VAL / "labels" / (img.stem + ".txt")
        if lbl.exists():
            dst = MIXED_TRAIN / "images" / img.name
            if not dst.exists():
                shutil.copy(img, dst)
                shutil.copy(lbl, MIXED_TRAIN / "labels" / lbl.name)
                n2 += 1

    # val: clear/val + оставшийся ood
    n3 = copy_split(
        CLEAR_VAL / "images", CLEAR_VAL / "labels",
        MIXED_VAL / "images", MIXED_VAL / "labels",
    )

    n4 = 0
    for img in ood_val_imgs:
        lbl = OOD_VAL / "labels" / (img.stem + ".txt")
        if lbl.exists():
            dst = MIXED_VAL / "images" / img.name
            if not dst.exists():
                shutil.copy(img, dst)
                shutil.copy(lbl, MIXED_VAL / "labels" / lbl.name)
                n4 += 1

    total_train = len(list((MIXED_TRAIN / "images").iterdir()))
    total_val = len(list((MIXED_VAL / "images").iterdir()))

    print(f"mixed/train: {total_train} images ({n1} clear + {n2} ood)")
    print(f"mixed/val:   {total_val} images ({n3} clear + {n4} ood)")
    print(f"\nСледующий шаг: python scripts/retrain.py --baseline_run_id <ID>")


if __name__ == "__main__":
    main()
