import json
import os
from pathlib import Path

BASE = Path(r"C:\msc-object-detection")

RAW = BASE / "data" / "raw"
SPLITS = BASE / "data" / "splits"
YOLO = BASE / "data" / "yolo"

TRAIN_SOURCE = RAW / "images" / "train"
VAL_SOURCE = RAW / "images" / "val"

CLASS_NAMES = [
    "traffic sign",
    "traffic light",
    "bus",
    "truck",
    "motorcycle"
]


def load_split(filename):
    with open(SPLITS / filename, "r", encoding="utf-8") as f:
        return json.load(f)


def build_image_index(folder):

    print(f"Indexing: {folder}")

    index = {}

    for path in folder.rglob("*.jpg"):
        index[path.name] = path

    print(f"Found {len(index)} images")

    return index


def create_hardlinks(split_name, image_names, image_index):

    target_dir = YOLO / "images" / split_name
    target_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    existing = 0
    missing = 0

    for image_name in image_names:

        source = image_index.get(image_name)

        if source is None:
            print(f"Missing source image: {image_name}")
            missing += 1
            continue

        target = target_dir / image_name

        if target.exists():
            existing += 1
            continue

        # Hard link = no duplicate image storage
        os.link(source, target)
        created += 1

    print(
        f"{split_name}: "
        f"created={created}, "
        f"existing={existing}, "
        f"missing={missing}"
    )


def create_yaml(filename, train_split):

    yaml_path = YOLO / filename

    content = f"""path: {YOLO.as_posix()}

train: images/{train_split}
val: images/validation

names:
  0: traffic sign
  1: traffic light
  2: bus
  3: truck
  4: motorcycle
"""

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Created YAML: {yaml_path}")


# --------------------------------------------------
# Image indexes
# --------------------------------------------------

train_index = build_image_index(TRAIN_SOURCE)
val_index = build_image_index(VAL_SOURCE)


# --------------------------------------------------
# Splits
# --------------------------------------------------

train_10 = load_split("train_10.json")
train_50 = load_split("train_50.json")
train_100 = load_split("train_100.json")
train_full = load_split("train_full.json")

validation = load_split("validation.json")
test_day = load_split("test_day.json")
test_night = load_split("test_night.json")


# --------------------------------------------------
# Create hard-linked image folders
# --------------------------------------------------

create_hardlinks(
    "train_10",
    train_10,
    train_index
)

create_hardlinks(
    "train_50",
    train_50,
    train_index
)

create_hardlinks(
    "train_100",
    train_100,
    train_index
)

create_hardlinks(
    "train_full",
    train_full,
    train_index
)

create_hardlinks(
    "validation",
    validation,
    val_index
)

create_hardlinks(
    "test_day",
    test_day,
    val_index
)

create_hardlinks(
    "test_night",
    test_night,
    val_index
)


# --------------------------------------------------
# YAML configs
# --------------------------------------------------

create_yaml(
    "dataset_10.yaml",
    "train_10"
)

create_yaml(
    "dataset_50.yaml",
    "train_50"
)

create_yaml(
    "dataset_100.yaml",
    "train_100"
)

create_yaml(
    "dataset_full.yaml",
    "train_full"
)


print("\nYOLO dataset preparation complete.")