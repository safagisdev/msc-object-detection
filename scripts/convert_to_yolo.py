import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

BASE = Path(r"C:\msc-object-detection")
RAW = BASE / "data" / "raw"
SPLITS = BASE / "data" / "splits"
YOLO_ROOT = BASE / "data" / "yolo"

TRAIN_JSON = RAW / "labels" / "bdd100k_labels_images_train.json"
VAL_JSON = RAW / "labels" / "bdd100k_labels_images_val.json"

TRAIN_IMAGES = RAW / "images" / "train"
VAL_IMAGES = RAW / "images" / "val"

CLASS_MAP = {
    "traffic sign": 0,
    "traffic light": 1,
    "bus": 2,
    "truck": 3,
    "motor": 4,
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split(name):
    with open(SPLITS / name, "r", encoding="utf-8") as f:
        return set(json.load(f))


def build_annotation_index(data):
    return {
        item["name"]: item
        for item in data
    }


def build_image_index(image_dir):
    print(f"Indexing images in: {image_dir}")

    index = {}

    for path in image_dir.rglob("*.jpg"):
        index[path.name] = path

    print(f"Found {len(index)} images")

    return index


def bdd_box_to_yolo(box, width, height):
    x1 = float(box["x1"])
    y1 = float(box["y1"])
    x2 = float(box["x2"])
    y2 = float(box["y2"])

    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return None

    x_center = x1 + box_w / 2
    y_center = y1 + box_h / 2

    return (
        x_center / width,
        y_center / height,
        box_w / width,
        box_h / height,
    )


def write_labels(image_names, annotation_index, image_index, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_images = 0
    empty_labels = 0
    total_boxes = 0

    for image_name in tqdm(sorted(image_names)):

        image_path = image_index.get(image_name)

        if image_path is None:
            print(f"Missing image: {image_name}")
            missing_images += 1
            continue

        item = annotation_index.get(image_name)

        if item is None:
            print(f"Missing annotation: {image_name}")
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        lines = []

        for label in item.get("labels", []):
            category = label.get("category")

            if category not in CLASS_MAP:
                continue

            if "box2d" not in label:
                continue

            converted = bdd_box_to_yolo(
                label["box2d"],
                width,
                height
            )

            if converted is None:
                continue

            cls_id = CLASS_MAP[category]
            xc, yc, bw, bh = converted

            lines.append(
                f"{cls_id} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{bw:.6f} "
                f"{bh:.6f}"
            )

            total_boxes += 1

        label_path = output_dir / (
            Path(image_name).stem + ".txt"
        )

        if not lines:
            empty_labels += 1

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    return {
        "missing_images": missing_images,
        "empty_labels": empty_labels,
        "total_boxes": total_boxes,
    }


# Load annotations
train_data = load_json(TRAIN_JSON)
val_data = load_json(VAL_JSON)

train_index = build_annotation_index(train_data)
val_index = build_annotation_index(val_data)

# Build recursive image indexes
train_image_index = build_image_index(TRAIN_IMAGES)
val_image_index = build_image_index(VAL_IMAGES)

split_configs = {
    "train_10": (
        load_split("train_10.json"),
        train_index,
        train_image_index,
    ),
    "train_50": (
        load_split("train_50.json"),
        train_index,
        train_image_index,
    ),
    "train_100": (
        load_split("train_100.json"),
        train_index,
        train_image_index,
    ),
    "train_full": (
        load_split("train_full.json"),
        train_index,
        train_image_index,
    ),
    "validation": (
        load_split("validation.json"),
        val_index,
        val_image_index,
    ),
    "test_day": (
        load_split("test_day.json"),
        val_index,
        val_image_index,
    ),
    "test_night": (
        load_split("test_night.json"),
        val_index,
        val_image_index,
    ),
}

for split_name, config in split_configs.items():

    image_names, annotation_index, image_index = config

    print(f"\nProcessing {split_name}")

    output_dir = (
        YOLO_ROOT /
        "labels" /
        split_name
    )

    stats = write_labels(
        image_names,
        annotation_index,
        image_index,
        output_dir
    )

    print(stats)

print("\nDone.")