import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(r"C:\msc-object-detection\data\raw")
LABELS = BASE / "labels"

TARGET_CLASSES = {
    "traffic sign",
    "traffic light",
    "bus",
    "truck",
    "motor",
}

FILES = {
    "train": LABELS / "bdd100k_labels_images_train.json",
    "val": LABELS / "bdd100k_labels_images_val.json",
}

for split, json_path in FILES.items():
    print(f"\n=== {split.upper()} ===")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_count = 0
    relevant_image_count = 0
    class_objects = Counter()
    class_images = Counter()
    timeofday_count = Counter()
    relevant_timeofday_count = Counter()

    for item in data:
        image_count += 1

        attributes = item.get("attributes", {})
        timeofday = attributes.get("timeofday", "unknown")
        timeofday_count[timeofday] += 1

        labels = item.get("labels", [])

        classes_in_image = set()
        has_target = False

        for label in labels:
            category = label.get("category")

            if category in TARGET_CLASSES and "box2d" in label:
                has_target = True
                class_objects[category] += 1
                classes_in_image.add(category)

        if has_target:
            relevant_image_count += 1
            relevant_timeofday_count[timeofday] += 1

        for cls in classes_in_image:
            class_images[cls] += 1

    print(f"Total images: {image_count}")
    print(f"Images containing at least one target class: {relevant_image_count}")

    print("\nTime of day - all images:")
    for k, v in timeofday_count.items():
        print(f"  {k}: {v}")

    print("\nTime of day - relevant images:")
    for k, v in relevant_timeofday_count.items():
        print(f"  {k}: {v}")

    print("\nTarget class statistics:")
    for cls in sorted(TARGET_CLASSES):
        print(
            f"  {cls:15s} "
            f"objects={class_objects[cls]:7d} "
            f"images={class_images[cls]:7d}"
        )