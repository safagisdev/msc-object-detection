import json
import random
from pathlib import Path
from collections import Counter

SEED = 42

BASE = Path(r"C:\msc-object-detection")
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "splits"

OUT.mkdir(parents=True, exist_ok=True)

TRAIN_JSON = RAW / "labels" / "bdd100k_labels_images_train.json"
VAL_JSON = RAW / "labels" / "bdd100k_labels_images_val.json"

TARGET_CLASSES = [
    "traffic sign",
    "traffic light",
    "bus",
    "truck",
    "motor",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_target_classes(item):
    classes = set()

    for label in item.get("labels", []):
        category = label.get("category")

        if category in TARGET_CLASSES and "box2d" in label:
            classes.add(category)

    return classes


def get_timeofday(item):
    return item.get("attributes", {}).get(
        "timeofday",
        "undefined"
    )


def relevant_images(data, timeofday=None):
    output = []

    for item in data:

        classes = get_target_classes(item)

        if not classes:
            continue

        if timeofday is not None:
            if get_timeofday(item) != timeofday:
                continue

        output.append({
            "name": item["name"],
            "classes": classes,
            "timeofday": get_timeofday(item),
        })

    return output


def create_balanced_subset(candidates, budget, seed):
    """
    Sample exactly `budget` anchor images for each class.

    The final dataset is the union of those selections.
    Because images can contain several classes, the final
    number of unique images may be less than 5 * budget.
    """

    by_class = {
        cls: [] for cls in TARGET_CLASSES
    }

    for item in candidates:
        for cls in item["classes"]:
            by_class[cls].append(item["name"])

    selected = set()

    for i, cls in enumerate(TARGET_CLASSES):

        available = sorted(set(by_class[cls]))

        if len(available) < budget:
            raise RuntimeError(
                f"{cls} has only {len(available)} images."
            )

        rng = random.Random(seed + i)

        chosen = rng.sample(
            available,
            budget
        )

        selected.update(chosen)

    return sorted(selected)


def calculate_counts(selected_names, candidates):

    selected_set = set(selected_names)

    counts = Counter()

    for item in candidates:

        if item["name"] not in selected_set:
            continue

        for cls in item["classes"]:
            counts[cls] += 1

    return counts


def save_list(filename, names):

    path = OUT / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2)

    print(
        f"Saved: {path} "
        f"({len(names)} unique images)"
    )


# -------------------------------
# LOAD
# -------------------------------

train_data = load_json(TRAIN_JSON)
val_data = load_json(VAL_JSON)

# -------------------------------
# DAYTIME TRAINING
# -------------------------------

train_day = relevant_images(
    train_data,
    timeofday="daytime"
)

print(
    "\nDaytime training pool:",
    len(train_day)
)

previous_set = set()

for budget in [10, 50, 100]:

    selected = create_balanced_subset(
        train_day,
        budget,
        SEED
    )

    # Ensure nested datasets
    selected = sorted(
        set(selected) | previous_set
    )

    previous_set = set(selected)

    counts = calculate_counts(
        selected,
        train_day
    )

    save_list(
        f"train_{budget}.json",
        selected
    )

    print(f"\nBudget {budget}")
    print(
        "Unique images:",
        len(selected)
    )

    for cls in TARGET_CLASSES:

        display = (
            "motorcycle"
            if cls == "motor"
            else cls
        )

        print(
            f"  {display:15s}: "
            f"{counts[cls]} images"
        )


# -------------------------------
# FULL DAYTIME TRAINING SET
# -------------------------------

full_names = sorted([
    item["name"]
    for item in train_day
])

save_list(
    "train_full.json",
    full_names
)


# -------------------------------
# VALIDATION / TEST
# -------------------------------

# -------------------------------
# VALIDATION / FINAL TEST
# -------------------------------

val_day = relevant_images(
    val_data,
    timeofday="daytime"
)

val_night = relevant_images(
    val_data,
    timeofday="night"
)

rng = random.Random(SEED)
rng.shuffle(val_day)
rng.shuffle(val_night)

# 20% of DAYTIME val images used for validation/tuning
day_val_size = int(len(val_day) * 0.20)

validation_day = val_day[:day_val_size]
test_day = val_day[day_val_size:]

# ALL nighttime images remain unseen until final testing
test_night = val_night

validation = sorted([
    x["name"]
    for x in validation_day
])

save_list(
    "validation.json",
    validation
)

save_list(
    "test_day.json",
    sorted([
        x["name"]
        for x in test_day
    ])
)

save_list(
    "test_night.json",
    sorted([
        x["name"]
        for x in test_night
    ])
)

print("\n--------------------------")
print("FINAL SPLIT SUMMARY")
print("--------------------------")

print("Train full       :", len(full_names))
print("Validation (day) :", len(validation))
print("Test daytime     :", len(test_day))
print("Test nighttime   :", len(test_night))