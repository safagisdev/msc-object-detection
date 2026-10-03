import sys

if len(sys.argv) != 2:
    raise RuntimeError(
        "Usage: python evaluate_grounding_dino_test.py day|night"
    )

SPLIT = sys.argv[1].lower()

if SPLIT == "day":
    SPLIT_NAME = "test_day"

elif SPLIT == "night":
    SPLIT_NAME = "test_night"

else:
    raise RuntimeError(
        "Split must be 'day' or 'night'."
    )


IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "yolo"
    / "images"
    / SPLIT_NAME
)

LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "yolo"
    / "labels"
    / SPLIT_NAME
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / f"grounding_dino_{SPLIT_NAME}"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)