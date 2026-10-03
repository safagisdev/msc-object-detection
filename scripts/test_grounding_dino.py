from pathlib import Path
from PIL import Image
import torch

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection
)

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

IMAGE_FOLDER = Path(
    r"C:\msc-object-detection\data\yolo\images\test_day"
)

CLASSES = [
    "traffic sign",
    "traffic light",
    "bus",
    "truck",
    "motorcycle"
]

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25


print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Select one image
# --------------------------------------------------

image_files = list(IMAGE_FOLDER.glob("*.jpg"))

if not image_files:
    raise FileNotFoundError(
        f"No JPG images found in {IMAGE_FOLDER}"
    )

image_path = image_files[0]

print("Test image:", image_path)

image = Image.open(image_path).convert("RGB")


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("\nLoading Grounding DINO...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForZeroShotObjectDetection.from_pretrained(
    MODEL_ID
).to("cuda")

model.eval()

print(
    "Model loaded on:",
    next(model.parameters()).device
)


# --------------------------------------------------
# Run one class at a time
# --------------------------------------------------

all_detections = []

print("\nRunning zero-shot detection...\n")

for class_name in CLASSES:

    prompt = class_name + "."

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to("cuda")

    torch.cuda.synchronize()

    with torch.no_grad():
        outputs = model(**inputs)

    torch.cuda.synchronize()

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]]
    )

    result = results[0]

    boxes = result["boxes"]
    scores = result["scores"]

    print(
        f"{class_name}: "
        f"{len(boxes)} detections"
    )

    for box, score in zip(
        boxes,
        scores
    ):

        box_values = [
            round(v, 1)
            for v in box.tolist()
        ]

        score_value = float(score.item())

        all_detections.append(
            {
                "class": class_name,
                "score": score_value,
                "box": box_values
            }
        )


# --------------------------------------------------
# Print all detections
# --------------------------------------------------

print("\n-------------------------------")
print("FINAL DETECTIONS")
print("-------------------------------")

print("Total detections:", len(all_detections))

for detection in all_detections:

    print(
        f'{detection["class"]} | '
        f'score={detection["score"]:.3f} | '
        f'box={detection["box"]}'
    )


print(
    "\nGrounding DINO test completed successfully."
)