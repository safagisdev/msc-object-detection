import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

PROJECT_ROOT = Path(r"C:\msc-object-detection")

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "yolo"
    / "images"
    / "test_night"
)

LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "yolo"
    / "labels"
    / "test_night"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "grounding_dino_test_night"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RAW_PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "predictions_raw.json"
)

COCO_PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "predictions_coco.json"
)

COCO_GT_FILE = (
    OUTPUT_DIR
    / "ground_truth_coco.json"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "metrics_summary.json"
)


# ============================================================
# CLASSES
# ============================================================

CLASSES = [
    "traffic sign",
    "traffic light",
    "bus",
    "truck",
    "motorcycle"
]

CLASS_TO_YOLO_ID = {
    "traffic sign": 0,
    "traffic light": 1,
    "bus": 2,
    "truck": 3,
    "motorcycle": 4
}

CLASS_TO_COCO_ID = {
    name: i + 1
    for i, name in enumerate(CLASSES)
}

COCO_ID_TO_CLASS = {
    i + 1: name
    for i, name in enumerate(CLASSES)
}


# ============================================================
# FROZEN PARAMETERS FROM VALIDATION
# ============================================================

PREDICTION_BOX_THRESHOLD = 0.05
TEXT_THRESHOLD = 0.25

SAME_CLASS_NMS_IOU = 0.50
CROSS_CLASS_DUPLICATE_IOU = 0.85

FROZEN_CONFIDENCE_THRESHOLD = 0.27

MATCH_IOU = 0.50

DEVICE = "cuda"

SAVE_EVERY = 20


# ============================================================
# IOU
# ============================================================

def box_iou(box_a, box_b):

    x1 = max(
        box_a[0],
        box_b[0]
    )

    y1 = max(
        box_a[1],
        box_b[1]
    )

    x2 = min(
        box_a[2],
        box_b[2]
    )

    y2 = min(
        box_a[3],
        box_b[3]
    )

    inter_w = max(
        0.0,
        x2 - x1
    )

    inter_h = max(
        0.0,
        y2 - y1
    )

    intersection = (
        inter_w
        * inter_h
    )

    area_a = (
        max(
            0.0,
            box_a[2] - box_a[0]
        )
        *
        max(
            0.0,
            box_a[3] - box_a[1]
        )
    )

    area_b = (
        max(
            0.0,
            box_b[2] - box_b[0]
        )
        *
        max(
            0.0,
            box_b[3] - box_b[1]
        )
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection
        / union
    )


# ============================================================
# SAME CLASS NMS
# ============================================================

def nms_predictions(
    predictions,
    iou_threshold
):

    if not predictions:
        return []

    predictions = sorted(
        predictions,
        key=lambda x: x["score"],
        reverse=True
    )

    kept = []

    while predictions:

        best = predictions.pop(0)

        kept.append(best)

        remaining = []

        for pred in predictions:

            iou = box_iou(
                best["box"],
                pred["box"]
            )

            if iou < iou_threshold:
                remaining.append(pred)

        predictions = remaining

    return kept


# ============================================================
# CROSS CLASS DUPLICATE REMOVAL
# ============================================================

def remove_cross_class_duplicates(
    predictions
):

    predictions = sorted(
        predictions,
        key=lambda x: x["score"],
        reverse=True
    )

    kept = []

    for pred in predictions:

        duplicate = False

        for existing in kept:

            if (
                pred["class_id"]
                ==
                existing["class_id"]
            ):
                continue

            iou = box_iou(
                pred["box"],
                existing["box"]
            )

            if (
                iou
                >=
                CROSS_CLASS_DUPLICATE_IOU
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(pred)

    return kept


# ============================================================
# YOLO GROUND TRUTH
# ============================================================

def read_yolo_ground_truth(
    label_path,
    width,
    height
):

    objects = []

    if not label_path.exists():
        return objects

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            parts = (
                line.strip().split()
            )

            if len(parts) < 5:
                continue

            class_id = int(
                parts[0]
            )

            xc = (
                float(parts[1])
                * width
            )

            yc = (
                float(parts[2])
                * height
            )

            bw = (
                float(parts[3])
                * width
            )

            bh = (
                float(parts[4])
                * height
            )

            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2

            objects.append({
                "class_id":
                    class_id,

                "box": [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            })

    return objects


# ============================================================
# BUILD COCO GROUND TRUTH
# ============================================================

def build_coco_ground_truth(
    image_files
):

    dataset = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    for class_name in CLASSES:

        dataset[
            "categories"
        ].append({
            "id":
                CLASS_TO_COCO_ID[
                    class_name
                ],

            "name":
                class_name
        })

    annotation_id = 1

    image_metadata = {}

    for image_id, image_path in enumerate(
        image_files,
        start=1
    ):

        with Image.open(
            image_path
        ) as img:

            width, height = (
                img.size
            )

        image_metadata[
            image_path.name
        ] = {
            "id":
                image_id,

            "width":
                width,

            "height":
                height
        }

        dataset[
            "images"
        ].append({
            "id":
                image_id,

            "file_name":
                image_path.name,

            "width":
                width,

            "height":
                height
        })

        label_path = (
            LABEL_DIR
            /
            f"{image_path.stem}.txt"
        )

        gt_objects = (
            read_yolo_ground_truth(
                label_path,
                width,
                height
            )
        )

        for obj in gt_objects:

            class_id = (
                obj["class_id"]
            )

            x1, y1, x2, y2 = (
                obj["box"]
            )

            bw = x2 - x1
            bh = y2 - y1

            dataset[
                "annotations"
            ].append({
                "id":
                    annotation_id,

                "image_id":
                    image_id,

                "category_id":
                    class_id + 1,

                "bbox": [
                    float(x1),
                    float(y1),
                    float(bw),
                    float(bh)
                ],

                "area":
                    float(
                        bw * bh
                    ),

                "iscrowd":
                    0
            })

            annotation_id += 1

    return (
        dataset,
        image_metadata
    )


# ============================================================
# GROUNDING DINO INFERENCE
# ============================================================

def run_inference(
    model,
    processor,
    image_files,
    image_metadata
):

    if RAW_PREDICTIONS_FILE.exists():

        print(
            "\nExisting prediction cache found."
        )

        with open(
            RAW_PREDICTIONS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            all_predictions = (
                json.load(f)
            )

        completed_images = {
            item["file_name"]
            for item
            in all_predictions
        }

        print(
            f"Resuming from "
            f"{len(completed_images)} "
            f"completed images."
        )

    else:

        all_predictions = []

        completed_images = set()

    pending_images = [
        p
        for p in image_files
        if p.name
        not in completed_images
    ]

    print(
        f"\nImages remaining: "
        f"{len(pending_images)}"
    )

    for index, image_path in enumerate(
        tqdm(
            pending_images,
            desc="Grounding DINO test night"
        ),
        start=1
    ):

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        image_predictions = []

        for class_name in CLASSES:

            prompt = (
                class_name
                + "."
            )

            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(
                DEVICE
            )

            with torch.no_grad():

                outputs = model(
                    **inputs
                )

            results = (
                processor
                .post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=
                        PREDICTION_BOX_THRESHOLD,
                    text_threshold=
                        TEXT_THRESHOLD,
                    target_sizes=[
                        image.size[::-1]
                    ]
                )
            )

            result = results[0]

            class_predictions = []

            for box, score in zip(
                result["boxes"],
                result["scores"]
            ):

                box_values = [
                    float(v)
                    for v
                    in box.tolist()
                ]

                class_predictions.append({
                    "class":
                        class_name,

                    "class_id":
                        CLASS_TO_YOLO_ID[
                            class_name
                        ],

                    "score":
                        float(
                            score.item()
                        ),

                    "box":
                        box_values
                })

            class_predictions = (
                nms_predictions(
                    class_predictions,
                    SAME_CLASS_NMS_IOU
                )
            )

            image_predictions.extend(
                class_predictions
            )

        image_predictions = (
            remove_cross_class_duplicates(
                image_predictions
            )
        )

        all_predictions.append({
            "file_name":
                image_path.name,

            "image_id":
                image_metadata[
                    image_path.name
                ]["id"],

            "predictions":
                image_predictions
        })

        if (
            index
            % SAVE_EVERY
            == 0
        ):

            with open(
                RAW_PREDICTIONS_FILE,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    all_predictions,
                    f,
                    indent=2
                )

    with open(
        RAW_PREDICTIONS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_predictions,
            f,
            indent=2
        )

    return all_predictions


# ============================================================
# PREDICTIONS -> COCO
# ============================================================

def predictions_to_coco(
    all_predictions
):

    coco_predictions = []

    for image_result in all_predictions:

        image_id = (
            image_result[
                "image_id"
            ]
        )

        for pred in (
            image_result[
                "predictions"
            ]
        ):

            x1, y1, x2, y2 = (
                pred["box"]
            )

            width = (
                x2 - x1
            )

            height = (
                y2 - y1
            )

            coco_predictions.append({
                "image_id":
                    image_id,

                "category_id":
                    pred["class_id"]
                    + 1,

                "bbox": [
                    x1,
                    y1,
                    width,
                    height
                ],

                "score":
                    pred["score"]
            })

    return coco_predictions


# ============================================================
# MAP
# ============================================================

def evaluate_map(
    ground_truth_file,
    prediction_file
):

    coco_gt = COCO(
        str(
            ground_truth_file
        )
    )

    coco_dt = coco_gt.loadRes(
        str(
            prediction_file
        )
    )

    evaluator = COCOeval(
        coco_gt,
        coco_dt,
        "bbox"
    )

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    map_50_95 = float(
        evaluator.stats[0]
    )

    map_50 = float(
        evaluator.stats[1]
    )

    return (
        map_50,
        map_50_95
    )


# ============================================================
# PER CLASS MAP
# ============================================================

def evaluate_per_class_map(
    ground_truth_file,
    prediction_file
):

    coco_gt = COCO(
        str(
            ground_truth_file
        )
    )

    coco_dt = coco_gt.loadRes(
        str(
            prediction_file
        )
    )

    results = {}

    for category_id in range(
        1,
        len(CLASSES) + 1
    ):

        evaluator = COCOeval(
            coco_gt,
            coco_dt,
            "bbox"
        )

        evaluator.params.catIds = [
            category_id
        ]

        evaluator.evaluate()
        evaluator.accumulate()

        precision = (
            evaluator.eval[
                "precision"
            ]
        )

        valid = (
            precision[
                precision > -1
            ]
        )

        map_50_95 = (
            float(
                np.mean(valid)
            )
            if len(valid)
            else 0.0
        )

        precision_50 = (
            precision[0]
        )

        valid_50 = (
            precision_50[
                precision_50 > -1
            ]
        )

        map_50 = (
            float(
                np.mean(valid_50)
            )
            if len(valid_50)
            else 0.0
        )

        results[
            COCO_ID_TO_CLASS[
                category_id
            ]
        ] = {
            "mAP50":
                map_50,

            "mAP50_95":
                map_50_95
        }

    return results


# ============================================================
# PRECISION / RECALL / F1
# ============================================================

def evaluate_threshold(
    all_predictions,
    image_files,
    image_metadata,
    threshold
):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    class_stats = {
        class_id: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
        for class_id
        in range(
            len(CLASSES)
        )
    }

    prediction_lookup = {
        item["file_name"]:
            item[
                "predictions"
            ]
        for item
        in all_predictions
    }

    for image_path in image_files:

        meta = (
            image_metadata[
                image_path.name
            ]
        )

        width = (
            meta["width"]
        )

        height = (
            meta["height"]
        )

        label_path = (
            LABEL_DIR
            /
            f"{image_path.stem}.txt"
        )

        gt_objects = (
            read_yolo_ground_truth(
                label_path,
                width,
                height
            )
        )

        predictions = [
            p
            for p in (
                prediction_lookup.get(
                    image_path.name,
                    []
                )
            )
            if (
                p["score"]
                >= threshold
            )
        ]

        for class_id in range(
            len(CLASSES)
        ):

            gt_class = [
                g
                for g
                in gt_objects
                if (
                    g["class_id"]
                    == class_id
                )
            ]

            pred_class = [
                p
                for p
                in predictions
                if (
                    p["class_id"]
                    == class_id
                )
            ]

            pred_class = sorted(
                pred_class,
                key=lambda x:
                    x["score"],
                reverse=True
            )

            matched_gt = set()

            tp = 0
            fp = 0

            for pred in pred_class:

                best_iou = 0.0
                best_gt_index = None

                for (
                    gt_index,
                    gt
                ) in enumerate(
                    gt_class
                ):

                    if (
                        gt_index
                        in matched_gt
                    ):
                        continue

                    iou = box_iou(
                        pred["box"],
                        gt["box"]
                    )

                    if (
                        iou
                        > best_iou
                    ):

                        best_iou = iou
                        best_gt_index = (
                            gt_index
                        )

                if (
                    best_gt_index
                    is not None
                    and
                    best_iou
                    >= MATCH_IOU
                ):

                    tp += 1

                    matched_gt.add(
                        best_gt_index
                    )

                else:

                    fp += 1

            fn = (
                len(gt_class)
                -
                len(matched_gt)
            )

            class_stats[
                class_id
            ]["tp"] += tp

            class_stats[
                class_id
            ]["fp"] += fp

            class_stats[
                class_id
            ]["fn"] += fn

            total_tp += tp
            total_fp += fp
            total_fn += fn

    precision = (
        total_tp
        /
        (
            total_tp
            +
            total_fp
        )
        if (
            total_tp
            +
            total_fp
            > 0
        )
        else 0.0
    )

    recall = (
        total_tp
        /
        (
            total_tp
            +
            total_fn
        )
        if (
            total_tp
            +
            total_fn
            > 0
        )
        else 0.0
    )

    f1 = (
        2
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        )
        if (
            precision
            +
            recall
            > 0
        )
        else 0.0
    )

    per_class = {}

    for (
        class_id,
        values
    ) in class_stats.items():

        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]

        p = (
            tp
            /
            (tp + fp)
            if (
                tp + fp
                > 0
            )
            else 0.0
        )

        r = (
            tp
            /
            (tp + fn)
            if (
                tp + fn
                > 0
            )
            else 0.0
        )

        class_f1 = (
            2
            *
            p
            *
            r
            /
            (p + r)
            if (
                p + r
                > 0
            )
            else 0.0
        )

        per_class[
            CLASSES[class_id]
        ] = {
            "precision":
                p,

            "recall":
                r,

            "f1":
                class_f1,

            "tp":
                tp,

            "fp":
                fp,

            "fn":
                fn
        }

    return {
        "threshold":
            float(threshold),

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "tp":
            total_tp,

        "fp":
            total_fp,

        "fn":
            total_fn,

        "per_class":
            per_class
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "GROUNDING DINO TEST - NIGHT"
    )

    print(
        "=" * 70
    )

    print(
        "\nCUDA available:",
        torch.cuda.is_available()
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        )
    )

    image_files = sorted(
        IMAGE_DIR.glob(
            "*.jpg"
        )
    )

    print(
        "\nTest night images:",
        len(image_files)
    )

    if len(
        image_files
    ) == 0:

        raise RuntimeError(
            f"No images found "
            f"in {IMAGE_DIR}"
        )

    print(
        "\nBuilding COCO ground truth..."
    )

    coco_gt, image_metadata = (
        build_coco_ground_truth(
            image_files
        )
    )

    with open(
        COCO_GT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            coco_gt,
            f,
            indent=2
        )

    print(
        "Ground-truth objects:",
        len(
            coco_gt[
                "annotations"
            ]
        )
    )

    print(
        "\nLoading Grounding DINO..."
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_ID
        )
    )

    model = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(
            MODEL_ID
        )
        .to(
            DEVICE
        )
    )

    model.eval()

    print(
        "Model loaded on:",
        next(
            model.parameters()
        ).device
    )

    all_predictions = (
        run_inference(
            model,
            processor,
            image_files,
            image_metadata
        )
    )

    number_predictions = sum(
        len(
            x[
                "predictions"
            ]
        )
        for x
        in all_predictions
    )

    print(
        "\nTotal retained predictions:",
        number_predictions
    )

    coco_predictions = (
        predictions_to_coco(
            all_predictions
        )
    )

    with open(
        COCO_PREDICTIONS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            coco_predictions,
            f,
            indent=2
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "COCO MAP EVALUATION"
    )

    print(
        "=" * 70
    )

    map50, map50_95 = (
        evaluate_map(
            COCO_GT_FILE,
            COCO_PREDICTIONS_FILE
        )
    )

    per_class_map = (
        evaluate_per_class_map(
            COCO_GT_FILE,
            COCO_PREDICTIONS_FILE
        )
    )

    final_result = (
        evaluate_threshold(
            all_predictions,
            image_files,
            image_metadata,
            FROZEN_CONFIDENCE_THRESHOLD
        )
    )

    summary = {
        "split":
            "test_night",

        "model":
            MODEL_ID,

        "images":
            len(
                image_files
            ),

        "frozen_parameters": {
            "prediction_box_threshold":
                PREDICTION_BOX_THRESHOLD,

            "text_threshold":
                TEXT_THRESHOLD,

            "same_class_nms_iou":
                SAME_CLASS_NMS_IOU,

            "cross_class_duplicate_iou":
                CROSS_CLASS_DUPLICATE_IOU,

            "confidence_threshold":
                FROZEN_CONFIDENCE_THRESHOLD
        },

        "mAP50":
            map50,

        "mAP50_95":
            map50_95,

        "precision":
            final_result[
                "precision"
            ],

        "recall":
            final_result[
                "recall"
            ],

        "f1":
            final_result[
                "f1"
            ],

        "per_class":
            final_result[
                "per_class"
            ],

        "per_class_map":
            per_class_map
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "GROUNDING DINO TEST NIGHT RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nConfidence threshold: "
        f"{FROZEN_CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        f"\nPrecision:    "
        f"{final_result['precision']:.4f}"
    )

    print(
        f"Recall:       "
        f"{final_result['recall']:.4f}"
    )

    print(
        f"F1:           "
        f"{final_result['f1']:.4f}"
    )

    print(
        f"mAP@50:       "
        f"{map50:.4f}"
    )

    print(
        f"mAP@50-95:    "
        f"{map50_95:.4f}"
    )

    print(
        "\nPer-class results:"
    )

    for class_name in CLASSES:

        pr = (
            final_result[
                "per_class"
            ][class_name]
        )

        ap = (
            per_class_map[
                class_name
            ]
        )

        print(
            f"\n{class_name}"
        )

        print(
            f"  Precision:  "
            f"{pr['precision']:.4f}"
        )

        print(
            f"  Recall:     "
            f"{pr['recall']:.4f}"
        )

        print(
            f"  F1:         "
            f"{pr['f1']:.4f}"
        )

        print(
            f"  mAP50:      "
            f"{ap['mAP50']:.4f}"
        )

        print(
            f"  mAP50-95:   "
            f"{ap['mAP50_95']:.4f}"
        )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nTEST NIGHT COMPLETE."
    )


if __name__ == "__main__":
    main()