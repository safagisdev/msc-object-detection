import json
import time
import statistics
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"C:\msc-object-detection")

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "yolo_full"
    / "weights"
    / "best.pt"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "yolo"
    / "images"
    / "test_day"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "efficiency_benchmark"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_DIR / "yolo_benchmark.json"


DEVICE = 0

IMAGE_SIZE = 640

BATCH_SIZE = 1

WARMUP_RUNS = 20

BENCHMARK_IMAGES = 100

CONFIDENCE_THRESHOLD = 0.25


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("YOLO11 EFFICIENCY BENCHMARK")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available."
        )

    print(
        "\nGPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

    # --------------------------------------------------------
    # Select same deterministic image sample
    # --------------------------------------------------------

    image_paths = sorted(
        IMAGE_DIR.glob("*.jpg")
    )

    if len(image_paths) < BENCHMARK_IMAGES:
        raise RuntimeError(
            f"Only {len(image_paths)} images found."
        )

    image_paths = image_paths[
        :BENCHMARK_IMAGES
    ]

    print(
        "\nBenchmark images:",
        len(image_paths)
    )

    print(
        "Image size:",
        IMAGE_SIZE
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Warm-up runs:",
        WARMUP_RUNS
    )

    # --------------------------------------------------------
    # Preload images into RAM
    # --------------------------------------------------------

    print(
        "\nPreloading images into RAM..."
    )

    images = []

    for path in image_paths:

        image = (
            Image.open(path)
            .convert("RGB")
            .copy()
        )

        images.append(image)

    print(
        "Images loaded."
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading YOLO model..."
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    # Force model to GPU before benchmark
    model.to(
        f"cuda:{DEVICE}"
    )

    print(
        "Model loaded."
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    parameter_count = sum(
        p.numel()
        for p in model.model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.model.parameters()
        if p.requires_grad
    )

    print(
        f"\nParameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(
        "\nRunning warm-up..."
    )

    for i in range(
        WARMUP_RUNS
    ):

        image = images[
            i % len(images)
        ]

        _ = model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            batch=BATCH_SIZE,
            device=DEVICE,
            verbose=False
        )

    torch.cuda.synchronize()

    print(
        "Warm-up complete."
    )

    # --------------------------------------------------------
    # GPU memory baseline
    # --------------------------------------------------------

    torch.cuda.empty_cache()

    torch.cuda.synchronize()

    baseline_memory = (
        torch.cuda.memory_allocated()
    )

    torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print(
        "\nRunning benchmark..."
    )

    latency_values = []

    total_start = time.perf_counter()

    for index, image in enumerate(
        images,
        start=1
    ):

        torch.cuda.synchronize()

        start = time.perf_counter()

        _ = model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            batch=BATCH_SIZE,
            device=DEVICE,
            verbose=False
        )

        torch.cuda.synchronize()

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000

        latency_values.append(
            latency_ms
        )

        if (
            index % 10
            == 0
        ):
            print(
                f"Processed "
                f"{index}/"
                f"{len(images)}"
            )

    total_end = time.perf_counter()

    total_time = (
        total_end
        - total_start
    )

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    peak_memory = (
        torch.cuda.max_memory_allocated()
    )

    incremental_peak = (
        peak_memory
        - baseline_memory
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mean_latency = (
        statistics.mean(
            latency_values
        )
    )

    median_latency = (
        statistics.median(
            latency_values
        )
    )

    min_latency = min(
        latency_values
    )

    max_latency = max(
        latency_values
    )

    latency_std = (
        statistics.stdev(
            latency_values
        )
        if len(latency_values) > 1
        else 0.0
    )

    fps = (
        len(images)
        / total_time
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    result = {

        "model":
            "YOLO11n full-data",

        "model_path":
            str(MODEL_PATH),

        "gpu":
            torch.cuda.get_device_name(0),

        "images":
            len(images),

        "batch_size":
            BATCH_SIZE,

        "image_size":
            IMAGE_SIZE,

        "warmup_runs":
            WARMUP_RUNS,

        "parameters":
            parameter_count,

        "trainable_parameters":
            trainable_parameters,

        "mean_latency_ms":
            mean_latency,

        "median_latency_ms":
            median_latency,

        "min_latency_ms":
            min_latency,

        "max_latency_ms":
            max_latency,

        "std_latency_ms":
            latency_std,

        "fps":
            fps,

        "total_benchmark_seconds":
            total_time,

        "baseline_gpu_memory_mb":
            baseline_memory
            / (1024 ** 2),

        "peak_gpu_memory_mb":
            peak_memory
            / (1024 ** 2),

        "incremental_peak_gpu_memory_mb":
            incremental_peak
            / (1024 ** 2)
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "YOLO BENCHMARK RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nParameters: "
        f"{parameter_count:,}"
    )

    print(
        f"\nMean latency: "
        f"{mean_latency:.2f} ms/image"
    )

    print(
        f"Median latency: "
        f"{median_latency:.2f} ms/image"
    )

    print(
        f"Latency std: "
        f"{latency_std:.2f} ms"
    )

    print(
        f"FPS: "
        f"{fps:.2f}"
    )

    print(
        f"\nBaseline GPU memory: "
        f"{baseline_memory / (1024 ** 2):.2f} MB"
    )

    print(
        f"Peak GPU memory: "
        f"{peak_memory / (1024 ** 2):.2f} MB"
    )

    print(
        f"Incremental peak memory: "
        f"{incremental_peak / (1024 ** 2):.2f} MB"
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nYOLO BENCHMARK COMPLETE."
    )


if __name__ == "__main__":
    main()