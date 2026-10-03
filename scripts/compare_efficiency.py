import json
from pathlib import Path


PROJECT_ROOT = Path(
    r"C:\msc-object-detection"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "efficiency_benchmark"
)

YOLO_FILE = (
    RESULT_DIR
    / "yolo_benchmark.json"
)

DINO_FILE = (
    RESULT_DIR
    / "grounding_dino_benchmark.json"
)


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def main():

    yolo = load_json(
        YOLO_FILE
    )

    dino = load_json(
        DINO_FILE
    )

    print("=" * 95)

    print(
        "FINAL EFFICIENCY COMPARISON"
    )

    print("=" * 95)

    print(
        f"{'Metric':<30}"
        f"{'YOLO11n':>25}"
        f"{'Grounding DINO':>30}"
    )

    print("-" * 95)

    print(
        f"{'Parameters':<30}"
        f"{yolo['parameters']:>25,}"
        f"{dino['parameters']:>30,}"
    )

    print(
        f"{'Mean latency (ms)':<30}"
        f"{yolo['mean_latency_ms']:>25.2f}"
        f"{dino['mean_latency_ms']:>30.2f}"
    )

    print(
        f"{'Median latency (ms)':<30}"
        f"{yolo['median_latency_ms']:>25.2f}"
        f"{dino['median_latency_ms']:>30.2f}"
    )

    print(
        f"{'FPS':<30}"
        f"{yolo['fps']:>25.2f}"
        f"{dino['fps']:>30.2f}"
    )

    print(
        f"{'Peak GPU memory (MB)':<30}"
        f"{yolo['peak_gpu_memory_mb']:>25.2f}"
        f"{dino['peak_gpu_memory_mb']:>30.2f}"
    )

    print(
        f"{'Extra inference memory (MB)':<30}"
        f"{yolo['incremental_peak_gpu_memory_mb']:>25.2f}"
        f"{dino['incremental_peak_gpu_memory_mb']:>30.2f}"
    )

    print("=" * 95)

    latency_ratio = (
        dino[
            "mean_latency_ms"
        ]
        /
        yolo[
            "mean_latency_ms"
        ]
    )

    fps_ratio = (
        yolo["fps"]
        /
        dino["fps"]
    )

    parameter_ratio = (
        dino["parameters"]
        /
        yolo["parameters"]
    )

    print(
        f"\nGrounding DINO latency / YOLO latency: "
        f"{latency_ratio:.2f}x"
    )

    print(
        f"YOLO FPS / Grounding DINO FPS: "
        f"{fps_ratio:.2f}x"
    )

    print(
        f"Grounding DINO parameters / YOLO parameters: "
        f"{parameter_ratio:.2f}x"
    )

    print(
        "\nComparison complete."
    )


if __name__ == "__main__":
    main()