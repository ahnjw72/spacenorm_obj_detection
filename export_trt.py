"""export_trt.py

Convert a YOLOv11 .pt model to a TensorRT .engine file.

Usage:
    python export_trt.py --model yolo11x_set01-0148.pt [options]

The exported .engine file is written to the same directory as the input .pt file.
"""

import argparse
from pathlib import Path

# Disable Ultralytics telemetry before any YOLO import
from ultralytics.utils import SETTINGS
SETTINGS.update({'sync': False})

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv11 .pt model to TensorRT .engine")

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the input .pt model file",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device index (default: 0)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="Use FP16 precision (default: True — ~2x faster on RTX GPUs with negligible accuracy loss)",
    )
    parser.add_argument(
        "--no-half",
        dest="half",
        action="store_false",
        help="Disable FP16; use FP32 precision instead",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size to optimize for (default: 1)",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=4,
        help="TensorRT workspace size in GiB (default: 4)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"[export_trt] Input model : {model_path.resolve()}")
    print(f"[export_trt] Image size  : {args.imgsz}")
    print(f"[export_trt] Device      : cuda:{args.device}")
    print(f"[export_trt] Precision   : {'FP16 (half)' if args.half else 'FP32'}")
    print(f"[export_trt] Batch size  : {args.batch}")
    print(f"[export_trt] Workspace   : {args.workspace} GiB")
    print()

    model = YOLO(str(model_path))

    engine_path = model.export(
        format="engine",
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        batch=args.batch,
        workspace=args.workspace,
        verbose=True,
    )

    print()
    print(f"[export_trt] Export complete: {engine_path}")
    print()
    print("To use the engine in this project, update default.json:")
    print(f'  "model": {{ "value": "spacenorm_obj_detection/{Path(engine_path).name}", ... }}')


if __name__ == "__main__":
    main()
