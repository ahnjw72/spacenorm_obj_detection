#!/usr/bin/env python3
"""Evaluate mAP of a YOLO model against a ground-truth image/label set.

See README.md in this directory for usage and the ground-truth format.
"""

import argparse
import glob
import json
import re
import sys
import tempfile
from pathlib import Path

CLASS_NAMES = ["person", "bird", "cat", "dog", "horse", "sheep", "cow"]
IMAGE_GLOBS = ["*.jpg", "*.png"]

INT_RE = re.compile(r"-?\d+")


def resolve_gt_dirs(patterns):
    dirs = []
    seen = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise SystemExit(f"error: --gt-path pattern matched nothing: {pattern!r}")
        for m in matches:
            p = Path(m)
            if not p.is_dir():
                continue
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                dirs.append(resolved)
    if not dirs:
        raise SystemExit("error: no directories matched any --gt-path pattern")
    return dirs


def collect_images(dirs):
    images = []
    for d in dirs:
        for pattern in IMAGE_GLOBS:
            images.extend(sorted(d.glob(pattern)))
    return images


def resolve_gt_list_files(patterns):
    files = []
    seen = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise SystemExit(f"error: --gt-list pattern matched nothing: {pattern!r}")
        for m in matches:
            p = Path(m)
            if not p.is_file():
                continue
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)
    if not files:
        raise SystemExit("error: no files matched any --gt-list pattern")
    return files


def read_image_list(list_files):
    images = []
    seen = set()
    missing = []
    for list_file in list_files:
        for line in list_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            resolved = Path(line).resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if not resolved.is_file():
                missing.append(line)
                continue
            images.append(resolved)
    if missing:
        preview = "\n".join(f"  {m}" for m in missing[:10])
        more = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise SystemExit(
            f"error: {len(missing)} image(s) listed in --gt-list do not exist on disk:\n{preview}{more}"
        )
    return images


def validate_label_line(fields):
    if len(fields) != 5:
        return f"expected 5 fields, got {len(fields)}"
    class_id_str, *coord_strs = fields
    if not INT_RE.fullmatch(class_id_str):
        return f"class id {class_id_str!r} is not an integer"
    class_id = int(class_id_str)
    if not (0 <= class_id < len(CLASS_NAMES)):
        return f"class id {class_id} out of range [0, {len(CLASS_NAMES) - 1}]"
    for name, val in zip(("cx", "cy", "w", "h"), coord_strs):
        try:
            f = float(val)
        except ValueError:
            return f"{name}={val!r} is not a float"
        if not (0.0 <= f <= 1.0):
            return f"{name}={f} out of range [0.0, 1.0]"
    return None


def partition_valid_images(images):
    valid = []
    skipped = []
    for img in images:
        label = img.with_suffix(".txt")
        if not label.exists():
            skipped.append((img, f"missing label file {label.name}"))
            continue
        bad = None
        for lineno, line in enumerate(label.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            reason = validate_label_line(line.split())
            if reason:
                bad = f"{label.name}:{lineno}: {reason} (line: {line!r})"
                break
        if bad:
            skipped.append((img, bad))
        else:
            valid.append(img)
    return valid, skipped


def run_eval(images, model_path, imgsz):
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory(prefix="eval_map_") as tmp:
        tmp = Path(tmp)
        list_path = (tmp / "images.txt").resolve()
        list_path.write_text("\n".join(str(p) for p in images) + "\n")

        yaml_path = (tmp / "dataset.yaml").resolve()
        yaml_path.write_text(
            "path: .\n"
            f"train: {list_path}\n"
            f"val: {list_path}\n"
            f"nc: {len(CLASS_NAMES)}\n"
            f"names: {CLASS_NAMES!r}\n"
        )

        model = YOLO(model_path)
        return model.val(data=str(yaml_path), imgsz=imgsz, split="val")


def build_report(results, input_mode, gt_patterns, resolved_inputs, model_path, imgsz, num_images_found, skipped):
    box = results.box
    per_class = {name: None for name in CLASS_NAMES}
    for row in results.summary():
        per_class[row["Class"]] = {
            "images": int(row["Images"]),
            "instances": int(row["Instances"]),
            "precision": float(row["Box-P"]),
            "recall": float(row["Box-R"]),
            "map50": float(row["mAP50"]),
            "map50_95": float(row["mAP50-95"]),
        }

    return {
        "input_mode": input_mode,
        "gt_paths": list(gt_patterns),
        "resolved_inputs": [str(p) for p in resolved_inputs],
        "model": str(model_path),
        "imgsz": imgsz,
        "num_images": num_images_found - len(skipped),
        "num_images_found": num_images_found,
        "num_images_skipped": len(skipped),
        "skipped_images": [{"image": str(img), "reason": reason} for img, reason in skipped],
        "summary": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "per_class": per_class,
    }


def print_summary_table(report):
    header = f"{'Class':<10}{'Images':>8}{'Instances':>11}{'P':>10}{'R':>10}{'mAP50':>10}{'mAP50-95':>10}"
    print()
    print(header)

    s = report["summary"]
    total_instances = sum(c["instances"] for c in report["per_class"].values() if c)
    print(
        f"{'all':<10}{report['num_images']:>8}{total_instances:>11}"
        f"{s['precision']:>10.3f}{s['recall']:>10.3f}{s['map50']:>10.3f}{s['map50_95']:>10.3f}"
    )

    for name, c in report["per_class"].items():
        if c is None:
            print(f"{name:<10}{0:>8}{0:>11}{'--':>10}{'--':>10}{'--':>10}{'--':>10}")
        else:
            print(
                f"{name:<10}{c['images']:>8}{c['instances']:>11}"
                f"{c['precision']:>10.3f}{c['recall']:>10.3f}{c['map50']:>10.3f}{c['map50_95']:>10.3f}"
            )
    print()

    if report["num_images_skipped"]:
        print(
            f"Skipped {report['num_images_skipped']} image(s) with missing/malformed "
            "annotations -- see stderr / JSON skipped_images"
        )
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    gt_group = parser.add_mutually_exclusive_group(required=True)
    gt_group.add_argument(
        "--gt-path",
        nargs="+",
        help="Ground-truth directory path(s), glob patterns allowed (e.g. 'data/cctv_train_data/set013*')",
    )
    gt_group.add_argument(
        "--gt-list",
        nargs="+",
        help="Image-list text file(s), glob patterns allowed (e.g. "
        "'data/cctv_train_data/train_test_txts/test_01_to_0150.txt'). Each line is a "
        "repo-root-relative image path; a same-named '.txt' label file is expected alongside it.",
    )
    parser.add_argument("--model", required=True, help="Path to YOLO .pt weights")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size (default: 1280)")
    parser.add_argument("--output", help="Optional path to also write the JSON report to")
    args = parser.parse_args()

    if args.gt_path:
        input_mode = "gt-path"
        gt_patterns = args.gt_path
        resolved_inputs = resolve_gt_dirs(args.gt_path)
        images = collect_images(resolved_inputs)
        if not images:
            raise SystemExit("error: no .jpg/.png images found in resolved --gt-path directories")
    else:
        input_mode = "gt-list"
        gt_patterns = args.gt_list
        resolved_inputs = resolve_gt_list_files(args.gt_list)
        images = read_image_list(resolved_inputs)
        if not images:
            raise SystemExit("error: no images listed in resolved --gt-list files")

    valid_images, skipped = partition_valid_images(images)
    if skipped:
        print(f"Skipping {len(skipped)} image(s) with missing/malformed annotations:", file=sys.stderr)
        for img, reason in skipped:
            print(f"  {img}: {reason}", file=sys.stderr)

    if not valid_images:
        raise SystemExit("error: no valid image/label pairs remain after filtering")

    results = run_eval(valid_images, args.model, args.imgsz)
    report = build_report(
        results, input_mode, gt_patterns, resolved_inputs, args.model, args.imgsz, len(images), skipped
    )

    print_summary_table(report)
    print(json.dumps(report, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"JSON report written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
