# eval_map.py

Evaluate the mAP of a YOLO model against a ground-truth image/label set, using the
Ultralytics Python API (`YOLO(...).val(...)`) under the hood.

## Requirements

Run under the `PyTorch` conda environment (has `ultralytics`, `torch`, `cv2` installed):

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate PyTorch
```

## Usage

`--gt-path`/`--gt-list` and `--model` are resolved relative to the current working
directory, so **run these commands from the repository root**
(`spacenorm_obj_detection/`), the same way as every other command in this project's
`CLAUDE.md`:

```bash
python scripts/cctv/eval_map/eval_map.py (--gt-path <path> [<path> ...] | --gt-list <file> [<file> ...]) --model <weights.pt> [--imgsz 1280] [--output report.json]
```

`--gt-path` and `--gt-list` are mutually exclusive — pick exactly one input mode per run.

### Examples

Single ground-truth directory:

```bash
python scripts/cctv/eval_map/eval_map.py \
  --gt-path data/cctv_train_data/set0149 \
  --model docker_build/yolo11x_set01-0148.pt \
  --imgsz 1280
```

Multiple directories, glob patterns allowed (each pattern is expanded independently, results deduplicated):

```bash
python scripts/cctv/eval_map/eval_map.py \
  --gt-path "data/cctv_train_data/set013*" data/cctv_train_data/set0149 \
  --model docker_build/yolo11x_set01-0148.pt \
  --imgsz 1280 \
  --output report.json
```

Evaluate only a held-out split, using one of the image-list files under
`data/cctv_train_data/train_test_txts/` instead of a whole `setNNNN` directory — this
avoids scoring the model on images it also saw during training, which is necessary for
a fair precision/recall comparison between checkpoints:

```bash
python scripts/cctv/eval_map/eval_map.py \
  --gt-list data/cctv_train_data/train_test_txts/test_01_to_0150.txt \
  --model docker_build/yolo11x_set01-0148.pt \
  --imgsz 1280
```

Make sure the list file actually matches the checkpoint's training-time split (the
project may have multiple cumulative split files, one per `SET_END_NUM` used when
`make_train_test.py` was last run) — evaluating on a different split's test file than
the one used to withhold data from that checkpoint's training run defeats the purpose.

## Arguments

| Argument | Required | Description |
|---|---|---|
| `--gt-path` | one of `--gt-path`/`--gt-list` | One or more ground-truth directory paths or glob patterns (e.g. `"data/cctv_train_data/set013*"`). Each pattern is expanded with `glob.glob`; only matched directories are used. Directories are **not** searched recursively — each must directly contain the image/label pairs. |
| `--gt-list` | one of `--gt-path`/`--gt-list` | One or more image-list text file paths or glob patterns (e.g. `data/cctv_train_data/train_test_txts/test_01_to_0150.txt`). Each line is a repo-root-relative image path; blank lines are skipped. A same-named `.txt` label file is expected alongside each listed image, same as directory mode. |
| `--model` | yes | Path to a YOLO `.pt` weights file. |
| `--imgsz` | no | Inference image size, default `1280` (matches `spacenorm_cfg/behavior/default.json`'s `img_size`). |
| `--output` | no | If given, also writes the JSON report to this file (it is always printed to stdout). |

## Ground-truth format

Each `--gt-path` directory must contain `*.jpg` and/or `*.png` images with a same-named
`*.txt` label file next to each one (standard YOLO layout, flat — no `images/`/`labels/`
subdirectories). Both extensions are collected from the same directory, so a set mixing
JPEG and PNG frames (e.g. from `dataset_builder`'s lossless PNG staging) works as-is.

`--gt-list` files follow the same image/label pairing convention, just enumerated
explicitly one path per line instead of implied by directory contents. Every listed
image must exist on disk — the script fails fast (before running any inference) if a
list references an image that's missing, since that indicates a stale list file rather
than a legitimately unlabeled image.

Each label file uses the standard YOLO annotation format, one object per line:

```
<class_id> <cx> <cy> <w> <h>
```

- `class_id`: integer, one of the 7 fixed classes used by this project (index: name):
  `0: person, 1: bird, 2: cat, 3: dog, 4: horse, 5: sheep, 6: cow`
- `cx cy w h`: floats normalized to `[0.0, 1.0]` (center x/y, width, height, relative to
  image dimensions).
- An empty label file is valid — it means the image is a pure background (no objects).

### Handling of missing/malformed annotations

Before running evaluation, the script checks every image/label pair and **skips** (does
not evaluate) any image with a problem, printing every skip reason to stderr (not just
the first):

- Missing label file for an image → skipped.
- A label line without exactly 5 fields → whole image skipped.
- Non-integer `class_id`, or one outside `[0, 6]` → whole image skipped.
- Non-float coordinate, or a coordinate outside `[0.0, 1.0]` → whole image skipped.

A malformed line skips its *entire image* rather than just that line, since dropping only
the bad line would silently under-annotate the image and skew the mAP result. The run
only aborts if **no** valid image/label pairs remain at all. This makes it safe to point
`--gt-path` at a broad glob covering many sets at once (e.g. `data/cctv_train_data/set*`)
even if a handful of images/sets are incomplete or still being labeled — those are
excluded from the aggregated result and reported, everything else still gets evaluated
together.

## Output

The script prints a human-readable summary table (same shape as `yolo val`'s CLI
output) followed by a JSON report:

```json
{
  "input_mode": "gt-path",
  "gt_paths": ["data/cctv_train_data/set0149"],
  "resolved_inputs": ["/abs/path/data/cctv_train_data/set0149"],
  "model": "docker_build/yolo11x_set01-0148.pt",
  "imgsz": 1280,
  "num_images": 205,
  "num_images_found": 207,
  "num_images_skipped": 2,
  "skipped_images": [
    {"image": "/abs/path/data/cctv_train_data/set0149/foo.jpg", "reason": "missing label file foo.txt"},
    {"image": "/abs/path/data/cctv_train_data/set0149/bar.jpg", "reason": "bar.txt:1: expected 5 fields, got 3 (line: '0 0.5 0.5')"}
  ],
  "summary": {
    "precision": 0.869,
    "recall": 0.852,
    "map50": 0.928,
    "map50_95": 0.693
  },
  "per_class": {
    "person": {
      "images": 77,
      "instances": 78,
      "precision": 0.869,
      "recall": 0.852,
      "map50": 0.928,
      "map50_95": 0.693
    },
    "bird": null,
    "cat": null,
    "dog": null,
    "horse": null,
    "sheep": null,
    "cow": null
  }
}
```

`input_mode` records which of `--gt-path`/`--gt-list` was used; `resolved_inputs` holds
the resolved directories (`gt-path` mode) or resolved list files (`gt-list` mode) after
glob expansion. `num_images` is the count actually evaluated (i.e.
`num_images_found - num_images_skipped`); `skipped_images` lists every excluded image
and why, mirroring what's printed to stderr.

A class with `null` per-class metrics means it had zero ground-truth instances across
all resolved inputs (Ultralytics does not compute per-class metrics for classes absent
from the ground truth).

Ultralytics also writes its own run artifacts (confusion matrix, PR curves, etc.) to
`runs/detect/val*` as a side effect of calling `.val()` — this is standard Ultralytics
behavior, not something this script manages.
