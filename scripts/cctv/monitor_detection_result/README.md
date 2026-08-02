# monitor_detection_result.py

Standalone diagnostic tool that runs YOLO inference plus the production
post-processing pipeline (`utils/post_processing.py`) over an input video and
writes one annotated JPG per processed frame.

By default, only detections at or above `conf_thresh` are shown — the same
set production's inference call would ever see. Every one of those is kept
in the output (nothing past that point is silently dropped) and color-coded
by the reason it was filtered (or kept) at each pipeline stage, so you can
visually inspect *why* a given detection did or didn't end up being
reported. Pass `--conf_thresh_low` with a lower value to additionally
surface sub-`conf_thresh` detections when investigating a miss.

---

## Why this exists

The production pipeline (`spacenorm_obj_detection.py`,
`offline_spacenorm_obj_detection.py`) only shows you the boxes that survive
every filter. When a person isn't reported, there's no way to tell from the
live output alone whether it was because of `conf_thresh`, the ROI polygon,
`min_obj_size_ratio`, or the MOG2 background/motion check. This script runs
the exact same filter functions but retains and tags every dropped box so
that question can be answered by looking at one image.

---

## Requirements

The environment needs `ultralytics`, `torch`, `opencv-python`, and `numpy`
(the same stack the main service uses). On this machine that's the
`PyTorch` conda env:

```bash
/home/ahnjw/anaconda3/envs/PyTorch/bin/python \
  scripts/cctv/monitor_detection_result/monitor_detection_result.py ...
```

The system `python3` does **not** have these packages installed.

---

## Usage

By default this script reproduces exactly the configuration a given site's
live `spacenorm_obj_detection` service runs with, so results are comparable
to what's happening in production:

```bash
python monitor_detection_result.py \
  --video <path-to-video> \
  [--site <site-name>] [--camera <camera-name>] \
  [--report_period <seconds>] \
  [--output_dir <dir>] \
  [--model <path>] [--conf_thresh <val>] \
  [--conf_thresh_low <val>]
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--video` | yes | — | Input video file |
| `--site` | only if `--camera` given | `None` | Site identifier (e.g. `cym`, `jaeil_cr`, `kumho`). Resolves `spacenorm_cfg/behavior/overrides/<site>.json` (merged over `default.json` if it exists) and `spacenorm_cfg/cctv/cctv_<site>.json` (for `--camera` lookup) |
| `--camera` | no | `None` | Camera name key to look up inside `spacenorm_cfg/cctv/cctv_<site>.json`, for ROI + `min_obj_size_ratio` |
| `--report_period` | no | the site's configured `report_period` (`default.json`, 3.0s unless overridden) | Seconds between sampled frames, decimated using the video's own FPS — reproduces the live per-camera processing cadence. Pass `0` to process every frame instead |
| `--output_dir` | no | `<video_stem>_monitor/` next to the video | Directory the annotated JPGs are written to |
| `--conf_thresh_low` | no | same as `conf_thresh` (config value, or `--conf_thresh` if given) | Confidence used for the raw YOLO inference pass. Lower this (e.g. `0.001`) to additionally surface sub-`conf_thresh` detections for investigation. Not a production parameter — specific to this diagnostic tool |
| `--conf_thresh` | no | from config | Override `conf_thresh` |
| `--model` | no | `<script_dir>/<basename of config's model path>` | Path to the `.pt` weights file |

### Model weights

`default.json`'s `model` field is a container-relative path (e.g.
`spacenorm_obj_detection/yolo11x_set01-0148.pt`) meant for the live service's
own directory layout — it does not apply here. Without `--model`, this
script takes just the **basename** of that config value and looks for it
next to `monitor_detection_result.py` itself, i.e.
`scripts/cctv/monitor_detection_result/yolo11x_set01-0148.pt`. Copy or
symlink the weights file there, or pass `--model <path>` to point at one
elsewhere (e.g. `docker_build/yolo11x_set01-0148.pt`).

If the resolved path doesn't exist, the script stops immediately with an
error. It **never** falls back to downloading a model — passing a
nonexistent path straight to `ultralytics.YOLO()` will silently fetch a
generic pretrained checkpoint from the internet and save it wherever that
path resolves to, which is exactly the failure mode this check exists to
avoid.

`spacenorm_cfg/behavior/default.json` is always loaded (it's the common
config for every site). Without `--site`, only those defaults apply — no
per-site override, and (without `--camera`) no ROI/`min_obj_size_ratio`
filtering or polygon drawing, matching production behavior for a camera
without ROI configured. `img_size`, `background_thresh1/2/3`,
`conf_thresh1/2_background`, `remove_background_bb`, etc. all come from the
resolved config exactly as production would use them — there's no override
flag for those (yet); pass `--site`/`--camera` for reproducibility, or edit
the underlying config JSON if you need a hypothetical value.

---

## Pipeline stages

Each raw detection moves through the same stages as production, in the same
order, using the actual functions from `utils/post_processing.py`:

```
YOLO inference (conf = conf_thresh_low, default = conf_thresh)
    -> conf_thresh gate                    (only visible if conf_thresh_low < conf_thresh)
    -> filter_only_person()                (drop non-person classes)
    -> filter_small_objects()              (drop boxes under min_obj_size_ratio, if set)
    -> remove_outside_ROI()                (drop boxes outside the ROI polygon, if set)
    -> check_bb_on_background()            (MOG2 motion/background classification)
```

A box is tagged with the *first* stage that removes it; it does not proceed
further down the pipeline once dropped (matching production). Attribution is
done by tracking Python object identity through each filter call — the
`post_processing.py` functions only ever append references from their input
lists, so this reliably determines which boxes survived each stage without
reimplementing any filter logic.

Since `conf_thresh_low` defaults to `conf_thresh`, the "below conf_thresh"
stage produces no boxes by default — it only activates when you explicitly
pass a lower `--conf_thresh_low`.

`check_bb_on_background()` is always called with `remove_member=False` so
every surviving box gets a `type` (0, -1, -2, 1, 2, or 3), regardless of the
`remove_background_bb` config flag — that flag only affects the displayed
reason string (`background_t*` vs `kept_background_t*`), not the coloring.

**Note on MOG2 continuity:** the background subtractor is only updated on the
*sampled* frames actually processed by this script, not on every raw frame of
the video — the motion model advances one step per output frame rather than
per real-time frame. This is simpler and was chosen deliberately over
continuous per-frame updates; it means the background/motion classification
here isn't a perfect match for what production would do on the same footage
in real time when `--report_period` skips frames.

---

## Color legend

Drawn as a small color-key panel in the top-right corner of every output JPG
(top-right rather than top-left, since CCTV OSD timestamps/labels commonly
occupy the top-left or bottom of the frame).

| Reason | Color (BGR) | Meaning |
|---|---|---|
| below conf_thresh | dark gray `(105,105,105)` | Detected by the model, but below `conf_thresh` — never reaches production's `filter_only_person()`. Only appears if `--conf_thresh_low` is explicitly set below `conf_thresh` |
| not person | magenta `(255,0,255)` | Non-person class (label shows the class name) |
| too small | yellow `(0,255,255)` | Bounding box area ratio below `min_obj_size_ratio` |
| outside ROI | red `(0,0,255)` | Box center falls outside every ROI polygon |
| background (t1/t2) | orange `(0,165,255)` | MOG2 type 1/2 — medium/low motion, low confidence |
| background (t3) | gray `(128,128,128)` | MOG2 type 3 — very low motion |
| kept, low motion | cyan `(255,255,0)` | MOG2 type -1/-2 — kept despite low motion due to high confidence |
| kept, reported | green `(0,255,0)` | MOG2 type 0 — final, would be reported |
| ROI polygon outline | white `(255,255,255)` | Camera's configured ROI region(s) |

Each box also has a small text label with its reason, confidence, and (for
the background stage) motion score.

---

## Output

One JPG per sampled frame, named:

```
<video_stem>_frame<frame_idx padded to 6 digits>_t<seconds>s.jpg
```

e.g. `test_clip_frame001380_t45.85s.jpg`.

Once all frames are processed, every JPG in `output_dir` is bundled into a
single archive in the same directory:

```
<output_dir>/<video_stem>_monitor.tar.gz
```

e.g. `test_clip_monitor.tar.gz`, for easy download/transfer off the box.

---

## Example

```bash
/home/ahnjw/anaconda3/envs/PyTorch/bin/python \
  scripts/cctv/monitor_detection_result/monitor_detection_result.py \
  --video scripts/cctv/video_to_image/01.videos/BlueDragon_W1/test_clip.mp4 \
  --site cym \
  --camera "1F_dock-roadside1" \
  --output_dir /tmp/monitor_out
```

This reproduces cym's live configuration exactly (default.json + any
`overrides/cym.json`, ROI/`min_obj_size_ratio` for `1F_dock-roadside1` from
`cctv_cym.json`, and cym's `report_period` for frame sampling). Add
`--report_period 2` to sample every 2 seconds instead, or `--model
<path>`/`--conf_thresh <val>` to test a hypothetical change against the same
site config.
