"""labelstudio_export.py — turn mined frames into a Label Studio import file.

Each task carries the pre-labels as a *prediction* so the reviewer starts from
almost-correct boxes and only corrects. Suspected false positives are included
as a distinctly-labeled prediction so the reviewer can see and delete them.

Images are referenced via Label Studio's local-file serving. On the review host:

    export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<PARENT of staging_dir>

Note the document root is the **parent** of ``staging_dir``, not ``staging_dir``
itself: Label Studio refuses to register a Local Storage whose path equals the
document root, and ``staging_dir`` must be registrable so that one storage entry
covers every NVR beneath it. Task image URLs are therefore
``/data/local-files/?d=<path relative to that PARENT>`` — e.g.
``?d=reviewing/<nvr>/ch00/morning/<frame>.png``. ``build_dataset._ls_document_root``
computes this and bakes it into the generated ``run_labelstudio.sh``; setting the
root to ``staging_dir`` instead is the most common cause of the "issue loading URL
from $image" 404 (see README "Troubleshooting").

Import the tasks_*.json via "Import" in the LS project, review, then
"Export → YOLO".

Label config to use in the LS project (rectangle labels):

    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image">
        <Label value="person" background="green"/>
        <Label value="bird"/><Label value="cat"/><Label value="dog"/>
        <Label value="horse"/><Label value="sheep"/><Label value="cow"/>
        <Label value="SUSPECT_FP" background="red"/>
        <Label value="SUSPECT_STATIC_FP" background="darkred"/>
      </RectangleLabels>
    </View>
"""
import json
from urllib.parse import quote

from flicker_miner import TRAINSET_NAMES, frame_categories

# The project's labeling config (person + 6 animals + the two review-only suspect
# labels). Paste this into a new LS project's "Labeling Setup → Custom template",
# or copy it from the label_config.xml written next to each sweep's tasks file.
LABEL_CONFIG = """<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person" background="green"/>
    <Label value="bird"/><Label value="cat"/><Label value="dog"/>
    <Label value="horse"/><Label value="sheep"/><Label value="cow"/>
    <Label value="SUSPECT_FP" background="red"/>
    <Label value="SUSPECT_STATIC_FP" background="darkred"/>
  </RectangleLabels>
</View>
"""


def write_label_config(out_path):
    """Write the project labeling config XML (for copy/paste into a new project)."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(LABEL_CONFIG)


def _rect(box, w, h, label, rid, tid=None, reason=None, conf=None):
    """One Label Studio rectanglelabels result (coords in PERCENT of image).
    ``tid`` (the track id this box belongs to, within this clip) and
    ``reason`` (this box's own category tags, e.g. ``["Weak_FN",
    "Anchor_start"]``) ride as region ``meta``, visible to the reviewer when
    the region is selected, without needing their own Label config entry.
    LS renders ``meta.text`` list items concatenated with no separator, so
    both (and ``conf``, below) are joined into a single pre-formatted string
    rather than kept as separate list entries.

    ``conf`` is the detector's own confidence for this box, or ``None`` for
    an interpolated box (no real detection exists at that step to have a
    confidence). When present it is also set as the region's ``score`` —
    Label Studio's native per-region confidence field, which drives
    sorting/coloring in the Regions pane — in addition to ``meta.text``,
    since ``score`` support in the reviewer's LS version isn't guaranteed but
    ``meta.text`` already is (it is how ``tid``/``reason`` reach the
    reviewer today)."""
    x1, y1, x2, y2 = box
    result = {
        "id": rid,
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "original_width": w,
        "original_height": h,
        "image_rotation": 0,
        "value": {
            "x": 100.0 * x1 / w,
            "y": 100.0 * y1 / h,
            "width": 100.0 * (x2 - x1) / w,
            "height": 100.0 * (y2 - y1) / h,
            "rotation": 0,
            "rectanglelabels": [label],
        },
    }
    if conf is not None:
        result["score"] = conf
    parts = []
    if tid is not None:
        parts.append(f"track {tid}")
    if reason:
        parts.append(", ".join(reason))
    if conf is not None:
        parts.append(f"conf {conf:.2f}")
    if parts:
        result["meta"] = {"text": [", ".join(parts)]}
    return result


def build_task(record, image_rel_path):
    """Build one Label Studio task dict from a mined frame record.

    ``frame_idx`` and ``capture_time`` (ALGORITHM.md 9) exist because the Data
    Manager otherwise has no sortable time field at all: task IDs reflect
    IMPORT order, which is the sweep's channel-then-clip-then-frame nesting, not
    a true chronological or within-clip ordering once you filter or import
    across sweeps. ``frame_idx`` is this frame's raw video-frame index WITHIN its
    clip (from pass 1's decode; not renumbered by ``track_vid_stride`` or by
    which frames were selected) — sorting by it orders one clip's frames
    correctly regardless of import order. ``capture_time`` is the CLIP's start
    timestamp, ISO 8601 to the second (``YYYY-MM-DDTHH:MM:SS``, no milliseconds
    — sub-second precision does not exist at the clip-sampling granularity this
    tool operates at) — every frame from the same clip shares one value, so
    sorting by it orders clips chronologically across an entire sweep, or across
    several imported sweeps, independent of channel or import order.
    """
    w, h = record["width"], record["height"]
    results = []
    n = 0
    for (box, _src, tid, cats, conf) in record["persons"]:
        results.append(_rect(box, w, h, "person", f"r{n}", tid, cats, conf)); n += 1
    for (box, cls, conf) in record["animal_boxes"]:
        results.append(_rect(box, w, h, TRAINSET_NAMES[cls], f"r{n}", conf=conf)); n += 1
    for (box, kind, tid, conf) in record["suspect"]:
        label = "SUSPECT_STATIC_FP" if kind == "static" else "SUSPECT_FP"
        results.append(_rect(box, w, h, label, f"r{n}", tid, conf=conf)); n += 1

    # All of these live in `data` so they are filterable/sortable Data Manager
    # columns. There is no single "reason" — a frame belongs to any number of
    # categories at once, so filter on the counts directly, e.g.
    # n_anchor_start>0 OR n_anchor_end>0 (every frame with an interpolation
    # anchor), or clip_id=<...> to review one clip at a time. `n_easy` is the
    # one exception: it is 1 only when every other n_* count above is 0
    # (flicker_miner.frame_categories's fallback), so `n_easy=0` alone excludes
    # every easy frame in one filter instead of ANDing all the other counts to 0.
    return {
        "data": {
            "image": "/data/local-files/?d=" + quote(image_rel_path),
            "clip_id": record.get("clip_id"),
            "channel": record.get("channel"),
            "bucket": record.get("bucket"),
            "frame_idx": record.get("raw_idx"),
            "capture_time": record.get("capture_time"),
            "num_detected": len(results),
            "n_person": sum(1 for r in results
                            if r["value"]["rectanglelabels"] == ["person"]),
            "n_intpfn": record["n_intpfn"],
            "n_weakfn": record["n_weakfn"],
            "n_anchor_start": record["n_anchor_start"],
            "n_anchor_end": record["n_anchor_end"],
            "n_sfp": record["n_sfp"],
            "n_fp": record["n_fp"],
            "n_track_context": record["n_context"],
            "n_easy": int(frame_categories(record) == {"easy"}),
            "track_ids": record.get("track_ids", []),
        },
        "predictions": [{
            "model_version": "prelabel",
            "result": results,
        }],
    }


def write_tasks(tasks, out_json_path):
    """Write a list of tasks as a Label Studio JSON import file."""
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=1)
