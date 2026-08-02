#!/usr/bin/env python3
"""build_dataset.py — semi-automatic hard-example miner for the deployed CCTV model.

Samples short recorded clips across a time window (default: past 24 h, one
clip every clip_interval_min) from every channel of a Hanwha NVR (over SUNAPI)
and, per clip, mines the DEPLOYED model's own errors — flicker false-negatives
and transient/static false-positives — using temporal tracking (see flicker_miner.py).

This is a REVIEW-ACCELERATOR, not an auto-labeler: nothing is treated as ground
truth. It ships almost-correct pre-labels, rich in FN/FP, into a Label Studio
import file so a human can review/correct cheaply, then export YOLO and promote
with promote_to_trainset.py.

Run (each invocation does one pass, then exits — schedule via cron for repeats):
    python3 build_dataset.py --config config.json               # sweep the whole window (default past 24h)
    python3 build_dataset.py --config config.json --channel 2   # ... just channel 2
    python3 build_dataset.py --config config.json --once        # one most-recent clip per channel
    python3 build_dataset.py --config config.json --once --channel 2 --dry-run

Crash-safety / resume: outputs are checkpointed after EVERY clip (manifest flushed,
the channel's task file rewritten, the persistence map saved, then the clip_id
appended to <staging>/<nvr>/mined_clips.txt). Re-running the same window therefore
resumes rather than restarts: clips in the ledger are skipped, so they are neither
re-inferred nor duplicated into manifest.csv nor double-counted in the persistence
map. Use --redo to force re-mining. Clips that failed to download are deliberately
NOT ledgered, so a later run retries them (a window may have no recording yet).
"""
import argparse
import csv
import json
import logging
import os
import shlex
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

# --- Make the in-repo SUNAPI downloader + sibling modules importable ----------
# .../scripts/cctv/dataset_builder/build_dataset.py -> repo root is parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_HERE = Path(__file__).resolve().parent
_SUNAPI_DIR = _REPO_ROOT / "scripts" / "cctv" / "SUNAPI" / "SunapiClipPy"
for p in (str(_HERE), str(_SUNAPI_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import sunapi_clip  # noqa: E402
import flicker_miner  # noqa: E402
import labelstudio_export as lsx  # noqa: E402
import persistence as pers  # noqa: E402
from ultralytics import YOLO  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build_dataset")


BUCKETS = {
    "overnight": (0, 6),
    "morning": (6, 12),
    "afternoon": (12, 18),
    "evening": (18, 24),
}

DEFAULTS = {
    # --- NVR / SUNAPI ---
    "verify_ssl": False,
    "rtsp_transport": "tcp",
    "rtsp_port": None,
    "channels": "all",
    # --- Deployed ("model under test") — must match production settings ---
    "model": "docker_build/yolo11x_set01-0148.pt",
    "img_size": 640,
    "conf_thresh": 0.6,          # production report threshold (what the field uses)
    # --- Mining (recall + tracking) ---
    "track_conf": 0.25,          # low conf so flickering/weak persons are seen
    "track_vid_stride": 2,       # process every Nth frame (smaller = better flicker fidelity)
    "iou_track": 0.3,            # IoU to associate detections into a track
    "max_gap_frames": 15,        # max processed-step gap to bridge/interpolate a track (15 steps = 1.0 s at 30 fps, stride 2)
    # Gap-bridge plausibility gate (ID-switch guard). Both thresholds are high
    # percentiles of the band the tracker's own iou_track=0.3 association already
    # permits (feasible max: displacement 0.523, area ratio 3.333), so the gate is
    # non-vacuous by construction yet discards almost no candidates. See
    # flicker_miner._bridgeable and ALGORITHM.md 4 for the derivation.
    "bridge_max_disp_frac": 0.45,   # reject if TOTAL centroid displacement across the gap > this * mean box-diagonal
    "bridge_max_scale_ratio": 3.2,  # reject if the two bracketing boxes' areas differ by more than this factor
    "fp_max_track_len": 2,       # tracks this short whose det>=conf_thresh are suspected FPs (2 steps = 0.13 s)
    # --- Static human-like FP (mannequin/poster) detection ---
    # static_min_frames governs FLAGGING FOR REVIEW and is deliberately loose
    # (10 steps = 0.66 s), because a wrongly flagged box costs one relabel click.
    # persist_min_track_steps governs what may be WRITTEN INTO the cross-clip map
    # and is far stricter, since the map's output feeds future flagging decisions
    # and must not accumulate briefly-paused real people.
    "static_min_frames": 10,     # a track present >= this many steps may be flagged SFP
    "static_max_move_frac": 0.15,  # centroid spread < this * box-diagonal => stationary
    "static_motion_thresh": 0.08,  # mean (1 - ZNCC) below this => appearance-static (illum-invariant)
    # --- Cross-clip persistence (a location that fires across many clips = fixture) ---
    "cross_clip_persistence": True,
    "persist_grid_cols": 64,
    "persist_grid_rows": 36,
    "persist_min_track_steps": 60,  # only an appearance-static track >= this long (4.0 s) feeds the map
    "persist_min_clips": 5,       # need this many FIXTURE-CARRYING clips before trusting the map
    "persist_thresh": 0.35,       # max cell value (see persistence.py) to call a location a fixture
    # --- Per-clip caps: per-category guarantees of evenly spread coverage ---
    "max_intpfn_per_clip": 40,   # interpolated-gap misses
    "max_weakfn_per_clip": 40,   # sub-threshold (weak) misses
    "max_anchor_per_clip": 20,   # detections bordering interp gaps (FP candidates)
    "max_sfp_per_clip": 30,
    "max_fp_per_clip": 20,
    "max_easy_per_clip": 10,     # easy budget for a clip that produced >=1 candidate
    "max_easy_barren_clip": 1,   # easy budget for a clip that produced NO candidate
    "easy_every_n": 5,           # minimum spacing, in processed steps, between selected easy frames
    # --- Time window sampled per sweep ---
    "clip_duration_sec": 60,     # length of each sampled clip
    "clip_download_timeout_sec": 60,   # kill a stalled ffmpeg download (hung RTSP session);
                                       # healthy backup downloads finish in a few seconds
    "clip_end_margin_sec": 120,  # window ends this far back so footage is recorded
    "lookback_hours": 24,        # rolling window depth (used when window_start/end unset)
    "window_start": None,        # optional absolute ISO window; set BOTH to override lookback
    "window_end": None,
    "clip_interval_min": 30,     # sample one clip every this many minutes across the window
    "staging_dir": "data/cctv_train_data_mining/reviewing",
    "keep_clips": False,
    "annotate_clips": True,      # when keep_clips: also write <clip>_annotated.mp4 (mining boxes drawn)
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class Config:
    # Keys that are required and therefore absent from DEFAULTS.
    _EXTRA_KEYS = ("host", "username", "password")

    def __init__(self, d):
        # Warn on keys this build does not know. Silently swallowing them lets an
        # obsolete or misspelled key look effective while the default is actually
        # in force — e.g. a config still carrying `sweep_interval_min` from before
        # the internal loop was removed.
        unknown = sorted(set(d) - set(DEFAULTS) - set(self._EXTRA_KEYS))
        if unknown:
            logger.warning(f"config: ignoring unknown key(s): {', '.join(unknown)}")
        merged = dict(DEFAULTS)
        merged.update(d)
        for k, v in merged.items():
            setattr(self, k, v)
        for req in ("host", "username", "password"):
            if not getattr(self, req, None):
                raise SystemExit(f"config is missing required field: {req}")
        if not os.path.isabs(self.model):
            self.model = str(_REPO_ROOT / self.model)
        if not os.path.isabs(self.staging_dir):
            self.staging_dir = str(_REPO_ROOT / self.staging_dir)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return Config(json.load(f))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def bucket_for(hour):
    for name, (lo, hi) in BUCKETS.items():
        if lo <= hour < hi:
            return name
    return "overnight"


def nvr_slug(host):
    h = host.split("://", 1)[-1].split(":", 1)[0]
    return h.split(".", 1)[0] or "nvr"


def _time_window(cfg):
    """Return (win_start, win_end) datetimes to sample clips across.

    Absolute window if BOTH window_start and window_end are set; otherwise a
    rolling window of `lookback_hours` ending `clip_end_margin_sec` before now
    (so the most recent sampled clip is already recorded on the NVR)."""
    ws, we = getattr(cfg, "window_start", None), getattr(cfg, "window_end", None)
    if bool(ws) != bool(we):
        raise SystemExit("set BOTH window_start and window_end, or neither (use lookback_hours)")
    if ws and we:
        return datetime.fromisoformat(ws), datetime.fromisoformat(we)
    win_end = datetime.now() - timedelta(seconds=cfg.clip_end_margin_sec)
    return win_end - timedelta(hours=cfg.lookback_hours), win_end


def resolve_channels(cfg):
    if cfg.channels == "all":
        cams = sunapi_clip.get_registered_channels(
            sunapi_clip.normalize_base_url(cfg.host), cfg.username, cfg.password,
            cfg.verify_ssl, 10.0)
        return sorted(int(c["Channel"]) for c in cams if c.get("Channel") is not None)
    return [int(c) for c in cfg.channels]


MANIFEST_COLUMNS = ["image", "clip_id", "channel", "bucket", "n_person",
                    "n_intpfn", "n_weakfn", "n_anchor_start", "n_anchor_end",
                    "n_sfp", "n_fp", "n_animal", "animals", "track_ids"]


def open_manifest(manifest_path):
    """Open manifest.csv for append, guarding against SCHEMA DRIFT.

    The header is written only when the file is new, so appending rows produced by
    a different build silently misaligns every column from the first schema change
    onward — the reader sees plausible values under the wrong names with no error
    anywhere. This has already happened once in practice: an existing manifest
    carries the pre-anchor-split 12-column schema (with a single `reason` column
    and a combined `n_anchor`), while this build writes 14 columns with `clip_id`
    inserted at position 2.

    So compare the existing header against MANIFEST_COLUMNS and, on mismatch, move
    the old file aside (never delete it — it is the only index of already-staged
    frames) and start a fresh one. Returns (file_handle, csv_writer)."""
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    if os.path.exists(manifest_path) and os.path.getsize(manifest_path) > 0:
        with open(manifest_path, "r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header != MANIFEST_COLUMNS:
            n = 1
            while os.path.exists(f"{manifest_path}.v{n}.bak"):
                n += 1
            backup = f"{manifest_path}.v{n}.bak"
            os.replace(manifest_path, backup)
            logger.warning(f"[sweep] manifest schema changed ({len(header or [])} -> "
                           f"{len(MANIFEST_COLUMNS)} columns); moved existing manifest to "
                           f"{os.path.basename(backup)} and starting a new one")
    new_manifest = not os.path.exists(manifest_path)
    fh = open(manifest_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if new_manifest:
        writer.writerow(MANIFEST_COLUMNS)
    return fh, writer


def write_yolo_labels(txt_path, boxes, cls_ids, width, height):
    """Write a YOLO label file: 'cls xc yc w h' normalized to [0,1] (empty if none)."""
    dw, dh = 1.0 / width, 1.0 / height
    with open(txt_path, "w") as f:
        for (x1, y1, x2, y2), cid in zip(boxes, cls_ids):
            xc = ((x1 + x2) / 2.0) * dw
            yc = ((y1 + y2) / 2.0) * dh
            f.write(f"{cid} {xc} {yc} {(x2 - x1) * dw} {(y2 - y1) * dh}\n")


# ---------------------------------------------------------------------------
# Write one mined frame to staging (jpg + YOLO txt), return its LS task
# ---------------------------------------------------------------------------
def _save_record(rec, cfg, nvr, channel, bucket, stamp, capture_time):
    import cv2
    rec["channel"], rec["bucket"] = channel, bucket
    rec["clip_id"] = f"{nvr}_ch{channel:02d}_{stamp}"
    # capture_time is the CLIP's start timestamp (one value per clip, shared by
    # every frame in it), NOT this frame's own time within the clip — that is
    # frame_idx (rec["raw_idx"], the raw video-frame index pass 1 assigned it).
    # Together they give the Data Manager a sortable field ALGORITHM.md 9 notes
    # is otherwise missing: import order is the only ordering LS has today.
    rec["capture_time"] = capture_time
    rec["track_ids"] = sorted({tid for (_b, _s, tid) in rec["persons"]}
                               | {tid for (_b, _k, tid) in rec["suspect"]})
    ch_name = f"ch{channel:02d}"
    # Flat per-channel/bucket staging — no per-category subfolder. Which
    # category(s) a frame belongs to is independent, filterable data
    # (n_intpfn, n_weakfn, ... in manifest.csv / the LS task), not a folder.
    out_dir = os.path.join(cfg.staging_dir, nvr, ch_name, bucket)
    os.makedirs(out_dir, exist_ok=True)

    ani_tag = ("_ani-" + "-".join(rec["animals"])) if rec["animals"] else ""
    name = f"{nvr}_{ch_name}_{bucket}_{stamp}_{rec['raw_idx']:06d}{ani_tag}"
    jpg_path = os.path.join(out_dir, name + ".jpg")
    txt_path = os.path.join(out_dir, name + ".txt")

    # YOLO pre-labels = corrected oracle: persons (class 0) + animals. Suspected
    # FPs are NOT written here (they are phantoms) — they ride only in the LS task.
    boxes = [b for (b, _src, _tid) in rec["persons"]] + [b for (b, _c) in rec["animal_boxes"]]
    cls_ids = [flicker_miner.PERSON_ID] * len(rec["persons"]) + \
              [cls for (_b, cls) in rec["animal_boxes"]]

    cv2.imwrite(jpg_path, rec["image"])
    write_yolo_labels(txt_path, boxes, cls_ids, rec["width"], rec["height"])

    # Manifest path stays relative to staging_dir; the Label Studio ?d= path is
    # relative to the local-files document root (the PARENT of staging), so ONE
    # Local Storage registered at <staging_dir> covers every NVR.
    rel_manifest = os.path.relpath(jpg_path, cfg.staging_dir)
    rel_docroot = os.path.relpath(jpg_path, _ls_document_root(cfg.staging_dir))
    return lsx.build_task(rec, rel_docroot), rel_manifest


def _ls_document_root(staging_dir):
    """Label Studio local-files document root = the PARENT of staging_dir. A
    single Local Storage registered at <staging_dir> (a subdir of this root)
    then serves every NVR, and task ?d= paths are relative to this parent
    (e.g. staging/<nvr>/...)."""
    return os.path.dirname(os.path.normpath(staging_dir))


def load_mined_clips(path):
    """Return the set of clip_ids already mined by an earlier run.

    A sweep is long (2:57:24 measured for 24 h x 13 channels), so a kill, an OOM or
    a machine reboot part-way through is a normal event rather than an exotic one.
    Without a record of what has been done, re-running the same absolute window
    restarts from clip 0: it re-downloads and re-infers everything already
    completed, re-appends duplicate rows for those frames to manifest.csv, and
    re-folds their tracks into the persistence map (double-counting the evidence).
    This ledger turns that re-run into a resume."""
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def record_mined_clip(path, clip_id):
    """Append one clip_id to the ledger and flush it to disk immediately.

    Appended per clip rather than per sweep, and recorded whether or not the clip
    yielded any frame: a clip that produced nothing is still work that need not be
    repeated, and it leaves no manifest row to infer that from."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(clip_id + "\n")
        f.flush()
        os.fsync(f.fileno())


def _write_ls_launcher(ls_dir, document_root):
    """Write an executable launcher next to the tasks file that starts Label
    Studio with local-file serving enabled and the document root set so the
    tasks' ?d= image paths resolve."""
    path = os.path.join(ls_dir, "run_labelstudio.sh")
    content = (
        "#!/usr/bin/env bash\n"
        "# Auto-generated by build_dataset.py. Launches Label Studio with local-file\n"
        "# serving so the tasks_*.json images (/data/local-files/?d=...) resolve.\n"
        "# Register <document_root>/staging as ONE Local Storage to cover all NVRs.\n"
        "export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true\n"
        f"export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={shlex.quote(document_root)}\n"
        'exec label-studio "$@"\n'
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(path, 0o755)
    return path


# ---------------------------------------------------------------------------
# One sweep over the channels
# ---------------------------------------------------------------------------
def sweep(cfg, model, only_channel=None, forced_bucket=None, dry_run=False,
          single_clip=False, redo=False):
    base_url = sunapi_clip.normalize_base_url(cfg.host)
    nvr = nvr_slug(cfg.host)
    channels = [only_channel] if only_channel is not None else resolve_channels(cfg)
    logger.info(f"[sweep] nvr={nvr} channels={channels}")

    clips_dir = os.path.join(cfg.staging_dir, ".clips")
    manifest_path = os.path.join(cfg.staging_dir, nvr, "manifest.csv")
    ledger_path = os.path.join(cfg.staging_dir, nvr, "mined_clips.txt")
    ls_dir = os.path.join(cfg.staging_dir, nvr, "labelstudio")
    # Clips are downloaded and mined even in a dry run (it only skips writing the
    # dataset), so .clips/ must always exist.
    os.makedirs(clips_dir, exist_ok=True)
    manifest_file = manifest_writer = None
    if not dry_run:
        os.makedirs(ls_dir, exist_ok=True)
        manifest_file, manifest_writer = open_manifest(manifest_path)
    # Two distinct uses of the ledger, deliberately separated:
    #   ledgered      — every clip_id already on record, used ONLY to avoid appending
    #                   a duplicate line. Loaded even under --redo, so repeatedly
    #                   forcing a re-run cannot grow the file without bound.
    #   already_mined — the clip_ids this run will SKIP. Empty under --redo, which is
    #                   what makes --redo re-mine everything.
    ledgered = set() if dry_run else load_mined_clips(ledger_path)
    already_mined = set() if redo else set(ledgered)
    if already_mined:
        logger.info(f"[sweep] resume ledger: {len(already_mined)} clip(s) already mined "
                    f"will be skipped (--redo to re-mine them)")
    elif redo and ledgered:
        logger.info(f"[sweep] --redo: ignoring {len(ledgered)} ledgered clip(s); "
                    f"they will be re-mined")

    # Sample start times across the window (one clip_duration_sec clip per step),
    # computed once so the window is identical for every channel this sweep.
    # --once (single_clip) collapses the window to a single most-recent clip.
    if single_clip:
        win_end = datetime.now() - timedelta(seconds=cfg.clip_end_margin_sec)
        win_start = win_end - timedelta(seconds=cfg.clip_duration_sec)
    else:
        win_start, win_end = _time_window(cfg)
    step = timedelta(minutes=cfg.clip_interval_min)
    dur = timedelta(seconds=cfg.clip_duration_sec)
    sample_starts, t = [], win_start
    while t + dur <= win_end:
        sample_starts.append(t)
        t += step
    if not sample_starts:
        logger.warning(f"[sweep] empty window {win_start}..{win_end}; nothing to do")
        if manifest_file is not None:
            manifest_file.close()
        return Counter()
    logger.info(f"[sweep] window {win_start:%Y-%m-%dT%H:%M}..{win_end:%Y-%m-%dT%H:%M}: "
                f"{cfg.clip_duration_sec}s clip every {cfg.clip_interval_min} min "
                f"-> {len(sample_starts)} clips/channel x {len(channels)} channels")

    totals = Counter()
    n_tasks_written = 0
    sweep_stamp = win_end.strftime("%Y%m%dT%H%M%S")
    try:
        for ch in channels:
            persist = persist_path = None
            if cfg.cross_clip_persistence:
                persist_path = os.path.join(cfg.staging_dir, nvr, "persistence", f"ch{ch:02d}.json")
                persist = pers.load(persist_path, cfg.persist_grid_cols, cfg.persist_grid_rows)

            # One task file PER CHANNEL, rewritten after every clip that yields
            # tasks. Previously every task of the whole sweep was held in memory and
            # written once at the very end, so a kill at hour 2 of a 3 h sweep left
            # the staged JPEGs and manifest rows on disk with no importable task file
            # at all — the review queue for all completed work was lost.
            #
            # On a resume the file must be EXTENDED, not rewritten from scratch:
            # sweep_stamp derives from the window, so a re-run of the same window
            # targets the same path, and starting from an empty list would truncate
            # the tasks the interrupted run had already written. Resumed clips are
            # skipped by the ledger, so their tasks are read back exactly once and
            # cannot be duplicated. With --redo the clips ARE re-mined, so the file
            # is deliberately started fresh instead.
            ch_tasks_path = os.path.join(ls_dir, f"tasks_{sweep_stamp}_ch{ch:02d}.json")
            ch_tasks = []
            if not dry_run and not redo and os.path.exists(ch_tasks_path):
                try:
                    with open(ch_tasks_path, "r", encoding="utf-8") as f:
                        ch_tasks = json.load(f)
                    logger.info(f"  ch{ch:02d}: extending existing "
                                f"{os.path.basename(ch_tasks_path)} "
                                f"({len(ch_tasks)} task(s) already in it)")
                except (ValueError, OSError) as e:
                    logger.warning(f"  ch{ch:02d}: could not read existing "
                                   f"{os.path.basename(ch_tasks_path)} ({e}); "
                                   f"starting a new task list")
                    ch_tasks = []
            n_preexisting = len(ch_tasks)

            for start_dt in sample_starts:
                end_dt = start_dt + dur
                bucket = forced_bucket or bucket_for(start_dt.hour)
                stamp = start_dt.strftime("%Y%m%dT%H%M%S")
                start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
                end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%S")

                # clip_id must match _save_record's, since it is the ledger key.
                clip_id = f"{nvr}_ch{ch:02d}_{stamp}"
                if clip_id in already_mined:
                    logger.info(f"  ch{ch:02d} [{bucket}] {start_iso}: already mined, skipping")
                    continue

                clip_path = os.path.join(clips_dir, f"{nvr}_ch{ch:02d}_{stamp}.mp4")
                logger.info(f"  ch{ch:02d} [{bucket}] {start_iso}..{end_iso}")
                try:
                    sunapi_clip.download_channel_clip(
                        base_url, cfg.username, cfg.password, ch, start_iso, end_iso,
                        clip_path, verify=cfg.verify_ssl, timeout=10.0,
                        rtsp_transport=cfg.rtsp_transport, rtsp_port=cfg.rtsp_port, verbose=False,
                        download_timeout=cfg.clip_download_timeout_sec)
                except (SystemExit, Exception) as e:
                    logger.warning(f"  ch{ch:02d} {stamp}: download failed: {e}")
                    continue

                if not os.path.exists(clip_path) or os.path.getsize(clip_path) == 0:
                    logger.warning(f"  ch{ch:02d} {stamp}: empty/missing clip, skipping")
                    continue

                records, all_frames = flicker_miner.mine_clip(
                    clip_path, model, cfg, persist=persist, update_persist=not dry_run)

                for rec in records:
                    for cat in flicker_miner.frame_categories(rec):
                        totals[cat] += 1
                    if dry_run:
                        continue
                    task, rel = _save_record(rec, cfg, nvr, ch, bucket, stamp, start_iso)
                    ch_tasks.append(task)
                    manifest_writer.writerow([
                        rel, rec["clip_id"], ch, bucket, rec["n_person"],
                        rec["n_intpfn"], rec["n_weakfn"],
                        rec["n_anchor_start"], rec["n_anchor_end"],
                        rec["n_sfp"], rec["n_fp"], len(rec["animal_boxes"]),
                        ",".join(rec["animals"]),
                        ";".join(str(t) for t in rec["track_ids"]),
                    ])

                if not dry_run:
                    # Durable, per-clip checkpoint. Order matters: the manifest rows
                    # and the task file are flushed BEFORE the ledger records the
                    # clip as done, so a crash between the two re-mines one clip
                    # (harmless, idempotent) rather than marking a clip complete
                    # whose outputs were never written.
                    manifest_file.flush()
                    if ch_tasks:
                        lsx.write_tasks(ch_tasks, ch_tasks_path)
                    if persist is not None:
                        pers.save(persist_path, persist)
                    if clip_id not in ledgered:
                        record_mined_clip(ledger_path, clip_id)
                        ledgered.add(clip_id)

                if cfg.keep_clips:
                    # Verification aid: an annotated copy next to the kept original.
                    if cfg.annotate_clips and not dry_run:
                        ann_path = os.path.splitext(clip_path)[0] + "_annotated.mp4"
                        try:
                            if flicker_miner.render_annotated_clip(
                                    clip_path, all_frames, ann_path, cfg.track_vid_stride):
                                logger.info(f"  ch{ch:02d}: wrote {os.path.basename(ann_path)}")
                        except Exception as e:
                            logger.warning(f"  ch{ch:02d}: annotated-clip render failed: {e}")
                else:
                    try:
                        os.remove(clip_path)
                    except OSError:
                        pass

            # The map is already saved per clip above; this only reports it.
            if persist is not None and not dry_run:
                # Calibration aid: persist_thresh is only meaningful relative to
                # the map's observed top cell value. If top_fraction stays far
                # below persist_thresh over many sweeps, either the site has no
                # fixture or the threshold needs lowering — the log is the record
                # needed to tell those apart.
                logger.info(f"  ch{ch:02d}: persistence map fixture_clips="
                            f"{persist.fixture_clips} cells={len(persist.hits)} "
                            f"top_cell={persist.top_fraction():.3f} "
                            f"(persist_thresh={cfg.persist_thresh}, "
                            f"min_clips={cfg.persist_min_clips})")
            if ch_tasks and not dry_run:
                n_new = len(ch_tasks) - n_preexisting
                n_tasks_written += n_new
                logger.info(f"  ch{ch:02d}: +{n_new} new Label Studio task(s) "
                            f"({len(ch_tasks)} total) -> "
                            f"{os.path.basename(ch_tasks_path)}")
    finally:
        if manifest_file is not None:
            manifest_file.close()

    if n_tasks_written and not dry_run:
        launcher = _write_ls_launcher(ls_dir, _ls_document_root(cfg.staging_dir))
        lsx.write_label_config(os.path.join(ls_dir, "label_config.xml"))
        logger.info(f"  wrote {n_tasks_written} Label Studio tasks across "
                    f"{len(channels)} per-channel file(s) in {ls_dir}")
        logger.info(f"  review: run {launcher} then import "
                    f"tasks_{sweep_stamp}_ch*.json (one import per channel)")

    logger.info(f"[sweep done] Intp_FN={totals['Intp_FN']} Weak_FN={totals['Weak_FN']} "
                f"Anchor_start={totals['Anchor_start']} Anchor_end={totals['Anchor_end']} "
                f"SFP={totals['SFP']} FP={totals['FP']} easy={totals['easy']}"
                + (" (dry-run, nothing written)" if dry_run else ""))
    return totals


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Path to config JSON")
    ap.add_argument("--once", action="store_true",
                    help="Grab a single most-recent clip per channel, then exit "
                         "(default: sample the whole configured window)")
    ap.add_argument("--channel", type=int, help="Restrict to this one channel")
    ap.add_argument("--bucket", choices=list(BUCKETS), help="Force the time-bucket tag")
    ap.add_argument("--dry-run", action="store_true",
                    help="Download + mine + log counts, but write nothing")
    ap.add_argument("--redo", action="store_true",
                    help="Ignore the mined_clips.txt ledger and re-mine clips already "
                         "done (default: skip them, so re-running a window resumes "
                         "instead of restarting and duplicating rows)")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger.info(f"Loading deployed model: {cfg.model}  (img_size={cfg.img_size}, "
                f"prod_conf={cfg.conf_thresh}, track_conf={cfg.track_conf})")
    model = YOLO(cfg.model)

    # One pass, then exit — no perpetual loop. Default covers the whole configured
    # window (past lookback_hours); --once grabs a single most-recent clip/channel.
    # For recurring collection, schedule this via cron / systemd timer.
    t0 = time.monotonic()
    sweep(cfg, model, only_channel=args.channel, forced_bucket=args.bucket,
          dry_run=args.dry_run, single_clip=args.once, redo=args.redo)
    elapsed = time.monotonic() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    logger.info(f"Total sweep time: {h:d}:{m:02d}:{s:02d} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
