- [CCTV hard-example miner (SUNAPI → deployed model → Label Studio)](#cctv-hard-example-miner-sunapi--deployed-model--label-studio)
  - [The idea](#the-idea)
    - [Why SFP is not just "MOG2 again"](#why-sfp-is-not-just-mog2-again)
    - [Cross-clip persistence (second static-FP signal)](#cross-clip-persistence-second-static-fp-signal)
    - [Known limitations](#known-limitations)
  - [Requirements](#requirements)
  - [Layout](#layout)
  - [Usage](#usage)
    - [Crash safety and resume](#crash-safety-and-resume)
    - [Time window a sweep covers](#time-window-a-sweep-covers)
    - [Reviewing output](#reviewing-output)
  - [Review in Label Studio → export YOLO](#review-in-label-studio--export-yolo)
    - [0. Install (one-time)](#0-install-one-time)
    - [1. Launch with local-file serving](#1-launch-with-local-file-serving)
    - [2. Create the project (one-time)](#2-create-the-project-one-time)
    - [3. Register the staged images as Local Storage (required, once)](#3-register-the-staged-images-as-local-storage-required-once)
    - [4. Import a sweep's tasks](#4-import-a-sweeps-tasks)
    - [Automating project setup: create\_ls\_project.py](#automating-project-setup-create_ls_projectpy)
    - [5. Review](#5-review)
    - [6. Export](#6-export)
    - [Troubleshooting: "There was an issue loading URL from $image value"](#troubleshooting-there-was-an-issue-loading-url-from-image-value)
  - [Config keys](#config-keys)
  - [Promote reviewed data](#promote-reviewed-data)
  - [Caveats](#caveats)


# CCTV hard-example miner (SUNAPI → deployed model → Label Studio)

Grows the training set in `data/cctv_train_data/` by mining the **deployed
model's own errors** from live-site footage — the frames worth retraining on.

It is a **review-accelerator, not an auto-labeler.** No model output is treated
as ground truth. The tool ships *almost-correct* pre-labels that are rich in
false-negatives (flicker) and false-positives into a Label Studio project, so a
human reviews/corrects cheaply, exports YOLO, and promotes the result.

## The idea

The production model (`yolo11x_set01-0148.pt` @ 640, conf 0.6) **flickers**: when
a person is present for a stretch, some frames drop below the report threshold
and are missed. Those missed frames are the most valuable training data — and a
naïve "label every frame with a model" collector would file them as empty
background, poisoning the set.

**Scope — what "production view" means here.** The mined comparison is against the
**raw detector** at its deployed threshold (conf ≥ `conf_thresh`), *not* against
what the deployed service reports. The service applies four more stages after that
threshold — `filter_small_objects`, `remove_outside_ROI`, `check_bb_on_background`
(MOG2) and `hysteresis` — and none is modelled here. That is deliberate: the review
loop produces a training set, training changes only the network's weights, and no
weight change can fix an ROI polygon or a MOG2 threshold. The cost is a defined
blind spot: a correct detection that MOG2 or the ROI mask suppresses is a real
end-to-end miss, but this tool files that frame as `easy`. See `ALGORITHM.md` §1a.

**Cost model — recall of candidates beats precision of pre-labels.** A wrongly
flagged box costs the reviewer one click on a box already in front of them. A
*missed* candidate costs a training example permanently, because nothing revisits an
unflagged frame. Thresholds here are tuned accordingly, and `ALGORITHM.md` §1a says
so at each point of choice. The one exception is the cross-clip persistence map,
whose output feeds later decisions and so is gated much more strictly.

Instead, per clip, the miner:

1. Runs the **deployed** model on every `track_vid_stride`-th frame at a low
   recall conf (`track_conf`), keeping weak/flickering detections.
2. Links person detections into **tracks** (greedy IoU + short-gap bridging) and
   interpolates a box across gaps where the detection dropped out.
3. Compares each frame's **production view** (conf ≥ `conf_thresh`) against the
   tracks. A tracked person the production view missed is a flicker miss. Six
   independent categories result, and a frame can belong to any number of them at
   once:
   - **Intp_FN** — an **interpolated** gap frame: the model produced no detection
     here, but the track exists before & after, so the box is interpolated. *Primary target.*
   - **Weak_FN** — the model *did* fire here but **below** `conf_thresh` (a weak,
     sub-threshold detection).
   - **Anchor_start** / **Anchor_end** — **any** detection (**strong or weak**)
     that **borders** an interpolated gap: `Anchor_start` **opens** the gap (the
     detection just before it), `Anchor_end` **closes** it (just after). These are
     the "seemingly true" frames used to fill an `Intp_FN`, and are themselves
     **FP candidates** (if the anchor is a false positive, the interpolated
     FN is spurious), so they get their own reasons to double-check during review.
     Confidence doesn't gate this: a **weak** anchor is tagged `Anchor_start`/
     `Anchor_end` *in addition to* `Weak_FN`, not instead of it — both are true
     facts about the same box (see `ALGORITHM.md` §4b).
   - **FP** — a production detection whose track is **gap-free** and transient
     (≤ `fp_max_track_len` frames): an isolated blip with no temporal support at
     all, before or after. A track that bridges even one gap is *never* tagged
     `FP`, regardless of how few total detections it has — that shape is
     indistinguishable from a real person briefly, completely occluded, so it is
     routed to the ordinary flicker path (`Anchor_start`/`Intp_FN`/`Anchor_end`)
     instead (see `ALGORITHM.md` §4b).
   - **SFP (static FP)** — a **persistent, stationary, high-confidence** person
     track whose box appearance barely changes over time under an
     **illumination-invariant** measure (zero-mean normalized cross-correlation).
     This targets human-like apparatus (mannequins, posters, standees) that MOG2
     background subtraction leaks through when lighting drifts. Flagged as
     `SUSPECT_STATIC_FP` for review.
4. Selects Intp_FN/Weak_FN/Anchor_start/Anchor_end/SFP/FP frames (plus a few
   `easy` frames for balance) and writes almost-correct pre-labels + a Label Studio task.
   Each category's per-clip cap is filled by an **even temporal subsample** across
   the whole clip, not by the first frames that match, and the `easy` budget is
   **conditional**: a clip that found at least one candidate contributes up to
   `max_easy_per_clip`, a clip that found none contributes only
   `max_easy_barren_clip`. Without that split, `easy` frames were 88 % of the review
   queue (see `ALGORITHM.md` §5).

Pre-labels come **only** from the deployed (domain-fine-tuned) model + tracking —
never a bigger foreign model, which would add spurious boxes and more review pain.

> **Full details:** the project's scope and cost model (§1a), the exact
> classification logic, the anchor/FN mechanism (with worked examples), the
> calibration of the gap-bridge gate (§4a), the selection algorithm (§5), the SFP
> signals and the persistence map's semantics (§6) are documented in
> **[ALGORITHM.md](ALGORITHM.md)**. The sections below are a summary. There is no
> per-frame "reason" or category precedence — a frame belongs to any number of the
> six categories independently.

### Why SFP is not just "MOG2 again"
`check_bb_on_background()` fails on static human-like objects precisely because a
lighting drift makes MOG2 see foreground over the whole frame, so the static
object's box looks like it is moving and passes as a detection. The SFP detector
avoids that trap: it compares the *same box's crop across frames* with ZNCC,
which cancels brightness/contrast changes — a poster stays ≈1.0 (static) while a
real person's limbs/posture drop it. Stationarity (centroid barely moves) over a
long track is required too. A real person who merely stood still is only *flagged*
for review, not auto-discarded — the human confirms mannequin (delete → hard
negative) vs. person (keep).

### Cross-clip persistence (second static-FP signal)
Static FPs are also checked by a **second signal**: a per-camera map
(`persistence.py`) accumulates, across sweeps/days, how often each screen cell is
covered by a **fixture-like** track. A location reaching `persist_thresh` over
`persist_min_clips` of history is almost certainly a fixture — no real person stands
in the same pixels across many days and lightings. A stationary track is flagged SFP
if **either** the within-clip ZNCC signal **or** the cross-clip map says so. State
lives in `<reviewing>/<nvr>/persistence/ch<NN>.json` and survives restarts;
decisions use the *prior* state, then the current clip is folded in. Disable with
`"cross_clip_persistence": false`.

Two properties of this map are easy to get wrong (see `ALGORITHM.md` §6a/§6b for
the full treatment):

- **What may be written into it is far stricter than what gets flagged.** Flagging
  uses `static_min_frames` (10 steps = 0.66 s), so a briefly-paused person can be
  flagged — cheap, one relabel click. The map instead requires an appearance-static
  track of at least `persist_min_track_steps` (60 steps = 4.0 s). If the map
  accumulated every briefly-stationary track, then any spot where people habitually
  pause would eventually cross threshold and start auto-flagging real people, with
  the flags reinforcing the belief that produced them — a feedback loop the reviewer
  cannot break, because reviewers correct frames, never the map.
- **The stored value is conditional.** It is `hits / fixture_clips`, where
  `fixture_clips` counts only clips that contributed a fixture-like track — clips
  with none carry no evidence and are not counted, which is what lets a real
  fixture's cell value actually converge toward 1.0. Each sweep logs
  `fixture_clips`, cell count and observed `top_cell` per channel, since
  `persist_thresh` is only meaningful relative to that maximum.

Changing `persist_grid_cols`/`persist_grid_rows`, or upgrading past a state-version
change, invalidates an existing map: it is moved aside to `ch<NN>.json.v<N>.bak` and
rebuilt, since neither the stored ratio nor the cell indexing survives such a change.

### Known limitations
This catches *flicker* FNs, *transient* FPs, and *static* FPs (within-clip and
cross-clip). Out of scope by construction:

- A person missed for an **entire** clip — no neighbouring detection to interpolate
  from, so no track and no candidate.
- Any error introduced **downstream of the detector** (small-object filter, ROI mask,
  MOG2, hysteresis). Such a frame carries a `strong` detection and is filed `easy`.
  See the scope note above and `ALGORITHM.md` §1a.
- **Animal** flicker or misses — animals bypass tracking entirely; only detections
  clearing `conf_thresh` in that exact frame are pre-labelled.

## Requirements

Runs in the project's **`PyTorch`** conda env (Ultralytics 8.4.83, OpenCV,
`requests`, `ffmpeg`). Activate first: `conda activate PyTorch`. GPU recommended —
mining runs the model densely (every `track_vid_stride`-th frame of every sampled
clip), so it is compute-heavy — see the volume note under "Time window".

## Layout

```
scripts/cctv/dataset_builder/
  mine_dataset.py         orchestration: sample+download clips across a window, mine, write outputs
  flicker_miner.py         per-clip mining: detect + track + FN/SFP/FP classify + annotated-clip render
  persistence.py           per-camera cross-clip fixture map (static-FP signal)
  labelstudio_export.py    build the Label Studio import JSON (pre-labels preloaded)
  ls_api.py                shared Label Studio API auth client (used by the two scripts below)
  create_ls_project.py     create+populate a Label Studio project via its API (no manual XML paste)
  promote_to_trainset.py   merge a reviewed/exported dir (or an LS project via --ls-project) into data/cctv_train_data/setNNNN
  config.example.json      copy to config.json and fill in
  ALGORITHM.md             detailed classification algorithm & rationale (reasons, anchors, SFP)
```

Only the SUNAPI downloader (`../SUNAPI/SunapiClipPy/sunapi_clip.py`) is reused
from the rest of the repo; everything else here is self-contained.

## Usage

```bash
cp config.example.json config.json     # edit host/username/password
# config.json holds a PLAINTEXT password — do not commit it.

# 1) NVR reachable? list channels
#    (read the password from the environment; do not paste credentials into docs,
#     shell history, or a committed file)
python3 ../SUNAPI/SunapiClipPy/sunapi_clip.py \
    --host "$NVR_HOST" --username admin --password "$NVR_PASSWORD" --list-channels

# 2) quick test: one most-recent clip on one channel, write nothing
python3 mine_dataset.py --config config.json --once --channel 0 --dry-run

# 3) one most-recent clip on one channel (real)
python3 mine_dataset.py --config config.json --once --channel 0

# 4) full window (default past 24h) for one channel, then exit
python3 mine_dataset.py --config config.json --channel 0

# 5) full window for ALL channels, then exit (large — see note below)
python3 mine_dataset.py --config config.json

# 6) pin an absolute historical window without editing config.json
python3 mine_dataset.py --config config.json \
    --window-start 2026-08-07T15:54:00 --window-end 2026-08-08T15:54:00
```

Every invocation does **one pass then exits** — there is no built-in loop. For
recurring collection, schedule it (cron / systemd timer), e.g. a daily 24 h sweep.

### Crash safety and resume

A full sweep is long — 2:57:24 measured for 24 h × 13 channels — so being killed
part-way through is routine, not exotic. Outputs are therefore **checkpointed after
every clip**, in this order:

1. `manifest.csv` is flushed;
2. the channel's task file is rewritten (`tasks_<sweep_stamp>_ch<NN>.json`);
3. the channel's persistence map is saved;
4. the `clip_id` is appended to `<staging>/<nvr>/mined_clips.txt` and `fsync`ed.

The ledger is written **last** on purpose: a crash between steps 1–3 and step 4
re-mines one clip, which is harmless because that is idempotent, whereas the reverse
order could mark a clip complete whose outputs were never written.

Re-running the same window is then a **resume**: ledgered clips are skipped, so they
are not re-downloaded, not re-inferred, not re-appended to `manifest.csv`, and — the
subtle one — not folded into the persistence map a second time, which would
double-count that clip's fixture evidence. An existing task file is **extended**,
not truncated, since `sweep_stamp` derives from the window and a re-run targets the
same path. Verified: a sweep killed during clip 3 of 3 leaves 240 importable tasks;
resuming yields exactly 360 with no duplicates, and a third run is a no-op.

- **`--redo`** ignores the ledger and re-mines everything, starting the task file
  fresh rather than appending (which would duplicate tasks). The ledger is still
  read, so repeated `--redo` runs cannot grow it without bound.
- Clips that **failed to download** are deliberately *not* ledgered, so a later run
  retries them — a window may simply have had no recording yet. Expect these
  (144 of 768 attempts in one measured sweep); they are logged and skipped.
- Tasks arrive as **one file per channel** (`tasks_<stamp>_ch*.json`), so import
  once per channel rather than once per sweep.

### Time window a sweep covers

A default run **samples a clip every `clip_interval_min` across a time window**,
per channel (`--once` instead grabs a single most-recent clip — see run modes
below). By default the window is the **past 24 h** (`lookback_hours: 24`), ending
`clip_end_margin_sec` before now,
so a 24 h window at the 30 min default yields ~48 clips/channel and naturally
spans all four time buckets (each clip is tagged by its own hour). Set **both**
`window_start` and `window_end` (ISO) in the config to pin a fixed historical
range instead, or pass **`--window-start`/`--window-end`** on the command line
to do the same without editing the config file. Each flag independently
overrides the matching config key when given; the same both-or-neither rule
still applies to the *result*, so e.g. passing just `--window-end` against a
config that already sets `window_start` is fine, but passing just one flag
against a config where neither key is set raises an error.

Run modes (both do one pass, then exit):
- **Default** (no `--once`): sample the whole window (past `lookback_hours`, or the absolute `window_start`/`window_end`), then exit. This is the "cover the past 24 h and finish" mode.
- **`--once`**: grab a **single most-recent** clip per channel (window collapses to one `clip_duration_sec` clip ending `clip_end_margin_sec` ago), then exit — for a quick test or a one-off grab.

`--channel N` restricts either mode to one channel. For recurring collection,
schedule the command externally (cron / systemd timer) — there is no internal loop.

> Volume scales with `lookback_hours × channels / clip_interval_min`, and each
> clip is mined densely — a 24 h × 64-channel run is large. Start with
> `--once --channel N` (one clip) or `--channel N` (one channel's window) to gauge
> time before running all channels.

### Reviewing output

```
<staging_dir>/<nvr>/ch<NN>/<bucket>/
    <nvr>_ch<NN>_<bucket>_<stamp>_<raw_idx>[_ani-..].png + .txt   (one pair per selected frame — flat, no per-category subfolder)
<staging_dir>/<nvr>/labelstudio/tasks_<stamp>_ch<NN>.json  Label Studio import (one per channel, rewritten after every clip)
<staging_dir>/<nvr>/mined_clips.txt   resume ledger: one clip_id per line, appended after each clip is fully written
<staging_dir>/<nvr>/persistence/ch<NN>.json         cross-clip fixture map (persists across sweeps)
<staging_dir>/<nvr>/manifest.csv   image, clip_id, channel, bucket, n_person, n_intpfn, n_weakfn, n_anchor_start, n_anchor_end, n_sfp, n_fp, n_animal, animals, track_ids
<staging_dir>/.clips/              downloaded mp4s (removed unless keep_clips; with keep_clips also <clip>_annotated.mp4)
```

Images are PNG, not JPEG: each staged file is written straight from `mine_clip`'s
in-memory pass-1 frame cache, so it is byte-identical to the array the detector
actually scored — a lossy re-encode would let the saved copy (the one Label
Studio shows and, eventually, retraining trains on) silently drift from the
confidence recorded in the task JSON. See `ALGORITHM.md` §3.

A frame is **not** filed into a single category's folder — a frame commonly
qualifies for more than one at once (e.g. an interpolated miss *and* an
unrelated anchor), so which categories it touches is independent, filterable
data instead: the `n_intpfn` / `n_weakfn` / `n_anchor_start` / `n_anchor_end` /
`n_sfp` / `n_fp` counts in `manifest.csv` and the Label Studio task (see
`ALGORITHM.md` §5). `clip_id` and `track_ids` let you filter to one clip, or
trace a person across the frames its track appears in.

The `.txt` is the corrected pre-label (person=0, animals 1–6). Suspected-FP boxes
are **not** in the `.txt`; they ride only in the Label Studio task (labeled
`SUSPECT_FP` for transient blips, `SUSPECT_STATIC_FP` for mannequin/poster
candidates) so the reviewer can see and delete them.

## Review in Label Studio → export YOLO

### 0. Install (one-time)

```bash
conda activate PyTorch          # or any env you prefer for the reviewer
pip install label-studio
```

### 1. Launch with local-file serving

Label Studio only serves the staged images when local-file serving is enabled
and the document root points at the **parent of `staging_dir`** (so one Local
Storage at `staging_dir` covers every NVR — see step 3). Each sweep writes a
ready-to-run launcher next to the tasks file (document root baked in) — just run it:

```bash
<staging_dir>/<nvr>/labelstudio/run_labelstudio.sh          # e.g. .../reviewing/cheilacc-ansung/labelstudio/
# forwards args, so run_labelstudio.sh -p 8081 works
```

Equivalent manual form (must be set in the SAME shell, BEFORE starting; restart
to pick up changes):

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<absolute parent of staging_dir>
label-studio
```

It opens `http://localhost:8080`. On first use, create a local account
(email + password, stored on this machine only).

> **Shortcut:** steps 2–4 below (paste the labeling config, register Local
> Storage, import tasks) can be done in one command with
> `create_ls_project.py` instead — see "Automating project setup" after
> step 4. The manual steps are kept here since they're the reference for
> what that script does under the hood, and the fallback if you'd rather not
> use an API token.

### 2. Create the project (one-time)

**Create Project → Labeling Setup → Custom template** and paste this config:

```xml
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
```

The config is **per project**, so Label Studio prompts for it on **every** new
project. To avoid re-pasting: **reuse one project** and Import each sweep's tasks
into it (filter by `channel`/`bucket`/`clip_id` to keep sweeps or clips apart) — the template
is saved once and never asked again. If you do create a new project, copy the
config from the ready-made file the build writes next to the tasks:
`<staging_dir>/<nvr>/labelstudio/label_config.xml` (source: the `LABEL_CONFIG`
constant in `labelstudio_export.py`).

### 3. Register the staged images as Local Storage (required, once)

The env vars enable the *feature*, but Label Studio's `/data/local-files/`
endpoint only serves paths that belong to a **registered Local Storage** — without
this step, images 404 and you get the "issue loading URL from $image" error.

**Settings → Cloud Storage → Add Source Storage → "Local files"**, and set
*Absolute local path* to the **reviewing dir** (a subdir of the document root, which
the launcher sets to reviewing's parent — Label Studio rejects a path equal to the
document root):

```
<staging_dir>          # e.g. .../cctv_train_data_mining/reviewing
```

Leave "Treat every bucket object as a source file" **off**, click **Save**, and do
**NOT Sync** — the entry only needs to exist to authorize serving. This one entry
covers **every NVR** (tasks' `?d=reviewing/<nvr>/…` paths resolve under it), so you
register it once and never touch it again.

> **Why no Sync:** Sync walks the whole tree recursively and does *not* pick out
> `tasks_*.json`. With the toggle **on** it makes a task out of every file it finds
> (each `.png`, `.txt`, and `.json`) with no pre-labels; with it **off** it tries
> to parse each file as a JSON task and chokes on the images. Either way your
> pre-label predictions are lost. Tasks come from **Import** (step 4), which reads
> the `predictions` in each `tasks_*.json`; Local Storage only serves the images.

### 4. Import a sweep's tasks

**Import** → select `<staging_dir>/<nvr>/labelstudio/tasks_<stamp>_ch<NN>.json`. Each
task loads with its pre-labels as **predictions**. A sweep writes **one file per
channel** (for crash safety — see "Crash safety and resume"), so import each channel's
file; the Import dialog accepts multiple files at once. Import more tasks files later
as sweeps accumulate. Re-importing the *same* file would create duplicate tasks, so
if you resumed an interrupted sweep, import that channel's file once, after the sweep
finishes — the file is extended in place, not appended to a new one.

### Automating project setup: create_ls_project.py

`scripts/cctv/dataset_builder/create_ls_project.py` does steps 2–4 in one
command: it creates a **new** project with the labeling config already baked
in, registers the input file(s)' `staging_dir` as that project's Local
Storage (skip with `--no-storage`), and imports every task — aggregating
several `tasks_*.json` files into one project if you pass more than one.

```bash
export LABEL_STUDIO_API_TOKEN=...   # Personal Access Token, from the LS UI's Account & Settings page

# one channel's tasks -> a project named after the file
python3 scripts/cctv/dataset_builder/create_ls_project.py \
    data/cctv_train_data_mining/reviewing/<nvr>/labelstudio/tasks_<stamp>_ch00.json

# several channels' tasks aggregated into one project (name required with >1 file)
python3 scripts/cctv/dataset_builder/create_ls_project.py \
    data/cctv_train_data_mining/reviewing/<nvr>/labelstudio/tasks_<stamp>_ch*.json \
    --name "<nvr> sweep <stamp>"
```

Requires Label Studio already running with local-file serving enabled (step
1) — this script only talks to its API, it doesn't start the server.

**On the token:** Account & Settings hands you a long-lived *refresh* token,
not something usable directly against ordinary endpoints. By default the
script exchanges it once for a short-lived *access* token via
`/api/token/refresh/` and sends that as `Authorization: Bearer <access>`,
re-exchanging automatically if a call 401s mid-run (access tokens default to
a 5-minute lifetime) — you don't need to do anything for this, just pass the
token Account & Settings gave you. If your server has **legacy** tokens
enabled instead (Label Studio 1.23 default: disabled — see
"Troubleshooting" below), pass `--legacy-token` to send that token as-is,
`Authorization: Token <token>`, with no exchange. Run `create_ls_project.py
--help` for the rest of the flags (`--staging-dir` to override the inferred
Local Storage path, `--label-config` for a non-default template, `--url` for
a non-default host/port).

### 5. Review

Open each task and correct the pre-labels:
- **FN frames** (`n_intpfn > 0 OR n_weakfn > 0`): confirm/adjust the box on the
  person the model missed.
- **`SUSPECT_FP`** (transient) and **`SUSPECT_STATIC_FP`** (mannequin/poster):
  if it's truly not a person, delete the box (→ hard negative); if it *is* a real
  person, relabel it `person`. Click a box to see its `track N` id and detector
  confidence (`conf 0.XX`) in the region info panel — interpolated boxes (no
  real detection at that step) show no confidence, since there isn't one to
  report. A suspect box can *also* be tagged an anchor
  (`n_anchor_start`/`n_anchor_end > 0` on its frame) if its track borders a
  bridged gap — that's a signal worth acting on: it means the `Intp_FN` frames
  interpolated on either side of it share its `track_id` and were guessed from
  a bracket the tool itself doubts, so if you delete this suspect box, go check
  those `Intp_FN` frames too (filter `track_ids` for the same N) — they're
  likely spurious as well (`ALGORITHM.md` §4b).
- Add any person/animal the model missed entirely.
- Filter/sort in the Data Manager by the task `data` columns: `clip_id`,
  `channel`, `bucket`, `num_detected`, `n_person`, `track_ids`, `frame_idx`,
  `capture_time`, and the per-category counts `n_intpfn` / `n_weakfn` /
  `n_anchor_start` / `n_anchor_end` / `n_sfp` / `n_fp` / `n_easy`. Sort by `capture_time` to
  put frames in chronological order across clips and channels (task import order
  otherwise groups by channel first, and `clip_id`'s string form doesn't sort
  chronologically); filter to one `clip_id` and sort by `frame_idx` to walk one
  clip's selected frames in true video order. A frame can independently qualify for
  several of these at once (there is no single "reason" to pick between them —
  see `ALGORITHM.md` §5), so filter on whichever count you're hunting for
  rather than a single label. Review `Anchor_start`/`Anchor_end` frames as FP
  candidates — if an anchor detection is a false positive, delete it (its
  `Intp_FN` gap fills are spurious too); filter `n_anchor_start > 0 OR
  n_anchor_end > 0` to catch every gap-bordering detection, whether the endpoint
  itself was strong or weak (a weak anchor also carries `n_weakfn > 0` on the
  same box — both counts describe the same detection, not alternatives; see
  `ALGORITHM.md` §4b). `n_easy` is 1 exactly when every other per-category count
  above is 0 (`flicker_miner.frame_categories`'s fallback, see `ALGORITHM.md`
  §5) — filter `n_easy = 0` to hide every plain frame in one shot instead of
  ANDing six counts to 0, e.g. when a clip's `easy` quota (`max_easy_per_clip`)
  still leaves more of them in the queue than you want to look at. To hunt the
  program's **missed false positives** — objects it drew as real `person` on
  otherwise-unflagged frames — filter **`n_easy = 1 AND num_detected > 0`** and
  delete any box that isn't actually a person. Use
  `clip_id` to concentrate a review session on one clip (or a chosen few) at a
  time instead of the whole sweep.

Click **Submit** to store each corrected annotation.

### 6. Export

**Export → YOLO** → produces an `images/` + `labels/` folder of corrected data.
Promote it (below).

### Troubleshooting: "There was an issue loading URL from $image value"

Two independent causes — check both:

1. **Local Storage not registered** (most common; step 3). Even with serving
   enabled, the endpoint 404s until the NVR subdir is added as Local Storage.
   Fastest confirmation — open the image URL directly in your logged-in browser:
   ```
   http://<host>:8080/data/local-files/?d=reviewing/<nvr>/ch00/morning/<some>.png
   ```
   **404** (authenticated) → the path isn't covered by a registered storage → do
   step 3. **Image** → serving is fine; the fault is elsewhere.
2. **Serving not enabled** in the running server — started `label-studio` without
   the env vars, or in a different shell. Relaunch via `run_labelstudio.sh` (env
   changes need a restart). Confirm the live process actually has them:
   ```bash
   tr '\0' '\n' < /proc/$(pgrep -f label-studio | head -1)/environ | grep LABEL_STUDIO_LOCAL
   ```
   Both `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` and
   `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<absolute parent of staging_dir>` must be present.

Notes:
- `LOCAL_FILES_DOCUMENT_ROOT` is the absolute **parent of** `staging_dir`; the
  registered storage is `staging_dir` itself (a subdir of the root — LS rejects a
  storage path equal to the root). The launcher sets the root automatically.
- The `/data/local-files/` endpoint needs a valid **session** (browser login). An
  API-token `curl` may 401 on 1.23 (legacy token auth is disabled by default) —
  that 401 is about the token, not about serving; use the browser-URL test above.
- **Docker:** set the env vars on the container, bind-mount `staging_dir` in, and
  set `LOCAL_FILES_DOCUMENT_ROOT` to the **container-side** path.

## Config keys

| key | default | meaning |
|-----|---------|---------|
| `host`/`username`/`password` | — | NVR SUNAPI credentials (required) |
| `verify_ssl` / `rtsp_transport` / `rtsp_port` | `false` / `tcp` / `null` | transport (see SUNAPI CLAUDE.md) |
| `channels` | `"all"` | `"all"` (auto-discover) or a list like `[0,2,5]` |
| `model` | `docker_build/yolo11x_set01-0148.pt` | **deployed** model under test; repo-relative |
| `img_size` / `conf_thresh` | `640` / `0.6` | **must match production** — this defines the flicker |
| `track_conf` | `0.25` | recall conf for tracking (keeps weak/flickering persons) |
| `track_vid_stride` | `2` | process every Nth frame; smaller = better flicker fidelity, more compute |
| `iou_track` | `0.3` | IoU to associate detections into a track. Also the *effective* ID-switch guard, and the constraint that bounds both `bridge_*` values below (`ALGORITHM.md` §4a) |
| `max_gap_frames` | `15` | max processed-step gap to bridge/interpolate a track — **15 steps = 1.0 s** at stride 2, 30 fps |
| `bridge_max_disp_frac` | `0.45` | reject a gap bridge whose **total** centroid displacement across the gap exceeds this × mean box-diagonal. Calibrated as a high percentile of the band `iou_track` already permits (feasible max 0.523); any value ≥ that maximum is dead code. Recalibrate if `iou_track` changes |
| `bridge_max_scale_ratio` | `3.2` | reject a gap bridge whose two bracketing boxes differ in area by more than this factor. Feasible max is exactly `1/iou_track` = 3.333 |
| `fp_max_track_len` | `2` | **gap-free** tracks this short with a ≥`conf_thresh` detection → suspected (transient) FP — **2 steps = 0.13 s**. A track with a bridged gap is excluded regardless of this value (`ALGORITHM.md` §4b) |
| `static_min_frames` | `10` | minimum track length to be **flagged** SFP — **10 steps = 0.66 s**, deliberately short (a briefly-paused person may be flagged; that costs one relabel click) |
| `static_max_move_frac` | `0.15` | centroid spread < this × box-diagonal → stationary |
| `static_motion_thresh` | `0.08` | mean (1−ZNCC) below this → appearance-static → SFP candidate |
| `cross_clip_persistence` | `true` | enable the per-camera cross-clip fixture map |
| `persist_grid_cols` / `persist_grid_rows` | `64` / `36` | persistence-map grid resolution. Changing either invalidates an existing map (moved aside, rebuilt) |
| `persist_min_track_steps` | `60` | minimum track length to be **written into** the map — **60 steps = 4.0 s**. Far stricter than `static_min_frames` because map entries feed future decisions and no reviewer corrects them |
| `persist_min_clips` | `5` | **fixture-carrying** clips of history before the map is consulted |
| `persist_thresh` | `0.35` | cell value (a conditional probability, `ALGORITHM.md` §6b; queried as `max` over the box's cells) at which a location is called a fixture. Only meaningful relative to the `top_cell` value each sweep logs per channel |
| `max_intpfn_per_clip` / `max_weakfn_per_clip` / `max_anchor_per_clip` | `40` / `40` / `20` | per-category caps (the anchor cap applies to `Anchor_start` and `Anchor_end` separately). A cap is a **guarantee of evenly spread coverage**, not a shared budget: each category gets an even temporal subsample of its own candidates and the union is taken, so a category's realised count may exceed its cap when categories co-occur |
| `max_sfp_per_clip` / `max_fp_per_clip` | `30` / `20` | per-clip caps for SFP / FP |
| `max_easy_per_clip` | `10` | `easy` budget for a clip that produced **at least one** candidate |
| `max_easy_barren_clip` | `1` | `easy` budget for a clip that produced **no** candidate. 90 % of clips are barren, so without this split `easy` was 88 % of the review queue; kept above 0 because empty scenes are the cleanest source of background negatives |
| `easy_every_n` | `5` | minimum spacing, in processed steps, between selected `easy` frames (near-duplicate guard); rarely binding once the even spread applies |
| `clip_duration_sec` | `60` | length of each sampled clip |
| `clip_download_timeout_sec` | `60` | wall-clock cap on each ffmpeg download; a stalled RTSP backup session (dead channel / no recording for the window) is killed so it can't hang the sweep. Healthy backup downloads finish in a few seconds, so this can be lowered to skip dead windows faster |
| `clip_end_margin_sec` | `120` | window ends this many seconds before now (so footage is recorded) |
| `lookback_hours` | `24` | rolling window depth — a sweep samples clips across `[now−margin−lookback, now−margin]` |
| `window_start` / `window_end` | `null` | optional absolute ISO window (set **both** to pin a fixed historical range; overrides `lookback_hours`) |
| `clip_interval_min` | `30` | sample one clip every this many minutes across the window (→ `lookback_hours×60 / clip_interval_min` clips per channel); ignored with `--once` |
| `staging_dir` | `data/cctv_train_data_mining/reviewing` | output root (repo-relative) |
| `keep_clips` | `false` | keep the downloaded mp4s under `.clips/` |
| `annotate_clips` | `true` | when `keep_clips` is on: also write `<clip>_annotated.mp4` with the mining boxes drawn over the whole clip (green = detected, red = missed/FN, yellow = FP?, magenta = static-FP?, orange = animal) for verifying the mining against the full video |

## Promote reviewed data

After exporting corrected YOLO data from Label Studio:

```bash
# Run this command at ~/Work/spacenorm_obj_detection/
python3 scripts/cctv/dataset_builder/promote_to_trainset.py --staging <label-studio-YOLO-export-dir> \
    --images data/cctv_train_data_mining/reviewing
```

Label Studio's YOLO export contains `labels/` + `classes.txt` but an **empty
`images/`** when tasks are served from Local Storage (it never stored the image
files). So pass **`--images`** pointing at the reviewing tree — promote pairs each
exported label with the original PNG **by filename** (the export keeps the
descriptive basenames). If the export *does* include images, `--images` is optional.

**Skip the manual export entirely with `--ls-project`:** instead of clicking
Export in the UI and pointing `--staging` at the downloaded zip, fetch the
export directly over the API:

```bash
export LABEL_STUDIO_API_TOKEN=...   # Personal Access Token, from Account & Settings
python3 scripts/cctv/dataset_builder/promote_to_trainset.py --ls-project 42 \
    --images data/cctv_train_data_mining/reviewing
```

`--ls-project` takes one or more project ids (or pasted project URLs like
`http://localhost:8080/projects/42/data` — whatever's on the clipboard) and
can be mixed with `--staging` in the same run; each project, like each
`--staging` item, becomes its own new `setNNNN`. It fetches only **annotated**
tasks (same as the UI's Export button — unreviewed pre-labels are never
included), so the "never on raw, unreviewed pre-labels" rule above still
holds. Auth works the same as `create_ls_project.py` — see that script's
section above for the Personal Access Token / `--legacy-token` note; `--url`/
`--token`/`--legacy-token` are only required when `--ls-project` is used.

Creates the next `data/cctv_train_data/setNNNN/` (flat jpg+txt) — that's all it
writes; the train/val split is left to the notebook (below).

**It skips frames already promoted.** One Label Studio project accumulates every
sweep, and "Export" covers the whole project unless you filter it, so two exports
easily overlap. Promoting both would copy the same frame into two `setNNNN/`
folders: the frame then enters the training list twice (double weight) and a stale
copy can contradict a later corrected one. Nothing downstream catches this —
`count_files.py` checks jpg/txt pairing *within* a set, not identity across sets. So
promote matches each frame's basename against every existing set, reports which set
already holds it, and skips it. Use `--allow-duplicates` to override.

**Unparseable label lines are dropped, loudly.** A line whose first field is not an
integer, or that does not have exactly five fields, is not a YOLO detection label;
copying it through would write a corrupt label file into the training set, where the
trainer either crashes or silently misreads it. Such frames lose boxes, so the
warning names the file and the offending lines — investigate the export before
training on that set.

**It remaps class ids to the trainset taxonomy by name** using the export's `classes.txt`:
Label Studio numbers YOLO classes **alphabetically** (e.g. `SUSPECT_FP=0`, `bird=2`,
`person=7`), which is *not* `person(0)..cow(6)`, so a positional copy would be
wrong. Promote reads `classes.txt`, remaps each box by its class name, and **drops**
`SUSPECT_FP`/`SUSPECT_STATIC_FP` and any non-trainset class; an image whose only
labels were suspects becomes a clean negative. It prints the detected mapping and
how many lines it dropped. Then regenerate
the cumulative `train_01_to_NNNN.txt` / `test_01_to_NNNN.txt` (referenced by
`data/cctv.yaml`) via `make_train_test_single_multi.ipynb`, add a line to
`data/cctv_train_data/.NOTE`, and verify pairing with
`python3 data/cctv_train_data/count_files.py`.

> You can also point `promote_to_trainset.py --staging` at a hand-reviewed folder
> directly, but the intended path is Label-Studio-corrected data — raw pre-labels
> must never be promoted unreviewed.

## Caveats

- **Review before promotion** — pre-labels are the deployed model's best guess, not truth.
- Tuning `conf_thresh`/`img_size` away from production changes *which* flicker you
  see; keep them at the deployed values (640 / 0.6, matching
  `spacenorm_cfg/behavior/default.json`) to mine real detector errors.
- `manifest.csv` has no schema version in its rows, so when the column set changes
  between builds the existing file is moved aside to `manifest.csv.v<N>.bak` and a
  new one is started. Rows are never appended under a mismatched header (that would
  misalign every column silently). Keep the `.bak` — it is the only index of frames
  staged by the older build.
- Re-running the **same** window **resumes** rather than restarts — see "Crash
  safety and resume" below. Clips recorded in `mined_clips.txt` are skipped, so no
  re-inference, no duplicate manifest rows, and no double-counted persistence
  evidence.
- Assumes builder-host local time == NVR local time (see `clip_end_margin_sec`).
- Reachability varies per NVR: `cheilacc-ansung.spacenorm.com` and
  `yujin.spacenorm.com` are confirmed working end-to-end; others may need
  `rtsp_port` / `rtsp_transport: "http"` — see `scripts/cctv/SUNAPI/CLAUDE.md`.
  Expect some download failures even on a healthy NVR (144 of 768 attempts in one
  measured 24 h sweep, from windows with no recording); they are logged and skipped.
```
