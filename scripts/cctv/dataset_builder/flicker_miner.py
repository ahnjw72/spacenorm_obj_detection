"""flicker_miner.py — mine the DEPLOYED model's errors from a video clip.

Objective (project owner's bottom line): nothing here is treated as ground
truth. The job is to produce ALMOST-CORRECT pre-labels that are rich in the
deployed model's errors, so a human can review/correct them cheaply.

Three error types are mined (raw-model behaviour; deployed model + temporal
tracking, no second model):

  * FN  (flicker miss)  — a person present/interpolated that the production view
    (conf >= conf_thresh) missed. Primary target. Pre-labelled with its box.
  * FP  (transient)     — a production detection whose track is a short blip
    (<= fp_max_track_len frames): no temporal support.
  * SFP (static FP)     — a PERSISTENT, STATIONARY, high-confidence person track
    whose box crop barely changes over time under an illumination-INVARIANT
    measure (ZNCC). This catches human-like apparatus (mannequins, posters,
    standees) that MOG2 background subtraction leaks through when lighting
    drifts. Flagged for review as SUSPECT_STATIC_FP — a human confirms
    mannequin (delete -> hard negative) vs a real person who stood still.

Pass 1 collects detections + tiny illumination-normalisable crops (memory-light)
AND caches every processed step's full frame, losslessly, to a per-clip temp
directory keyed by raw_idx. Selection then reads back only the wanted frames
from that cache — there is no second video decode — and the whole cache is
deleted immediately after, win or lose. See ALGORITHM.md 3 for why the cached
copy must be lossless and why it is staged to disk rather than held in RAM.
"""
import logging
import os
import shutil
import tempfile
from collections import Counter

import cv2
import numpy as np

logger = logging.getLogger("build_dataset.flicker")

TRAINSET_NAMES = ["person", "bird", "cat", "dog", "horse", "sheep", "cow"]
PERSON_ID = 0

_CROP_W, _CROP_H = 32, 64   # normalized person-crop size for appearance stability


# ---------------------------------------------------------------------------
# Geometry / appearance helpers
# ---------------------------------------------------------------------------
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def interp_box(b0, b1, frac):
    return tuple(b0[k] + (b1[k] - b0[k]) * frac for k in range(4))


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _diag(box):
    return ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5


def _bridgeable(box_p, box_q, max_disp_frac, max_scale_ratio):
    """True if the two detections bracketing a gap are geometrically consistent
    with ONE object, so interpolating across the gap is sound. Rejects the
    implausible jumps a tracker ID switch produces — otherwise a phantom person is
    interpolated between two *different* people (a bogus Intp_FN).

    Two cheap, appearance-free tests:
      * displacement — total centroid displacement across the gap must stay within
        ``max_disp_frac`` of the mean box diagonal;
      * scale        — the two boxes must not differ in area by more than
        ``max_scale_ratio`` (a near-vs-far person swap changes size sharply).

    SCOPE OF THIS GATE (important; do not over-credit it). Both tests are
    necessarily a TIGHTENING of ``iou_track``, not independent evidence. A track
    receives no detection during its own gap, so its ``active`` entry still holds
    ``box_p``; the association step therefore already required
    ``IoU(box_p, box_q) >= iou_track``. That single constraint bounds both
    quantities tested here:

        iou_track   max feasible dist/diag_avg   max feasible area ratio
        0.1         0.799                        10.000  (= 1/iou_track, exact)
        0.2         0.636                         5.000
        0.3         0.523                         3.333
        0.5         0.314                         2.000

    (Displacement column: 60k-sample rejection-sampled maximum over person-like
    boxes. Area-ratio column is closed form: IoU <= min_area/max_area, so
    area_ratio <= 1/IoU.) A threshold at or above the feasible maximum can never
    reject anything. The primary ID-switch protection is ``iou_track`` itself plus
    the short ``max_gap_frames`` horizon; this gate only trims the geometrically
    most implausible tail of what association already allowed.

    Thresholds are therefore calibrated as high percentiles of the feasible band
    at the deployed ``iou_track`` (0.3), so the gate is non-vacuous by
    construction while discarding almost no review candidates — see
    ALGORITHM.md 4.
    """
    diag = 0.5 * (_diag(box_p) + _diag(box_q))
    if diag <= 0:
        return False
    cp, cq = _center(box_p), _center(box_q)
    dist = ((cp[0] - cq[0]) ** 2 + (cp[1] - cq[1]) ** 2) ** 0.5
    if dist > max_disp_frac * diag:
        return False
    ap = max(1e-6, (box_p[2] - box_p[0]) * (box_p[3] - box_p[1]))
    aq = max(1e-6, (box_q[2] - box_q[0]) * (box_q[3] - box_q[1]))
    if max(ap / aq, aq / ap) > max_scale_ratio:
        return False
    return True


def _crop(gray, box):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = gray.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return cv2.resize(gray[y1:y2, x1:x2], (_CROP_W, _CROP_H)).astype(np.float32)


def _zncc(a, b):
    """Zero-mean normalized cross-correlation of two equal-size crops.

    Invariant to affine illumination change (brightness/contrast). ~1.0 means
    structurally identical; lower means the appearance changed (motion)."""
    za, zb = a - a.mean(), b - b.mean()
    denom = float(np.sqrt((za * za).sum() * (zb * zb).sum()))
    if denom < 1e-6:
        # A flat/untextured crop is no clue of a static object. Return 0
        # (uncorrelated) so it counts as motion, not as static — SFP is a strong
        # claim we only make on textured, appearance-stable objects.
        return 0.0
    return float((za * zb).sum() / denom)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _detect(model, img, imgsz, conf):
    """Return (persons, animals). persons=[(box,conf)], animals=[(box,conf,cls)]."""
    results = model(img, imgsz=imgsz, conf=conf, verbose=False)
    persons, animals = [], []
    if results and len(results[0].boxes):
        r = results[0]
        for i in range(len(r.boxes)):
            box = tuple(float(x) for x in r.boxes.xyxy[i].cpu().tolist())
            c = float(r.boxes.conf[i].cpu())
            cls = int(r.boxes.cls[i].cpu())
            if cls == PERSON_ID:
                persons.append((box, c))
            elif cls in range(1, len(TRAINSET_NAMES)):
                animals.append((box, c, cls))
    return persons, animals


# ---------------------------------------------------------------------------
# Greedy IoU tracker over processed steps
# ---------------------------------------------------------------------------
def _build_tracks(person_steps, iou_track, max_gap_steps):
    """person_steps: list per step of person dets (each a dict with 'box').

    Returns tid -> sorted list of (step, det) via greedy IoU association."""
    tracks = {}
    active = []
    next_tid = 0
    for s, dets in enumerate(person_steps):
        active = [t for t in active if s - t["last_step"] <= max_gap_steps]
        pairs = []
        for di, det in enumerate(dets):
            for t in active:
                ov = iou(det["box"], t["box"])
                if ov >= iou_track:
                    pairs.append((ov, di, t["tid"]))
        pairs.sort(key=lambda p: p[0], reverse=True)
        det_taken, trk_taken, assign = set(), set(), {}
        for ov, di, tid in pairs:
            if di in det_taken or tid in trk_taken:
                continue
            det_taken.add(di); trk_taken.add(tid); assign[di] = tid
        for di, det in enumerate(dets):
            tid = assign.get(di)
            if tid is None:
                tid = next_tid; next_tid += 1; tracks[tid] = []
            tracks[tid].append((s, det))
            active = [t for t in active if t["tid"] != tid]
            active.append({"tid": tid, "box": det["box"], "last_step": s})
    return tracks


def _track_box_at(seq, s, prod_conf, max_gap_steps,
                  max_disp_frac, max_scale_ratio):
    """(box, source, conf) of a track at step s, or (None, None, None).

    source: 'strong' (present, conf>=prod), 'weak' (present, conf<prod),
            'interp' (interpolated across a short gap). ``conf`` is the
            detector's own confidence for 'strong'/'weak' (a real detection
            exists at this step); 'interp' has no detection at this step at
            all, so there is nothing real to report and conf is None rather
            than a fabricated (e.g. averaged) number."""
    if s < seq[0][0] or s > seq[-1][0]:
        return None, None, None
    prev = nxt = None
    for (ss, det) in seq:
        if ss == s:
            return (det["box"],
                    ("strong" if det["conf"] >= prod_conf else "weak"),
                    det["conf"])
        if ss < s:
            prev = (ss, det)
        elif ss > s:
            nxt = (ss, det); break
    if prev is None or nxt is None or nxt[0] - prev[0] > max_gap_steps:
        return None, None, None
    if not _bridgeable(prev[1]["box"], nxt[1]["box"],
                       max_disp_frac, max_scale_ratio):
        return None, None, None      # implausible jump -> likely an ID switch, not a gap
    frac = (s - prev[0]) / (nxt[0] - prev[0])
    return interp_box(prev[1]["box"], nxt[1]["box"], frac), "interp", None


def _median_box(boxes):
    n = len(boxes)
    return tuple(sorted(b[k] for b in boxes)[n // 2] for k in range(4))


def _static_fp_tids(tracks, cfg, persist, W, H):
    """Suspected human-like apparatus FPs, from two signals:

      * within-clip: persistent + stationary + appearance-static under ZNCC
        (illumination invariant); and/or
      * cross-clip: the stationary track sits on a location that hosted a
        fixture-like track in many independent clips (persistence map), when
        enabled and trusted.

    Returns ``(sfp_tids, map_boxes)``.

    TWO DIFFERENT PREDICATES, DELIBERATELY (see ALGORITHM.md 6):

    * ``sfp_tids`` — what gets FLAGGED FOR REVIEW. Recall-oriented and
      deliberately loose: ``static_min_frames`` is short, so a person who merely
      paused briefly can land here. That is acceptable by design — a wrongly
      flagged box is one relabel click in Label Studio, and this project's
      objective is candidate recall for the reviewer, not label precision.

    * ``map_boxes`` — what is WRITTEN INTO the cross-clip persistence map, i.e.
      accumulated evidence that a screen location holds a fixture. This must be
      much stricter, because the map's own output feeds signal 2 above: if the map
      accumulated every briefly-stationary person, then any location where people
      habitually pause (reception desk, workstation, queue position) would
      eventually cross ``persist_thresh`` and start auto-flagging real people —
      a self-reinforcing loop with no external correction. A track therefore
      contributes to the map only if it is stationary AND appearance-static under
      ZNCC AND at least ``persist_min_track_steps`` steps long.

    ``ChannelPersistence.add_clip`` counts a clip in the map's denominator only
    when ``map_boxes`` is non-empty, so the map's stored value per cell is
    P(cell hosted a fixture-like track | this clip had a fixture-like track
    somewhere) — the SAME predicate in numerator and denominator. That
    conditioning is what separates a fixture (present at one cell in essentially
    every such clip => ratio -> 1.0) from transient standing people (scattered
    over many cells, each hit once or twice => ratio -> 1/N).
    """
    sfp = set()
    map_boxes = []    # median boxes of FIXTURE-LIKE tracks only
    for tid, seq in tracks.items():
        if len(seq) < cfg.static_min_frames:
            continue
        boxes = [det["box"] for (_s, det) in seq]
        centers = [_center(b) for b in boxes]
        mcx = sum(c[0] for c in centers) / len(centers)
        mcy = sum(c[1] for c in centers) / len(centers)
        spread = max(((cx - mcx) ** 2 + (cy - mcy) ** 2) ** 0.5 for (cx, cy) in centers)
        diag = sum(_diag(b) for b in boxes) / len(boxes)
        if diag <= 0 or spread > cfg.static_max_move_frac * diag:
            continue  # it moves -> not a static apparatus
        med = _median_box(boxes)

        # Signal 1: appearance is static under an illumination-invariant measure.
        appearance_static = False
        crops = [det["crop"] for (_s, det) in seq if det.get("crop") is not None]
        if len(crops) >= 2:
            motion = np.mean([1.0 - _zncc(crops[i - 1], crops[i]) for i in range(1, len(crops))])
            appearance_static = bool(motion <= cfg.static_motion_thresh)

        if appearance_static:
            sfp.add(tid)
            # Only a LONG appearance-static track is fixture-like enough to be
            # evidence about this location for future clips.
            if len(seq) >= cfg.persist_min_track_steps:
                map_boxes.append(med)
        elif persist is not None and persist.is_persistent(
                med, W, H, cfg.persist_thresh, cfg.persist_min_clips):
            # Signal 2: this location hosted a fixture-like track in a large
            # fraction of the clips that had one at all.
            sfp.add(tid)
    return sfp, map_boxes


CATEGORY_CAP_KEYS = {
    "Intp_FN": "n_intpfn", "Weak_FN": "n_weakfn",
    "SFP": "n_sfp", "FP": "n_fp",
    "Anchor_start": "n_anchor_start", "Anchor_end": "n_anchor_end",
    "Track_context": "n_context",
}

# Categories exempt from _select_frames's per-clip quotas: every candidate is
# kept, not an evenly-spread subsample of one. Currently just Track_context —
# see its rationale where n_context is computed in mine_clip.
UNCAPPED_CATEGORIES = {"Track_context"}


def frame_categories(fr):
    """Every category this frame independently qualifies for, from its n_*
    counts alone. A frame can belong to any number at once (e.g. a missed
    person AND an unrelated anchor). 'easy' means none of the others fired —
    either a plain accepted detection or nothing detected at all."""
    cats = {cat for cat, key in CATEGORY_CAP_KEYS.items() if fr[key]}
    return cats or {"easy"}


def _spread(idx, k, min_gap=1):
    """Choose at most ``k`` entries of the ascending step-index list ``idx``,
    spread EVENLY over its whole range (both endpoints included), honouring a
    minimum spacing of ``min_gap`` steps between picks.

    Why even spreading rather than "take the first k": quotas are filled by
    walking the clip, so a quota that binds would otherwise be satisfied entirely
    from the clip's opening steps and the rest of the clip would never be sampled.
    With ``max_easy_per_clip`` = 10 and ``easy_every_n`` = 5 the old first-k rule
    drew every ``easy`` frame from raw indices 0,10,...,90 — the first 3.3 s of a
    60 s clip (measured over 18720 staged easy frames; all of them). Even
    spreading makes a binding quota a uniform temporal SUBSAMPLE of the clip
    instead of a prefix of it.
    """
    if k <= 0 or not idx:
        return []
    if len(idx) <= k:
        picks = list(idx)
    elif k == 1:
        picks = [idx[len(idx) // 2]]
    else:
        # k evenly spaced positions across idx, first and last included.
        picks = [idx[round(t * (len(idx) - 1) / (k - 1))] for t in range(k)]
    out = []
    for p in picks:
        if not out or p - out[-1] >= min_gap:
            out.append(p)
    return out


def _select_frames(frames, cfg):
    """Pick frames for human review under six INDEPENDENT per-category quotas
    (each config's ``max_*_per_clip``) — no single winning "reason" per frame.

    Each category's quota is filled by an EVEN temporal spread over that
    category's own candidate frames (``_spread``), and the selection is the UNION
    over categories.

    Semantics of a cap, changed deliberately: a cap is now a per-category
    GUARANTEE OF EVENLY SPREAD COVERAGE, not a shared budget. Previously a frame
    spent against every category it touched, so a frame carrying both `Intp_FN`
    and `Anchor_start` consumed anchor budget that no anchor-only frame could then
    use — a category could be starved by an unrelated one. Under the union rule
    every category independently gets min(cap, available) frames' worth of
    coverage, and the selected count for a category may EXCEED its cap when
    frames co-occur. Over-shooting a cap is harmless here (the cap exists only to
    bound near-duplicate frames from one clip) and strictly increases candidate
    recall, which is this project's objective.

    ``easy`` frames (no category fired) get a CONDITIONAL budget: a clip that
    produced at least one error candidate contributes up to
    ``max_easy_per_clip``, a clip that produced none contributes only
    ``max_easy_barren_clip``. Rationale, from a measured 24 h sweep: 560 of 624
    mined clips yielded no candidate at all, yet each still contributed its full
    10 easy frames, so 6240 of 7089 selected frames (88 %) were `easy` and 5600 of
    those came from clips with nothing to review. The reviewer's queue was
    dominated by frames the miner has no hypothesis about. Barren clips are still
    the cleanest source of background negatives for retraining, so the budget is
    reduced rather than removed: one frame per barren clip keeps a temporally
    diverse negative pool at a tenth of the review cost.

    ``easy_every_n`` is a MINIMUM SPACING in processed steps between selected easy
    frames (a near-duplicate guard), not a modulo filter on the frame index.
    """
    cap_of = {"Intp_FN": cfg.max_intpfn_per_clip, "Weak_FN": cfg.max_weakfn_per_clip,
              "SFP": cfg.max_sfp_per_clip, "FP": cfg.max_fp_per_clip,
              "Anchor_start": cfg.max_anchor_per_clip, "Anchor_end": cfg.max_anchor_per_clip}
    # Uncapped categories (currently just Track_context) get every candidate,
    # not an evenly-spread subsample: len(frames) is a cap no real count can
    # ever reach.
    cap_of.update({cat: len(frames) for cat in UNCAPPED_CATEGORIES})
    cats_of = [frame_categories(fr) for fr in frames]

    # Per-category candidate step-index lists (ascending, by construction).
    idx_of = {cat: [] for cat in cap_of}
    easy_idx = []
    for i, cats in enumerate(cats_of):
        if cats == {"easy"}:
            easy_idx.append(i)
            continue
        for c in cats:
            idx_of[c].append(i)

    keep = set()
    for cat, idx in idx_of.items():
        keep.update(_spread(idx, cap_of[cat]))

    has_candidates = any(idx_of[c] for c in idx_of)
    easy_cap = cfg.max_easy_per_clip if has_candidates else cfg.max_easy_barren_clip
    keep.update(_spread(easy_idx, easy_cap, min_gap=max(1, int(cfg.easy_every_n))))

    selected = [frames[i] for i in sorted(keep)]     # ascending frame order (ALGORITHM.md 9)
    n = Counter()
    for i in sorted(keep):
        for c in cats_of[i]:
            n[c] += 1
    return selected, n


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def mine_clip(clip_path, model, cfg, persist=None, update_persist=False):
    """Return ``(selected_records, all_frame_annotations)``.

    ``all_frame_annotations`` is the per-processed-step classification for the
    WHOLE clip (same dict shape as a record but without ``image``); pass it to
    ``render_annotated_clip`` to draw the mining result over the full video for
    verification. Each selected record:

    { raw_idx, width, height,
      n_intpfn, n_weakfn, n_anchor_start, n_anchor_end, n_fp, n_sfp, n_person, animals [names],
      persons [(box, source, tid, cats, conf)], animal_boxes [(box, cls, conf)],
      suspect [(box, kind, tid, conf)]  kind in {'static','transient'},
      conf is None for an 'interp' box (no real detection exists at that step),
      else the detector's own confidence for the box,
      image (BGR ndarray) }

    There is no single per-frame "reason": a frame independently belongs to any
    number of the six categories at once (each driven by its own n_* count).
    ``frame_categories()`` below derives that set on demand — for selection here,
    for staging/logging in build_dataset.py, or for ad-hoc inspection — rather
    than collapsing it into one precedence-ordered label.
    """
    stride = max(1, int(cfg.track_vid_stride))
    prod_conf = cfg.conf_thresh
    max_gap = int(cfg.max_gap_frames)

    # ---- Pass 1: detections + crops + track metadata + a lossless per-step
    # frame cache. The cache holds the EXACT array _detect just scored, written
    # to a per-clip temp dir keyed by raw_idx: a frame later selected for review
    # (and, eventually, retraining) must be byte-identical to what produced its
    # recorded confidence, which neither a second video decode nor a lossy
    # re-encode (e.g. JPEG) can guarantee — see ALGORITHM.md 3. Whatever is not
    # selected is deleted with the rest of this directory before mine_clip
    # returns, so nothing here is ever visible outside one clip's processing.
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        logger.warning(f"      could not open clip {clip_path}")
        return [], []
    tmp_root = os.path.join(cfg.staging_dir, ".tmp_pass1")
    os.makedirs(tmp_root, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_root)

    def _cache_path(idx):
        return os.path.join(tmp_dir, f"{idx:06d}.png")

    step_meta = []
    raw_idx = -1
    while True:
        if not cap.grab():
            break
        raw_idx += 1
        if raw_idx % stride != 0:
            continue
        ok, img = cap.retrieve()
        if not ok:
            break
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        persons, animals = _detect(model, img, cfg.img_size, cfg.track_conf)
        pdets = [{"box": box, "conf": c, "crop": _crop(gray, box)} for (box, c) in persons]
        step_meta.append({"raw": raw_idx, "W": W, "H": H, "persons": pdets, "animals": animals})
        # Compression level, not quality: PNG is always lossless, level only
        # trades write speed for size. Kept low since most of these frames are
        # deleted unread a moment later.
        cv2.imwrite(_cache_path(raw_idx), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    cap.release()
    if not step_meta:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return [], []

    tracks = _build_tracks([m["persons"] for m in step_meta], cfg.iou_track, max_gap)
    track_len = {tid: len(seq) for tid, seq in tracks.items()}
    # Steps that border an interpolatable gap (the "seemingly true" detections used
    # as anchors to fill an FN gap) — flagged separately since they are themselves
    # strong FP candidates.
    anchor_start, anchor_end = {}, {}
    for tid, seq in tracks.items():
        starts, ends = set(), set()
        for i in range(len(seq) - 1):
            (p, dp), (q, dq) = seq[i], seq[i + 1]
            # Same plausibility gate as the interpolation (§ _track_box_at): only a
            # bridge that could be one moving object gets anchors + Intp_FN fills,
            # so an ID-switch gap does not spawn phantom anchors either.
            if 1 < q - p <= max_gap and _bridgeable(
                    dp["box"], dq["box"],
                    cfg.bridge_max_disp_frac, cfg.bridge_max_scale_ratio):
                starts.add(p)   # p OPENS the gap
                ends.add(q)     # q CLOSES the gap
        anchor_start[tid] = starts
        anchor_end[tid] = ends
    # A track that bridges even one gap is excluded from transient-FP
    # consideration entirely (see fp_tids below) — it has anchor_start/
    # anchor_end entries iff it does, so this is a free byproduct of the loop
    # just run, not a new computation.
    track_has_bridged_gap = {tid: bool(anchor_start[tid] or anchor_end[tid]) for tid in tracks}
    # A track that ever went weak, or ever bridged a gap, is direct temporal
    # evidence of the SAME real object being inconsistently detected — its
    # ordinary `strong` steps are then valuable CONTEXT for a reviewer judging
    # that inconsistency (what did the detector get right on this exact
    # person, right next to where it got it wrong?), not just background
    # agreement. A track that is uniformly strong throughout has no such
    # evidence to contextualise and is left as ordinary `easy`.
    track_has_context = {
        tid: track_has_bridged_gap[tid] or any(d["conf"] < prod_conf for _s, d in seq)
        for tid, seq in tracks.items()
    }
    W0, H0 = step_meta[0]["W"], step_meta[0]["H"]
    # Decide SFPs using the PRIOR persistence state, then fold this clip in.
    # Only FIXTURE-LIKE tracks (see _static_fp_tids) are folded in, and add_clip
    # counts the clip in the map's denominator only if there was at least one.
    sfp_tids, map_boxes = _static_fp_tids(tracks, cfg, persist, W0, H0)
    if persist is not None and update_persist:
        persist.add_clip(map_boxes, W0, H0)

    # ---- Classify every processed step ----
    frames = []
    for s, meta in enumerate(step_meta):
        oracle = []  # (tid, box, source, conf)
        for tid, seq in tracks.items():
            box, src, conf = _track_box_at(seq, s, prod_conf, max_gap,
                                           cfg.bridge_max_disp_frac,
                                           cfg.bridge_max_scale_ratio)
            if box is not None:
                oracle.append((tid, box, src, conf))

        # Transient-FP tids that actually produce a >=prod detection at this step.
        # Two independent gates, both required: (1) the track's TOTAL detection
        # count is short (track_len <= fp_max_track_len) -- the case-2 "very
        # short strong detection" signature; (2) the track has NO bridged gap
        # anywhere in it. Gate (2) exists because (1) alone cannot tell apart a
        # single brief phantom (2 CONSECUTIVE detections, span 1 step) from two
        # detections bridged across up to max_gap_frames (~1s) of total absence
        # -- and the second shape is structurally identical to a genuine person
        # briefly, completely occluded then reappearing at nearly the same spot,
        # which is exactly this project's PRIMARY target (req-1's flicker), not
        # a phantom. Verified directly: two constructed, wholly independent
        # isolated strong detections ~1s apart (no weak evidence anywhere, so
        # neither looks like case-1 either) still satisfy _bridgeable and get
        # merged + confidently interpolated -- track shape alone cannot
        # distinguish "one real, briefly-occluded object" from "two unrelated
        # phantoms that happened to land in compatible boxes" (the same kind of
        # residual, appearance-free limitation as the ID-switch risk in §7).
        # Given that ambiguity is unresolvable from geometry, a gap-bridging
        # track is deliberately given the BENEFIT OF THE DOUBT (routed to the
        # ordinary Anchor_start/Intp_FN/Anchor_end path -- "assume real, flag
        # for the reviewer to check", per §4c's existing anchor-review guidance)
        # rather than defaulted to SUSPECT_FP ("assume false") the way a truly
        # gap-free short track still is.
        fp_tids = set()
        for tid, seq in tracks.items():
            if (tid in sfp_tids or track_len[tid] > cfg.fp_max_track_len
                    or track_has_bridged_gap[tid]):
                continue
            for (ss, det) in seq:
                if ss == s and det["conf"] >= prod_conf:
                    fp_tids.add(tid)

        suspect, person_labels = [], []
        intp_fn, weak_fn, a_start, a_end, ctx = [], [], [], [], []
        for (tid, box, src, conf) in oracle:
            # ANCHOR dimension — INDEPENDENT of the SUSPECT dimension below, of
            # the RESULT dimension, and of itself (start vs end). Whether this
            # step borders a bridgeable gap in its OWN track is a structural
            # fact about track topology (computed in anchor_start/anchor_end
            # without regard to confidence or suspicion, §4), not a property of
            # what the model produced, or what the miner otherwise suspects,
            # here. So a box already flagged SUSPECT_FP/SUSPECT_STATIC_FP CAN
            # also be an anchor: e.g. a short (<= fp_max_track_len), otherwise
            # unrelated track can still bridge a gap between its two total
            # sightings, so its correctly-suspected endpoint should still be
            # findable via n_anchor_start/n_anchor_end, and, via the tid it
            # shares with the gap's interior Intp_FN fills, let a reviewer
            # connect "this bracket is a suspected phantom" to "so are the
            # person boxes interpolated between its two brackets" (§4b, §7).
            # Before this fix, `continue` on the SUSPECT branches skipped this
            # entirely, silently discarding the anchor fact for any suspect
            # box — the same class of bug the RESULT/ANCHOR independence fix
            # (§4b) addressed one level up. A single step can also be BOTH an
            # Anchor_end (closing one gap) and an Anchor_start (opening the
            # next) at once, since anchor_start[tid]/anchor_end[tid] are
            # independent sets, not alternatives (§4's "intermixed detections"
            # worked example). 'interp' can never match either set: both are
            # built exclusively from `seq`, which holds only DETECTED steps, so
            # an interpolated step is provably never a member.
            is_start = s in anchor_start.get(tid, ())            # detection OPENING an interp gap
            is_end = s in anchor_end.get(tid, ())                # detection CLOSING an interp gap
            if is_start:
                a_start.append(box)
            if is_end:
                a_end.append(box)

            # SUSPECT dimension — mutually exclusive by construction (a track
            # cannot satisfy both static_min_frames and <= fp_max_track_len
            # under any sane config): is this box's OWN track itself considered
            # a likely false detection (persistent apparatus, or an isolated
            # no-temporal-support blip)? This governs box ORIGIN (rendered
            # SUSPECT_FP/SUSPECT_STATIC_FP, withheld from the pre-label .txt)
            # — it says nothing about whether this step also borders a gap.
            if tid in sfp_tids:
                suspect.append((box, "static", tid, conf))
            elif tid in fp_tids:
                suspect.append((box, "transient", tid, conf))
            else:
                # cats: this specific box's own category tags (not the frame's
                # aggregate counts), threaded through to the LS task's per-box
                # `meta` so a reviewer can see why a given box exists without
                # cross-referencing the frame-level counts (useful once a
                # frame carries more than one box, §5b).
                cats = []
                # RESULT dimension — mutually exclusive by construction: 'weak'/
                # 'interp' already mean "no strong (>=conf_thresh) box for this
                # track here" (_track_box_at guarantees a track is strong XOR
                # weak XOR interp at any one step) — that's a miss by
                # definition, no further test needed. 'strong' falls through
                # (nothing to append here).
                if src == "interp":
                    intp_fn.append(box)                        # interpolated gap the model missed
                    cats.append("Intp_FN")
                elif src == "weak":
                    weak_fn.append(box)                        # sub-threshold detection the model missed
                    cats.append("Weak_FN")
                if is_start:
                    cats.append("Anchor_start")
                if is_end:
                    cats.append("Anchor_end")
                if not cats and track_has_context[tid]:
                    # A plain strong, non-anchor step whose track is
                    # otherwise inconsistent (weak somewhere, or bridged a
                    # gap) elsewhere. Kept as direct evidence of the
                    # detector's own inconsistency on this exact object,
                    # exempt from the easy budget (UNCAPPED_CATEGORIES).
                    ctx.append(box)
                    cats.append("Track_context")
                person_labels.append((box, src, tid, cats, conf))    # real-person pre-label

        animal_labels = [(b, cls, c) for (b, c, cls) in meta["animals"] if c >= prod_conf]
        animal_names = sorted({TRAINSET_NAMES[cls] for (_b, cls, _c) in animal_labels})
        n_static = sum(1 for (_b, _k, _t, _c) in suspect if _k == "static")
        n_transient = sum(1 for (_b, _k, _t, _c) in suspect if _k == "transient")

        frames.append({
            "raw_idx": meta["raw"], "width": meta["W"], "height": meta["H"],
            "n_intpfn": len(intp_fn), "n_weakfn": len(weak_fn),
            "n_anchor_start": len(a_start), "n_anchor_end": len(a_end),
            "n_fp": n_transient, "n_sfp": n_static, "n_context": len(ctx),
            "n_person": len(person_labels), "animals": animal_names,
            "persons": person_labels, "animal_boxes": animal_labels, "suspect": suspect,
        })

    # ---- Select: independent per-category quotas (see _select_frames) ----
    selected, n = _select_frames(frames, cfg)
    if not selected:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return [], frames

    # ---- Read the selected frames back from the pass-1 cache (no re-decode:
    # this is the exact array _detect scored for each of these steps) ----
    out = []
    for fr in selected:
        img = cv2.imread(_cache_path(fr["raw_idx"]))
        if img is not None:
            fr["image"] = img
            out.append(fr)
    if len(out) < len(selected):
        # A frame missing from its own clip's just-written cache would be a
        # cv2.imwrite failure in pass 1 (disk full, permissions) rather than
        # anything about the video itself — worth surfacing loudly either way,
        # since it silently loses a mined candidate.
        logger.warning(f"      {len(selected) - len(out)}/{len(selected)} selected "
                       f"frame(s) missing from the pass-1 cache, dropped")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f"      mined: {n['Intp_FN']} intpFN / {n['Weak_FN']} weakFN / "
                f"{n['Anchor_start']}+{n['Anchor_end']} anchorS/E / "
                f"{n['SFP']} SFP / {n['FP']} FP / {n['easy']} easy")
    return out, frames


# ---------------------------------------------------------------------------
# Annotated-clip rendering (verification aid)
# ---------------------------------------------------------------------------
# Colors are BGR. The overlay shows EXACTLY what mine_clip decided, so watching
# the annotated video against the original validates the mining:
#   green   = production detection (conf >= conf_thresh) kept as a person
#   red     = a tracked person the production view MISSED (weak/interp) -> the FN
#   yellow  = transient false-positive candidate (SUSPECT_FP)
#   magenta = static human-like FP candidate (SUSPECT_STATIC_FP)
#   orange  = animal detection
_COLORS = {
    "strong": (0, 255, 0), "miss": (0, 0, 255),
    "transient": (0, 255, 255), "static": (255, 0, 255), "animal": (0, 165, 255),
}


def _draw_box(img, box, color, label):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(14, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _draw_annotations(img, fr):
    for (box, src, tid, cats, conf) in fr["persons"]:
        tag = f"T{tid}" + ("/" + "+".join(cats) if cats else "")
        conf_sfx = f" {conf:.2f}" if conf is not None else ""
        if src == "strong":
            _draw_box(img, box, _COLORS["strong"], f"{tag} person{conf_sfx}")
        else:  # weak / interp -> production missed it (this is the FN)
            _draw_box(img, box, _COLORS["miss"], f"{tag} MISS/{src}{conf_sfx}")
    for (box, cls, conf) in fr["animal_boxes"]:
        _draw_box(img, box, _COLORS["animal"], f"{TRAINSET_NAMES[cls]} {conf:.2f}")
    for (box, kind, tid, conf) in fr["suspect"]:
        conf_sfx = f" {conf:.2f}" if conf is not None else ""
        label = f"T{tid} " + ("STATIC-FP?" if kind == "static" else "FP?") + conf_sfx
        _draw_box(img, box, _COLORS[kind], label)
    cats = "+".join(sorted(frame_categories(fr)))
    banner = (f"{cats}  iFN:{fr['n_intpfn']} wFN:{fr['n_weakfn']} "
              f"As:{fr['n_anchor_start']} Ae:{fr['n_anchor_end']} "
              f"FP:{fr['n_fp']} SFP:{fr['n_sfp']}")
    cv2.putText(img, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)


def render_annotated_clip(clip_path, frames, out_path, stride):
    """Write an annotated MP4 of the processed frames, drawing mine_clip's boxes
    and per-frame verdict. Output fps = source_fps / stride so timing roughly
    matches the original. Returns True on success."""
    if not frames:
        return False
    ann = {fr["raw_idx"]: fr for fr in frames}
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return False
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = max(1.0, src_fps / max(1, int(stride)))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
    raw_idx = -1
    written = 0
    try:
        while True:
            if not cap.grab():
                break
            raw_idx += 1
            fr = ann.get(raw_idx)
            if fr is None:
                continue                 # only render the processed frames
            ok, img = cap.retrieve()
            if not ok:
                break
            _draw_annotations(img, fr)
            writer.write(img)
            written += 1
    finally:
        writer.release()
        cap.release()
    return written > 0
