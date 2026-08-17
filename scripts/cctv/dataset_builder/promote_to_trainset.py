#!/usr/bin/env python3
"""promote_to_trainset.py — fold reviewed data into the training set.

Run this on a Label-Studio-corrected YOLO export (the intended path), or on a
hand-reviewed staging folder. NEVER on raw, unreviewed pre-labels. It:

  1. Pairs each exported label .txt with its image BY FILENAME. Label Studio omits
     image files for Local-Storage tasks (the export's images/ is empty), so pass
     --images <reviewing dir> to pair the corrected labels with the original PNGs
     (mine_dataset stages losslessly, ALGORITHM.md 3 — never JPEGs).
  2. Skips any frame whose basename is already present in an existing setNNNN, so
     two Label Studio exports with overlapping task ranges cannot copy the same
     frame into two sets (which would double its training weight and let a stale
     copy contradict a corrected one). --allow-duplicates overrides.
  3. Copies them flat into a new data/cctv_train_data/setNNNN/ folder (next unused
     4-digit set number, unless --set-name is given), remapping class ids to the
     trainset taxonomy BY NAME via the export's classes.txt (Label Studio numbers
     export classes alphabetically, not person(0)..cow(6)), and dropping review-only
     (SUSPECT_*) and any non-trainset classes. Unparseable label lines are dropped
     with a loud warning — never copied through, since a corrupt line in a YOLO
     label file either crashes the trainer or silently misreads the file.

It ONLY creates the setNNNN folder(s). It does NOT touch data/cctv.yaml or any
train/val list — regenerate those with make_train_test_single_multi.ipynb after
promotion (that notebook owns the train/test split).

--staging accepts one or more paths, each either a hand-reviewed directory or
a raw Label Studio YOLO-export .zip (extracted to a temp dir automatically).
--ls-project accepts one or more Label Studio project ids (or pasted project
URLs) and fetches that project's YOLO export directly over the API instead —
no manual "Export -> download zip -> copy it here" round trip. Either kind of
item becomes its own new setNNNN, and --staging/--ls-project items can be
mixed in one run.

    python3 promote_to_trainset.py --staging <yolo-export-dir> \\
        --images data/cctv_train_data_mining/reviewing
    python3 promote_to_trainset.py --staging <dir> --set-name set0200
    python3 promote_to_trainset.py --staging a.zip b.zip \\
        --images data/cctv_train_data_mining/reviewing
    LABEL_STUDIO_API_TOKEN=... python3 promote_to_trainset.py --ls-project 42 \\
        --images data/cctv_train_data_mining/reviewing
"""
import argparse
import io
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ls_api import add_auth_args, ls_client_from_args  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAINSET_DIR = _REPO_ROOT / "data" / "cctv_train_data"

# The trainset taxonomy (data/cctv.yaml order). Label Studio numbers YOLO-export
# classes by classes.txt order, which it sorts ALPHABETICALLY (e.g. SUSPECT_FP=0,
# bird=2, person=7, ...) — NOT this order. So we remap the exported ids to these
# BY NAME via the export's classes.txt, dropping review-only/unknown names.
TRAINSET_NAMES = ["person", "bird", "cat", "dog", "horse", "sheep", "cow"]
TRAINSET_ID = {name: i for i, name in enumerate(TRAINSET_NAMES)}


def _resolve(path):
    """Resolve a CLI path predictably regardless of CWD: use it as given (absolute
    or CWD-relative) if it exists; otherwise fall back to repo-root-relative, so the
    README's `data/cctv_train_data_mining/reviewing` form works from any directory."""
    if os.path.isabs(path):
        return path
    cwd_rel = os.path.abspath(path)
    if os.path.exists(cwd_rel):
        return cwd_rel
    repo_rel = os.path.join(str(_REPO_ROOT), path)
    return repo_rel if os.path.exists(repo_rel) else cwd_rel


def find_pairs(export_dir, images_root=None):
    """Pair each exported label .txt with its image, matched BY FILENAME.

    Label Studio omits the image FILES when tasks are served from Local Storage
    (the export's images/ is empty), so images are indexed from both the export
    dir and the optional images_root (the reviewing tree, which holds the original
    PNGs under the same descriptive basenames). `classes.txt` is ignored.

    Returns (sorted [(png, txt), ...], n_label_files_seen)."""
    index = {}   # basename.png -> path (first occurrence wins)
    for root in [export_dir] + ([images_root] if images_root else []):
        for r, _dirs, files in os.walk(root):
            for f in files:
                if f.lower().endswith(".png"):
                    index.setdefault(f, os.path.join(r, f))

    pairs, n_labels = [], 0
    for r, _dirs, files in os.walk(export_dir):
        for f in files:
            if not f.lower().endswith(".txt") or f == "classes.txt":
                continue
            n_labels += 1
            png = index.get(os.path.splitext(f)[0] + ".png")
            if png:
                pairs.append((png, os.path.join(r, f)))
            else:
                print(f"  [skip] no image found for label {f}")
    return sorted(pairs), n_labels


def _existing_set_dirs():
    """Every setNNNN directory already under the trainset root (may be empty)."""
    if not _TRAINSET_DIR.is_dir():
        return []
    return [_TRAINSET_DIR / name for name in sorted(os.listdir(_TRAINSET_DIR))
            if re.match(r"^set(\d+)", name) and (_TRAINSET_DIR / name).is_dir()]


def next_set_name():
    """Next unused setNNNN. Fails with an explanation rather than a bare
    FileNotFoundError when the trainset root does not exist at all."""
    if not _TRAINSET_DIR.is_dir():
        raise SystemExit(
            f"trainset root not found: {_TRAINSET_DIR}\n"
            "Create it first, or pass --set-name to choose the target folder "
            "explicitly. Refusing to guess a set number without seeing the "
            "existing sets, since a colliding number would silently mix two "
            "collections.")
    nums = [int(re.match(r"^set(\d+)", d.name).group(1)) for d in _existing_set_dirs()]
    nxt = (max(nums) + 1) if nums else 1
    return f"set{nxt:04d}"


def already_promoted():
    """Map image basename -> the setNNNN that already contains it.

    Promotion is keyed on the image basename, which mine_dataset makes unique and
    deterministic (nvr_ch_bucket_stamp_rawidx). Two Label Studio exports whose task
    ranges overlap — easy to produce, since one LS project accumulates every sweep
    and "Export" covers the whole project unless filtered — would otherwise copy the
    same frame into two different sets. Both copies then land in the training list,
    so those frames get double weight and, worse, a frame corrected in the later
    review is contradicted by its stale twin. Nothing downstream detects this:
    count_files.py checks jpg/txt pairing within a set (and doesn't know about
    .png at all yet), not identity across sets."""
    seen = {}
    for d in _existing_set_dirs():
        for name in os.listdir(d):
            if name.lower().endswith(".png"):
                seen.setdefault(name, d.name)
    return seen


def load_class_remap(staging):
    """Find a Label Studio YOLO-export classes.txt and map its positional ids to
    trainset ids BY NAME (LS orders classes.txt alphabetically, not by our
    taxonomy). Review-only/unknown names map to None (their boxes are dropped).

    Returns (remap|None, names|None, path|None). None when no classes.txt exists —
    then labels are assumed to already use trainset ids (a hand-reviewed staging
    folder written by mine_dataset)."""
    for root, _dirs, files in os.walk(staging):
        if "classes.txt" in files:
            path = os.path.join(root, "classes.txt")
            with open(path, "r", encoding="utf-8") as f:
                names = [ln.strip() for ln in f.read().splitlines()]
            remap = {i: TRAINSET_ID.get(nm) for i, nm in enumerate(names)}
            return remap, names, path
    return None, None, None


def _remap_lines(src_txt, remap):
    """Return (kept_lines, dropped_count) for a YOLO label file remapped to the
    trainset taxonomy. With `remap` (from a LS classes.txt) each line's class id is
    remapped BY NAME; ids mapping to None (SUSPECT_* / unknown) are dropped. Without
    `remap`, labels are assumed to be trainset ids already and only out-of-range ids
    (>= len(TRAINSET_NAMES)) are dropped defensively."""
    kept, dropped = [], 0
    malformed = []
    with open(src_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            try:
                cid = int(parts[0])
            except (ValueError, IndexError):
                # DROP, do not keep. A line whose first field is not an integer is
                # not a YOLO label, so copying it through writes a corrupt label
                # file into the training set — where the trainer either crashes or
                # silently misreads the whole file. "Keep it rather than lose data"
                # was the wrong call: there is no recoverable data in a line we
                # cannot parse, and the label is reproducible from the LS export.
                malformed.append(s)
                continue
            if len(parts) != 5:
                # A detection label is exactly "cls xc yc w h". A different field
                # count means a truncated write or a different task format
                # (segmentation polygons); either way it is not usable here.
                malformed.append(s)
                continue
            if remap is not None:
                new = remap.get(cid)
                if new is None:
                    dropped += 1
                    continue
                parts[0] = str(new)
                kept.append(" ".join(parts))
            else:
                if cid >= len(TRAINSET_NAMES):
                    dropped += 1
                    continue
                kept.append(s)
    return kept, dropped, malformed


def copy_label_filtered(src_txt, dst_txt, remap):
    """Write the remapped label file to dst_txt. An empty result is written as an
    empty file (a valid background/negative label).

    Returns (n_dropped_by_class, [malformed lines])."""
    kept, dropped, malformed = _remap_lines(src_txt, remap)
    with open(dst_txt, "w", encoding="utf-8") as f:
        if kept:
            f.write("\n".join(kept) + "\n")
    return dropped, malformed


def _is_zip(path):
    """True for a path that names an actual zip file (not just a .zip extension
    on something else)."""
    return os.path.isfile(path) and zipfile.is_zipfile(path)


def parse_project_id(spec):
    """Accept either a bare project id ('42') or a URL/path copy-pasted from
    the browser (e.g. 'http://host:8080/projects/42/data') -- so --ls-project
    works with whatever's on the clipboard, not just the bare number."""
    m = re.search(r"/projects/(\d+)", spec)
    if m:
        return int(m.group(1))
    try:
        return int(spec)
    except ValueError:
        raise SystemExit(f"--ls-project: {spec!r} is not a project id or a "
                         "Label Studio project URL")


def fetch_yolo_export(client, project_id):
    """Fetch a project's YOLO export as zip bytes via the API -- the same
    format and content 'Export -> YOLO' produces in the UI (labels/ +
    classes.txt; images/ is empty for Local-Storage-served tasks, same as a
    manual export, since LS never stored the image bytes -- --images is
    still required). Only ANNOTATED tasks are included (download_all_tasks
    defaults to false), matching the "never on raw, unreviewed pre-labels"
    rule this script exists to enforce."""
    resp = client.request("GET", f"/api/projects/{project_id}/export",
                          params={"exportType": "YOLO"}, timeout=300)
    if not resp.ok:
        # A wrong project id 404s as an HTML error page, not JSON -- show that
        # plainly instead of dumping a page of markup.
        detail = (resp.text[:300] if "json" in resp.headers.get("content-type", "")
                 else "(non-JSON error page; likely a wrong --ls-project id or --url)")
        raise SystemExit(f"error: export failed for LS project {project_id}: "
                         f"{resp.status_code} {detail}")
    return resp.content


def promote_one(staging, images_root, set_name, allow_duplicates, dry_run, label):
    """Promote a single staging dir into one new setNNNN.

    `staging` is an already-resolved directory (for a zip staging item, the
    directory it was extracted into). `label` is the original --staging
    argument as the user typed it, used only in messages so batch output says
    which item a message is about.

    Returns True if a set was written (or would be, under --dry-run), False
    if this item had nothing new to promote — the caller decides whether that
    aborts the batch or just gets skipped."""
    pairs, n_labels = find_pairs(staging, images_root)
    if not pairs:
        if n_labels and not images_root:
            print(f"  [skip] {label}: found {n_labels} label file(s) but no matching "
                  f"images under {staging}.\n"
                  "  Label Studio omits image files for Local-Storage tasks, so the export's "
                  "images/ is empty.\n  Re-run with --images pointing at the reviewing tree, "
                  "e.g.:\n    python3 promote_to_trainset.py --staging {export} "
                  "--images data/cctv_train_data_mining/reviewing".format(export=label))
        else:
            print(f"  [skip] {label}: no label/image pairs found under {staging}"
                  + (f" (+ images from {images_root})" if images_root else ""))
        return False

    # Determine how exported class ids map to the trainset taxonomy.
    remap, names, classes_path = load_class_remap(staging)
    if remap is not None:
        print(f"  [classes] remapping by name via {classes_path}:")
        for i, nm in enumerate(names):
            t = TRAINSET_ID.get(nm)
            print(f"      export {i} '{nm}' -> " + ("DROP (review-only/unknown)" if t is None else f"trainset {t}"))
    else:
        print("  [classes] no classes.txt found; assuming labels already use "
              "trainset ids person(0)..cow(6)")

    # Refuse to copy a frame that some earlier promotion already placed in a set.
    promoted = already_promoted()
    dups = [(j, t) for (j, t) in pairs if os.path.basename(j) in promoted]
    if dups:
        where = sorted({promoted[os.path.basename(j)] for (j, _t) in dups})
        print(f"  [dup] {len(dups)} of {len(pairs)} pair(s) are already in "
              f"{', '.join(where)}")
        for j, _t in dups[:5]:
            print(f"      {os.path.basename(j)} -> {promoted[os.path.basename(j)]}")
        if len(dups) > 5:
            print(f"      ... and {len(dups) - 5} more")
        if allow_duplicates:
            print("  [dup] --allow-duplicates given: copying them anyway")
        else:
            pairs = [(j, t) for (j, t) in pairs if os.path.basename(j) not in promoted]
            print(f"  [dup] skipping them ({len(pairs)} pair(s) left); "
                  f"--allow-duplicates to override")
            if not pairs:
                print(f"  [skip] {label}: every pair was already promoted; "
                      "nothing new to write")
                return False

    set_name = set_name or next_set_name()
    set_dir = _TRAINSET_DIR / set_name
    # pos/neg measured AFTER the remap/drop: a frame whose only labels were SUSPECT_*
    # becomes an (empty) negative, so count survivors, not source file size.
    pos = sum(1 for _j, t in pairs if _remap_lines(t, remap)[0])
    print(f"Promoting {len(pairs)} pairs ({pos} pos / {len(pairs) - pos} neg after remap) "
          f"-> {set_dir}")
    if dry_run:
        print("(dry-run) nothing written")
        return True

    set_dir.mkdir(parents=True, exist_ok=False)

    dropped_total = 0
    malformed_total = []
    for png, txt in pairs:
        base = os.path.basename(png)
        shutil.copy2(png, set_dir / base)
        dropped, malformed = copy_label_filtered(
            txt, set_dir / (os.path.splitext(base)[0] + ".txt"), remap)
        dropped_total += dropped
        malformed_total += [(os.path.basename(txt), ln) for ln in malformed]

    print(f"  wrote {len(pairs)} pairs to {set_dir}")
    if dropped_total:
        print(f"  [filter] dropped {dropped_total} review-only/unknown label line(s) "
              f"(SUSPECT_FP/SUSPECT_STATIC_FP and any non-trainset class)")
    if malformed_total:
        # Loud, not silent: an unparseable label line means the export is not what
        # this script expects, and the affected frames are now labelled with FEWER
        # boxes than the reviewer drew — which trains the model that a real person
        # is background. Worth investigating before training on this set.
        print(f"  [WARNING] dropped {len(malformed_total)} unparseable label line(s) "
              f"in {len({f for f, _ in malformed_total})} file(s). Those frames lost "
              f"boxes; inspect the export before training on {set_name}:")
        for fname, ln in malformed_total[:5]:
            print(f"      {fname}: {ln!r}")
        if len(malformed_total) > 5:
            print(f"      ... and {len(malformed_total) - 5} more")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staging", nargs="+", default=[],
                    help="One or more Label Studio YOLO export .zip files and/or "
                         "hand-reviewed folders to promote. Each item becomes its own "
                         "new setNNNN.")
    ap.add_argument("--ls-project", nargs="+", default=[], metavar="PROJECT",
                    help="One or more Label Studio project ids (or pasted project URLs) "
                         "to fetch a YOLO export from directly over the API, instead of "
                         "exporting from the UI and passing a downloaded --staging zip. "
                         "Only annotated (reviewed) tasks are included, same as the UI's "
                         "Export button. Each project becomes its own new setNNNN, same "
                         "as a --staging item. Requires --url/--token (or "
                         "$LABEL_STUDIO_URL/$LABEL_STUDIO_API_TOKEN).")
    ap.add_argument("--images", help="Directory tree with the original PNGs (the reviewing "
                    "dir). Required when the LS export has an empty images/ (Local-Storage "
                    "tasks); labels are paired to images by filename. Shared across all "
                    "--staging/--ls-project items.")
    ap.add_argument("--set-name", help="Target set folder name (default: next setNNNN). "
                    "Only valid with a single --staging/--ls-project item total.")
    ap.add_argument("--dry-run", action="store_true", help="Report only; copy nothing")
    ap.add_argument("--allow-duplicates", action="store_true",
                    help="Promote frames whose basename already exists in an earlier "
                         "setNNNN (default: skip them, so overlapping Label Studio "
                         "exports cannot double-weight the same frame)")
    add_auth_args(ap)
    args = ap.parse_args()

    if not args.staging and not args.ls_project:
        raise SystemExit("error: pass at least one --staging item or --ls-project id")

    project_ids = [parse_project_id(p) for p in args.ls_project]
    if args.set_name and (len(args.staging) + len(project_ids)) > 1:
        raise SystemExit("--set-name can only be used with a single --staging/--ls-project "
                          "item total (each item promotes into its own set; use the "
                          "default auto-numbered setNNNN when batching multiple).")

    images_root = _resolve(args.images) if args.images else None
    if images_root and not os.path.isdir(images_root):
        raise SystemExit(f"--images dir not found: {args.images}")

    resolved = [_resolve(s) for s in args.staging]
    missing = [orig for orig, res in zip(args.staging, resolved) if not os.path.exists(res)]
    if missing:
        raise SystemExit("--staging path(s) not found: " + ", ".join(missing))

    client = ls_client_from_args(args) if project_ids else None

    n = len(resolved) + len(project_ids)
    written, skipped = 0, 0
    i = 0
    for label, path in zip(args.staging, resolved):
        i += 1
        print(f"\n=== [{i}/{n}] {label} ===")
        if _is_zip(path):
            with tempfile.TemporaryDirectory(prefix="ls_export_") as tmpdir:
                with zipfile.ZipFile(path) as zf:
                    zf.extractall(tmpdir)
                ok = promote_one(tmpdir, images_root, args.set_name, args.allow_duplicates,
                                  args.dry_run, label)
        elif os.path.isdir(path):
            ok = promote_one(path, images_root, args.set_name, args.allow_duplicates,
                              args.dry_run, label)
        else:
            print(f"  [skip] {label}: not a directory or a zip file")
            ok = False
        written += ok
        skipped += not ok

    for spec, pid in zip(args.ls_project, project_ids):
        i += 1
        label = f"LS project {pid}" + (f" ({spec})" if spec != str(pid) else "")
        print(f"\n=== [{i}/{n}] {label} ===")
        content = fetch_yolo_export(client, pid)
        with tempfile.TemporaryDirectory(prefix="ls_export_") as tmpdir:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                zf.extractall(tmpdir)
            ok = promote_one(tmpdir, images_root, args.set_name, args.allow_duplicates,
                              args.dry_run, label)
        written += ok
        skipped += not ok

    print(f"\n{written} of {n} staging item(s) promoted"
          + (f", {skipped} skipped" if skipped else "") + ".")
    if written:
        print("\nNext steps:")
        print("  - Regenerate the train/val lists (train_01_to_NNNN.txt / test_01_to_NNNN.txt) "
              "via make_train_test_single_multi.ipynb so data/cctv.yaml picks up the new set(s).")
        print("  - Add a line for each new setNNNN to data/cctv_train_data/.NOTE.")
    if skipped:
        sys.exit(1)


if __name__ == "__main__":
    main()
