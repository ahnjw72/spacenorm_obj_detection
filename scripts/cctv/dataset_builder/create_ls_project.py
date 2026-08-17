#!/usr/bin/env python3
"""create_ls_project.py — create a Label Studio project via the API from one or
more mined tasks_*.json files, skipping the manual "paste XML into Labeling
Setup" step (and, unless --no-storage, the manual Local Storage step too).

Aggregates all input files' tasks into ONE new project, created with this
project's labeling config already baked in, registers the staging_dir(s) the
input file(s) live under as that project's Local Storage (so images resolve
with no manual Settings work), and imports every task. Label Studio itself
must already be running with local-file serving enabled
(LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED / _DOCUMENT_ROOT — e.g. via the
run_labelstudio.sh a sweep writes) before this script is run: it only
registers this ONE project's storage entry, not the server-wide serving
flags. See README.md "Review in Label Studio -> export YOLO".

Auth: an API token, from --token or $LABEL_STUDIO_API_TOKEN. By default this
is treated as a Personal Access Token — get one from the Label Studio UI's
Account & Settings page. That page hands you a long-lived REFRESH token, not
something usable directly: this script exchanges it once for a short-lived
ACCESS token via POST /api/token/refresh/ and sends THAT as
'Authorization: Bearer <access>', re-exchanging automatically if a call 401s
mid-run (access tokens default to a 5-minute lifetime). If your server has
legacy tokens enabled instead (LS 1.23 default: disabled — see README
"Troubleshooting"), pass --legacy-token to send your token as-is via
'Authorization: Token <token>' with no exchange.

Usage:
    export LABEL_STUDIO_API_TOKEN=...   # Personal Access Token, from Account & Settings
    python3 create_ls_project.py tasks_20260807T155400_ch00.json
    python3 create_ls_project.py tasks_..._ch00.json tasks_..._ch01.json \\
        --name "cheilacc-ansung sweep 0807"
    python3 create_ls_project.py tasks_..._ch00.json --legacy-token --no-storage
"""
import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from labelstudio_export import LABEL_CONFIG  # noqa: E402
from ls_api import add_auth_args, ls_client_from_args  # noqa: E402


def load_tasks(paths):
    combined = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise SystemExit(f"error: {p} is not a JSON list of tasks")
        combined.extend(data)
    return combined


def infer_staging_dir(json_path):
    """<staging_dir>/<nvr>/labelstudio/tasks_*.json -> <staging_dir>, per the
    fixed layout mine_dataset.py writes (README "Reviewing output"). Returns
    None if the file isn't in that layout (e.g. moved elsewhere), so the
    caller can fall back to --staging-dir instead of registering a bogus path."""
    p = Path(json_path).resolve()
    if p.parent.name != "labelstudio":
        return None
    return p.parents[2]  # labelstudio/ -> <nvr>/ -> <staging_dir>


def create_project(client, title, label_config):
    resp = client.request("POST", "/api/projects/",
                          json={"title": title, "label_config": label_config}, timeout=30)
    if not resp.ok:
        raise SystemExit(f"error: project creation failed: {resp.status_code} {resp.text[:500]}")
    try:
        return resp.json()["id"]
    except (ValueError, KeyError):
        raise SystemExit(f"error: unexpected project-creation response: {resp.text[:500]}")


def register_storage(client, project_id, staging_dir):
    resp = client.request("POST", "/api/storages/localfiles/",
                          json={"project": project_id, "path": str(staging_dir),
                                "title": staging_dir.name, "use_blob_urls": False},
                          timeout=30)
    if not resp.ok:
        print(f"warning: could not register Local Storage for {staging_dir}: "
              f"{resp.status_code} {resp.text[:500]}\n"
              f"  register it manually instead: Settings -> Cloud Storage -> "
              f"Add Source Storage -> Local files -> {staging_dir}", file=sys.stderr)
        return False
    return True


def import_tasks(client, project_id, tasks, batch_size):
    imported = 0
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        resp = client.request("POST", f"/api/projects/{project_id}/import", json=batch, timeout=120)
        if not resp.ok:
            raise SystemExit(f"error: task import failed on tasks {i}-{i + len(batch)}: "
                             f"{resp.status_code} {resp.text[:500]} "
                             f"({imported} task(s) already imported before this failure)")
        imported += len(batch)
        print(f"  imported {imported}/{len(tasks)} task(s)", file=sys.stderr)
    return imported


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks_json", nargs="+", metavar="TASKS_JSON",
                    help="One or more tasks_*.json files to aggregate into one project")
    ap.add_argument("--name", help="Project title (required with >1 input file; "
                    "defaults to the single input file's name, minus .json)")
    add_auth_args(ap)
    ap.add_argument("--label-config", help="Path to a custom label config XML "
                    "(default: this project's built-in LABEL_CONFIG)")
    ap.add_argument("--staging-dir", help="Override the inferred staging_dir(s) used for "
                    "Local Storage registration (needed if a tasks_*.json was moved out "
                    "of its <staging_dir>/<nvr>/labelstudio/ layout)")
    ap.add_argument("--no-storage", action="store_true",
                    help="Skip auto-registering Local Storage for the new project")
    ap.add_argument("--batch-size", type=int, default=500,
                    help="Tasks per /import request (default: 500)")
    args = ap.parse_args()

    if len(args.tasks_json) > 1 and not args.name:
        raise SystemExit("error: --name is required when aggregating more than one tasks_*.json file")
    name = args.name or Path(args.tasks_json[0]).stem

    tasks = load_tasks(args.tasks_json)
    if not tasks:
        raise SystemExit("error: input file(s) contained zero tasks; nothing to create")

    label_config = LABEL_CONFIG
    if args.label_config:
        label_config = Path(args.label_config).read_text(encoding="utf-8")

    client = ls_client_from_args(args)
    base_url = client.base_url

    project_id = create_project(client, name, label_config)
    print(f"created project {name!r} (id={project_id}) -> {base_url}/projects/{project_id}")

    if not args.no_storage:
        if args.staging_dir:
            staging_dirs = {Path(args.staging_dir).resolve()}
        else:
            staging_dirs, unresolved = set(), []
            for p in args.tasks_json:
                sd = infer_staging_dir(p)
                (staging_dirs.add(sd) if sd else unresolved.append(p))
            if unresolved:
                print(f"warning: could not infer staging_dir for: {', '.join(unresolved)} "
                      f"(expected .../<staging_dir>/<nvr>/labelstudio/tasks_*.json); "
                      f"pass --staging-dir to register storage for these", file=sys.stderr)
        for sd in staging_dirs:
            if register_storage(client, project_id, sd):
                print(f"registered Local Storage: {sd}")

    n = import_tasks(client, project_id, tasks, args.batch_size)
    print(f"imported {n} task(s) from {len(args.tasks_json)} file(s) into project {project_id}")


if __name__ == "__main__":
    main()
