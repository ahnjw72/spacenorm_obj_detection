"""offline_entrypoint.py

Container entrypoint for the offline batch detection service.

Mirrors the logic of entrypoint.py:
  1. Compute SHA256 of the .pt model file
  2. Derive a unique engine filename
  3. Build the TensorRT engine if not cached; reuse if it exists
  4. exec() offline_spacenorm_obj_detection, passing --model <engine_path>

Because both services bind-mount the same /var/lib/spacenorm_obj_detection
directory, the engine built by the main service entrypoint is reused here
automatically — no duplicate build needed when both run on the same node.

Environment variables (all optional — defaults match stack.yml):
  SPACENORM_MODEL_PT         Path to the source .pt file inside the container
  SPACENORM_ENGINE_CACHE_DIR Host-mounted directory for caching engine files
  SPACENORM_IMGSZ            Image size used when building the engine (default: 640)
  SPACENORM_TRT_HALF         FP16 mode — 'true' or 'false'  (default: true)
  SPACENORM_DEVICE           GPU device index  (default: 0)
  SPACENORM_TRT_WORKSPACE    TensorRT workspace in GiB  (default: 4)
  SPACENORM_OFFLINE_CFG      Path to offline_spacenorm_obj_detection.json inside
                             the container (passed as --common_config)
"""

import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables
# ---------------------------------------------------------------------------
PT_MODEL_PATH    = os.environ.get('SPACENORM_MODEL_PT',
                       '/app/spacenorm_obj_detection/yolo11x_set01-0148.pt')
ENGINE_CACHE_DIR = os.environ.get('SPACENORM_ENGINE_CACHE_DIR',
                       '/var/lib/spacenorm_obj_detection')
IMGSZ            = int(os.environ.get('SPACENORM_IMGSZ', '640'))
HALF             = os.environ.get('SPACENORM_TRT_HALF', 'true').lower() == 'true'
DEVICE           = int(os.environ.get('SPACENORM_DEVICE', '0'))
WORKSPACE        = int(os.environ.get('SPACENORM_TRT_WORKSPACE', '4'))
OFFLINE_CFG      = os.environ.get('SPACENORM_OFFLINE_CFG',
                       '/app/spacenorm_obj_detection/config_files/common_cfg/'
                       'offline_spacenorm_obj_detection.json')


# ---------------------------------------------------------------------------
# Helpers  (mirrors entrypoint.py)
# ---------------------------------------------------------------------------

def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def engine_filename(pt_path: str, imgsz: int, half: bool) -> str:
    stem      = Path(pt_path).stem
    digest    = sha256_file(pt_path)[:12]
    precision = 'fp16' if half else 'fp32'
    return f"{stem}_{digest}_imgsz{imgsz}_{precision}.engine"


def export_engine(pt_path: str, engine_path: str,
                  imgsz: int, half: bool, device: int, workspace: int) -> None:
    print(f"[offline_entrypoint] No cached engine found — building TensorRT engine.")
    print(f"[offline_entrypoint] This takes ~2-5 minutes on the first run for this node.")
    print(f"[offline_entrypoint]   source    : {pt_path}")
    print(f"[offline_entrypoint]   output    : {engine_path}")
    print(f"[offline_entrypoint]   imgsz     : {imgsz}")
    print(f"[offline_entrypoint]   precision : {'FP16' if half else 'FP32'}")
    print(f"[offline_entrypoint]   workspace : {workspace} GiB")
    print(flush=True)

    from ultralytics.utils import SETTINGS
    SETTINGS.update({'sync': False})
    from ultralytics import YOLO

    # Ultralytics writes intermediate files (.onnx, .engine) next to the source
    # .pt file.  The container runs as user 7000:7000 and /app/ is root-owned,
    # so export would fail with EACCES.  Copy the .pt to a temp directory first
    # so all intermediate files land in /tmp (world-writable), then move only
    # the final .engine to the cache dir (owned by 7000:7000).
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pt = os.path.join(tmp_dir, Path(pt_path).name)
        print(f"[offline_entrypoint] Copying .pt to temp dir: {tmp_pt}", flush=True)
        shutil.copy2(pt_path, tmp_pt)

        model = YOLO(tmp_pt)
        exported = model.export(
            format='engine',
            imgsz=imgsz,
            device=device,
            half=half,
            workspace=workspace,
            verbose=True,
        )

        # shutil.move handles cross-device moves (/tmp → bind-mount)
        exported_path = Path(str(exported))
        target_path   = Path(engine_path)
        if exported_path.resolve() != target_path.resolve():
            shutil.move(str(exported_path), str(target_path))

    print(f"[offline_entrypoint] Engine cached at: {engine_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pt_path = PT_MODEL_PATH

    if not Path(pt_path).exists():
        print(f"[offline_entrypoint] ERROR: .pt model not found: {pt_path}", file=sys.stderr)
        sys.exit(1)

    cache_dir = Path(ENGINE_CACHE_DIR)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"[offline_entrypoint] ERROR: cannot create cache dir {cache_dir} — "
              f"check that the host directory exists and is writable.", file=sys.stderr)
        sys.exit(1)

    fname       = engine_filename(pt_path, IMGSZ, HALF)
    engine_path = str(cache_dir / fname)

    if Path(engine_path).exists():
        print(f"[offline_entrypoint] Cached engine found: {engine_path}", flush=True)
    else:
        export_engine(pt_path, engine_path, IMGSZ, HALF, DEVICE, WORKSPACE)

    cmd = [
        sys.executable, '-u', '-m',
        'spacenorm_obj_detection.offline_spacenorm_obj_detection',
        '--common_config', OFFLINE_CFG,
        '--model', engine_path,
    ]
    print(f"[offline_entrypoint] Launching: {' '.join(cmd)}", flush=True)
    os.execv(sys.executable, cmd)


if __name__ == '__main__':
    main()
