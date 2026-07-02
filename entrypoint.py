"""entrypoint.py

Container entrypoint for spacenorm_obj_detection.

On each container start:
  1. Compute SHA256 of the .pt model file
  2. Derive a unique engine filename
       <stem>_<sha256[:12]>_imgsz<N>_<precision>_trt<version>.engine
     The TRT version is encoded in the filename so that a TRT software upgrade
     automatically invalidates the cached engine and triggers a rebuild.
  3. If that engine file does not exist in ENGINE_CACHE_DIR → export .pt → .engine
     (takes ~2-5 minutes on first run per node; subsequent starts are instant)
  4. Remove any stale engines for the same model (different TRT version or params).
  5. exec() the main service, passing the engine path via --model

Environment variables (all optional — defaults match default.json / stack.yml):
  SPACENORM_MODEL_PT         Path to the source .pt file inside the container
                             (default: /app/spacenorm_obj_detection/yolo11x_set01-0148.pt)
  SPACENORM_ENGINE_CACHE_DIR Host-mounted directory for caching engine files
                             (default: /var/lib/spacenorm_obj_detection)
  SPACENORM_IMGSZ            Inference image size used when building the engine
                             (default: 640)
  SPACENORM_TRT_HALF         FP16 mode — 'true' or 'false'  (default: true)
  SPACENORM_DEVICE           GPU device index  (default: 0)
  SPACENORM_TRT_WORKSPACE    TensorRT workspace in GiB  (default: 4)
"""

import hashlib
import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables
# ---------------------------------------------------------------------------
PT_MODEL_PATH     = os.environ.get('SPACENORM_MODEL_PT',
                        '/app/spacenorm_obj_detection/yolo11x_set01-0148.pt')
ENGINE_CACHE_DIR  = os.environ.get('SPACENORM_ENGINE_CACHE_DIR',
                        '/var/lib/spacenorm_obj_detection')
IMGSZ             = int(os.environ.get('SPACENORM_IMGSZ', '640'))
HALF              = os.environ.get('SPACENORM_TRT_HALF', 'true').lower() == 'true'
DEVICE            = int(os.environ.get('SPACENORM_DEVICE', '0'))
WORKSPACE         = int(os.environ.get('SPACENORM_TRT_WORKSPACE', '4'))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Return the full SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def trt_version_tag() -> str:
    """Return a compact TRT version string safe for use in filenames (e.g. '10030')."""
    try:
        import tensorrt as trt
        return trt.__version__.replace('.', '')
    except ImportError:
        return 'notrt'


def engine_filename(pt_path: str, imgsz: int, half: bool) -> str:
    """
    Build a unique engine filename encoding model identity, build params, and TRT version:
      <stem>_<sha256[:12]>_imgsz<N>_<fp16|fp32>_trt<version>.engine

    Any of these changing (retrained weights, different imgsz, TRT upgrade) produces
    a new filename, causing the old engine to be ignored and a fresh one to be built.
    """
    stem      = Path(pt_path).stem
    digest    = sha256_file(pt_path)[:12]
    precision = 'fp16' if half else 'fp32'
    trt_ver   = trt_version_tag()
    return f"{stem}_{digest}_imgsz{imgsz}_{precision}_trt{trt_ver}.engine"


def cleanup_stale_engines(cache_dir: Path, current_engine: str) -> None:
    """Remove engines that match this container's model+hash+imgsz+precision but have a
    different TRT version — i.e. engines made obsolete by a TRT upgrade on this node.

    Engines with a different hash, imgsz, or precision belong to other containers
    sharing the same cache directory and are left untouched.
    """
    current = Path(current_engine).name
    # Strip _trt<ver>.engine to get the shared prefix for this exact configuration.
    # e.g. "yolo11x_set01-0148_abc123_imgsz640_fp16_trt10030.engine"
    #   -> prefix = "yolo11x_set01-0148_abc123_imgsz640_fp16"
    prefix = current.rsplit('_trt', 1)[0]
    for path in cache_dir.glob(f"{prefix}_trt*.engine"):
        if path.name != current:
            print(f"[entrypoint] Removing stale engine: {path.name}", flush=True)
            path.unlink(missing_ok=True)


def export_engine(pt_path: str, engine_path: str,
                  imgsz: int, half: bool, device: int, workspace: int) -> None:
    """Export a .pt model to a TensorRT .engine file."""
    print(f"[entrypoint] No cached engine found — building TensorRT engine.")
    print(f"[entrypoint] This takes ~2-5 minutes on the first run for this node.")
    print(f"[entrypoint]   source    : {pt_path}")
    print(f"[entrypoint]   output    : {engine_path}")
    print(f"[entrypoint]   imgsz     : {imgsz}")
    print(f"[entrypoint]   precision : {'FP16' if half else 'FP32'}")
    print(f"[entrypoint]   workspace : {workspace} GiB")
    print(flush=True)

    # Import here so the main service path doesn't pay this overhead
    from ultralytics.utils import SETTINGS
    SETTINGS.update({'sync': False})
    from ultralytics import YOLO

    model = YOLO(pt_path)
    exported = model.export(
        format='engine',
        imgsz=imgsz,
        device=device,
        half=half,
        workspace=workspace,
        verbose=True,
    )

    # Ultralytics writes the engine next to the .pt; move it to the cache dir.
    # Use shutil.move() instead of Path.rename() because the source (/app/...)
    # and destination (/var/lib/... bind-mount) are on different filesystems —
    # os.rename() raises EXDEV (errno 18) on cross-device moves.
    exported_path = Path(str(exported))
    target_path   = Path(engine_path)
    if exported_path.resolve() != target_path.resolve():
        shutil.move(str(exported_path), str(target_path))

    print(f"[entrypoint] Engine cached at: {engine_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pt_path = PT_MODEL_PATH

    if not Path(pt_path).exists():
        print(f"[entrypoint] ERROR: .pt model not found: {pt_path}", file=sys.stderr)
        sys.exit(1)

    cache_dir = Path(ENGINE_CACHE_DIR)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"[entrypoint] ERROR: cannot create cache dir {cache_dir} — "
              f"check that the host directory exists and is writable.", file=sys.stderr)
        sys.exit(1)

    fname       = engine_filename(pt_path, IMGSZ, HALF)
    engine_path = str(cache_dir / fname)

    if Path(engine_path).exists():
        print(f"[entrypoint] Cached engine found: {engine_path}", flush=True)
    else:
        export_engine(pt_path, engine_path, IMGSZ, HALF, DEVICE, WORKSPACE)
        cleanup_stale_engines(cache_dir, engine_path)

    # Replace this process with the main service.
    # Pass --model to override the value from default.json so the service
    # picks up the .engine file instead of the .pt file.
    cmd = [
        sys.executable, '-u', '-m',
        'spacenorm_obj_detection.spacenorm_obj_detection',
        '--model', engine_path,
    ]
    print(f"[entrypoint] Launching: {' '.join(cmd)}", flush=True)
    os.execv(sys.executable, cmd)


if __name__ == '__main__':
    main()
