# syntax=docker/dockerfile:1
# ==========================================================
# Stage 1: Build OpenCV with GStreamer + FFmpeg support
#
# Uses the CUDA devel image so OpenCV can optionally link
# against CUDA headers. opencv_dnn is disabled, but having
# CUDA headers available keeps the build self-contained.
# ==========================================================
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Kakao mirror (faster in Korea; remove these two lines if building elsewhere)
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirror.kakao.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://mirror.kakao.com/ubuntu|g' /etc/apt/sources.list

WORKDIR /workspace

# Build tools + GStreamer / FFmpeg dev headers required by OpenCV cmake
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip \
    build-essential cmake unzip pkg-config wget git ca-certificates \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
# numpy must be present before cmake so the Python bindings are built correctly
RUN pip install --no-cache-dir numpy==1.26.4

# ----------------------------------------------------------
# OpenCV build
# ----------------------------------------------------------
ARG OPENCV_VERSION=4.8.1

WORKDIR /opt
RUN wget -q -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip -q opencv.zip && rm opencv.zip

WORKDIR /opt/opencv-${OPENCV_VERSION}
RUN rm -rf build && mkdir -p build
WORKDIR /opt/opencv-${OPENCV_VERSION}/build

RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_opencv_python3=ON \
    -DOPENCV_SKIP_PYTHON_LOADER=ON \
    -DOPENCV_GENERATE_TYPING_STUBS=OFF \
    -DPYTHON3_EXECUTABLE=$(which python3) \
    -DPYTHON3_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \
    -DPYTHON3_PACKAGES_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])") \
    -DPYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
    -DWITH_GSTREAMER=ON \
    -DWITH_FFMPEG=ON \
    -DWITH_V4L=ON \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_annotation=OFF \
    -DBUILD_opencv_visualisation=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_dnn=OFF \
    -DOPENCV_GENERATE_PKGCONFIG=ON

RUN make -j"$(nproc)" && make install && ldconfig

# Strip debug symbols for a modest size reduction
RUN find /usr/local -type f -name "*.so*" -exec strip --strip-unneeded {} + || true

# Collect ONLY the OpenCV shared libraries that the runtime image needs.
# /usr/local in this devel image also contains the entire CUDA SDK (~5-7 GB of
# headers, static libs, nvcc, etc.) that must NOT be copied to the runtime stage.
RUN mkdir /opencv-libs && \
    find /usr/local/lib -maxdepth 1 -name 'libopencv_*.so*' -exec cp {} /opencv-libs/ \;

# Pack the cv2 Python artifacts into a tarball so the runtime stage can
# install them without hard-coding a python3.x path.
RUN python3 <<'EOF'
import site, pathlib, tarfile

sp = pathlib.Path(site.getsitepackages()[0])
cv2_dir = sp / 'cv2'
tar_path = pathlib.Path('/tmp/opencv-cv2.tgz')

with tarfile.open(tar_path, 'w:gz') as tf:
    if cv2_dir.exists() and cv2_dir.is_dir():
        tf.add(str(cv2_dir), arcname='cv2')
    else:
        so_files = sorted(sp.glob('cv2*.so'))
        if not so_files:
            raise SystemExit(f'[FATAL] cv2 not found under site-packages: {sp}')
        for f in so_files:
            tf.add(str(f), arcname=f.name)

print(f'[OK] Exported cv2 artifacts to {tar_path}')
EOF

# ==========================================================
# Stage 2: Runtime image (no build tools, smaller footprint)
# ==========================================================
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Kakao mirror
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirror.kakao.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://mirror.kakao.com/ubuntu|g' /etc/apt/sources.list

WORKDIR /app

# Runtime packages only (no *-dev, no compilers)
# gstreamer1.0-libav provides the GStreamer FFmpeg integration needed for RTSP.
# Ubuntu 22.04 versioned av-libs are pulled automatically by ffmpeg/gstreamer1.0-libav.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    ffmpeg \
    libgl1 \
    libsm6 \
    libxext6 \
    vim-tiny \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/vim.tiny /usr/bin/vi || true

# Copy ONLY the OpenCV shared libs (not the full CUDA devel tree from builder)
# and the cv2 Python binding tarball.
COPY --from=builder /opencv-libs /usr/local/lib/
COPY --from=builder /tmp/opencv-cv2.tgz /tmp/opencv-cv2.tgz

RUN ldconfig

# Unpack cv2 into this image's site-packages (path resolved at runtime)
RUN python3 <<'EOF'
import site, tarfile, pathlib

sp = pathlib.Path(site.getsitepackages()[0])
tar_path = pathlib.Path('/tmp/opencv-cv2.tgz')

with tarfile.open(tar_path, 'r:gz') as tf:
    tf.extractall(sp)

tar_path.unlink(missing_ok=True)
print(f'[OK] Installed cv2 into {sp}')
EOF

# PyTorch with CUDA 12.8 wheels.
# cu128 wheels include kernels for all GPU architectures (Volta, Turing,
# Ampere, Ada, Hopper, Blackwell sm_120), so this image runs on any
# CUDA-capable card as long as the host driver is >= 570.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128

# Runtime Python dependencies (inference-only; training packages excluded)
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    matplotlib==3.8.4 \
    pillow \
    prometheus_client \
    paho-mqtt \
    pyyaml \
    requests \
    tqdm \
    flask \
    ultralytics \
    psutil

# TensorRT export pipeline.
# tensorrt-cu12 (or ultralytics transitive deps) may pull in plain onnxruntime
# (CPU build) which conflicts with onnxruntime-gpu.  Explicitly uninstall the
# CPU build first so there is no conflict, then install the GPU build normally.
# Do NOT use --force-reinstall: it cascades to all deps and can silently upgrade
# numpy past 1.x, breaking the OpenCV .so that was compiled against numpy 1.26.4.
RUN pip install --no-cache-dir tensorrt-cu12 "nvidia-modelopt[onnx]>=0.44"

RUN pip uninstall -y onnxruntime 2>/dev/null || true && \
    pip install --no-cache-dir \
        "onnx>=1.12.0,<2.0.0" \
        onnxslim \
        onnxruntime-gpu

# Re-pin numpy to 1.26.4 last.  Some packages above accept numpy 2.x and may
# upgrade it, but the OpenCV .so in this image was compiled against numpy 1.x
# and is not ABI-compatible with numpy 2.x.
# The upper bound (<2) guards against any future runtime pip install (e.g.
# Ultralytics AutoUpdate) accidentally pulling in numpy 2.x.
RUN pip install --no-cache-dir "numpy>=1.26.4,<2.0"

# Verify the ONNX/TRT stack is intact — fail the build if anything is missing.
RUN python3 -c "import onnx, onnxslim, onnxruntime, tensorrt; print(f'[OK] onnx={onnx.__version__}  onnxslim={onnxslim.__version__}  ort={onnxruntime.__version__}  trt={tensorrt.__version__}')"

# ultralytics pulls in opencv-python (no GStreamer) which overwrites the
# custom-built cv2 from the builder stage.  Remove it so Python falls back
# to the GStreamer-enabled .so we installed from the builder tarball.
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true

# Sanity check: ensure cv2 with GStreamer is importable and actually enabled.
# Exit non-zero (failing the build) if GStreamer support is missing.
RUN python3 -c "import cv2, sys; gst = cv2.getBuildInformation().split('GStreamer:')[1].split()[0]; print(f'[OK] cv2 {cv2.__version__} | GStreamer: {gst}'); sys.exit(0 if gst == 'YES' else 1)"

# Source code (spacenorm_cfg is excluded — injected at runtime by Docker Swarm)
COPY spacenorm_obj_detection /app/spacenorm_obj_detection

# Model weights — update the filename below to match your actual YOLOv11 .pt file
# and update docker_build.sh to copy the file into the build context first.
COPY yolo11x_set01-0148.pt /app/spacenorm_obj_detection
COPY yolo11x.pt /app/spacenorm_obj_detection

RUN mkdir -p /app/spacenorm_obj_detection/temp

CMD ["python3"]
