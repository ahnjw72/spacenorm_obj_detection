❯ The docker image made by ./docker_build/docker_build_opencv_gstreamer.sh is not compatible on some machines with incompatible NVIDIA device driver, CUDA and PyTorch versions. This is frustrating since
  the docker image cannot solve the machine-dependency problem. What is the most appropriate approach in this situation?

● This is a fundamental limitation of GPU containers: Docker isolates the OS and CUDA runtime, but it cannot isolate the NVIDIA kernel driver — the container always uses the host's driver. The
  compatibility constraint is:

  Host driver version  ≥  minimum driver required by CUDA runtime in container

  The current image uses pytorch/pytorch:2.1.2-cuda12.1 which requires driver ≥ 525. Machines with older drivers (e.g. 470.x for CUDA 11.4) cannot run it.

  ---
  Recommended approaches

  1. Build multiple CUDA-versioned images (best fit for this project)

  You already do this partially — Dockerfile.Blackwell exists for RTX5060. Extend the pattern to cover CUDA versions matching your fleet's driver landscape:

  ┌───────────────────────────┬─────────────────────────┬────────────┬────────────────────┐
  │         Image tag         │          Base           │ Min driver │    Target nodes    │
  ├───────────────────────────┼─────────────────────────┼────────────┼────────────────────┤
  │ spacenorm_yolov7:cuda12.1 │ pytorch:2.1.2-cuda12.1  │ 525        │ New machines       │
  ├───────────────────────────┼─────────────────────────┼────────────┼────────────────────┤
  │ spacenorm_yolov7:cuda11.8 │ pytorch:2.0.1-cuda11.8  │ 450        │ Mid-range machines │
  ├───────────────────────────┼─────────────────────────┼────────────┼────────────────────┤
  │ spacenorm_yolov7:cuda11.3 │ pytorch:1.11.0-cuda11.3 │ 450        │ Older machines     │
  └───────────────────────────┴─────────────────────────┴────────────┴────────────────────┘

  deploy_yolov7.sh already has a GROUP1/GROUP2 split — extend it to select the right image per node label (e.g. node.labels.cuda_version).

  2. Downgrade the base image to the lowest CUDA version the model actually needs

  The local environment runs torch 1.11.0+cu113, meaning CUDA 11.3 is sufficient for this model. The CUDA 12.1 base image is likely overkill unless you specifically need features from PyTorch 2.x.
  Rebuilding against CUDA 11.8 or 11.3 would make the image compatible with a much wider driver range with no functional loss.

  3. Standardize driver versions across the fleet

  Update all deployment nodes to a consistent driver version (e.g. 525+). This is the cleanest long-term solution and eliminates the image-matrix maintenance burden. In a controlled edge deployment like
  this one it is feasible. The downside is it requires physical/SSH access to each node and a maintenance window.

  ---
  Practical first step

  Check the actual driver versions on the failing machines:

  # On each node
  nvidia-smi --query-gpu=driver_version --format=csv,noheader

  Then cross-reference with https://docs.nvidia.com/deploy/cuda-compatibility/ to find the highest CUDA version each node supports. That determines which base image to target.

  ---
  What NOT to do

  - CUDA Forward Compatibility (running a newer CUDA runtime on an older driver via libcuda.so injection) — works in theory but fragile and not recommended for production deployments.
  - Installing CUDA inside the container at runtime — defeats the purpose of the image.

  Given that deploy_yolov7.sh already handles per-group image selection, option 1 (multi-image) or option 2 (downgrade base) are the most practical paths with the least infrastructure change.

✻ Crunched for 39s

❯ But current docker image from ./docker_build/docker_build_opencv_gstreamer.sh cannot run on a machine with Driver Version: 591.86 and CUDA Version: 13.1. This is not compatible with your statement "Host
   driver version  ≥  minimum driver required by CUDA runtime in container".

● You are right to correct me. The general rule holds in most cases, but there are real exceptions. Let me clarify what can go wrong in both directions.

  Why a newer driver can still break an older CUDA container

  My statement described the ideal behavior, but several things can break it in practice:

  1. NVIDIA Container Toolkit not updated to match the new driver
  The container toolkit (nvidia-container-runtime / libnvidia-container) on the host is a separate component from the driver. If the toolkit is outdated relative to driver 591.86, it may fail to correctly
  inject GPU libraries into the container even though the driver itself is capable.

  2. CUDA major version boundary (12.x → 13.x)
  NVIDIA's backward compatibility guarantee is strongest within a major version (e.g. CUDA 12.1 container on CUDA 12.8 driver). Across a major version jump (12 → 13), the driver-side API (libcuda.so) may
  introduce changes that older CUDA runtimes inside containers don't handle correctly.

  3. Driver 591.86 is a Windows driver
  This is significant. On Windows, GPU-accelerated Docker containers run through WSL2 with CUDA-on-WSL2, which has a different compatibility layer than native Linux. The GStreamer pipeline in particular
  (gstreamer1.0-libav, /dev/video, hardware decode) is unlikely to work at all through WSL2.

  ---
  Corrected picture

  The compatibility problem runs in both directions:

  Too old driver  ──────────────────────────────  Too new driver
       │                                                │
       │  CUDA 12.1 container                          │
       │  needs driver ≥ 525                           │
       │                                          Container toolkit
       │                                          or major-version API
    Cannot run                                    mismatch can also fail

  ---
  The right approach for this specific case

  Check the Container Toolkit version first — this is the most common cause:
  nvidia-container-toolkit --version   # or
  dpkg -l | grep nvidia-container
  Update it to the latest version. This alone often resolves failures with new drivers.

  If it is a Windows/WSL2 machine, the Docker image as built is not intended to run there. The GStreamer RTSP pipeline has no path to real camera hardware through WSL2. This is a use-case mismatch, not a
  CUDA version problem.

  If it is a native Linux machine with driver 591.86, the fix is to update the base image to pytorch/pytorch:2.5.1-cuda12.4 or similar — a newer CUDA 12.x image that the new driver has been validated
  against, while remaining compatible with older CUDA 12.x machines (driver ≥ 525).

✻ Crunched for 1m 15s

❯ The machine is equipped with RTX 4080.

● That additional detail clarifies the situation significantly.

  RTX 4080 is Ada Lovelace (sm_89), fully supported since CUDA 11.8 — the GPU architecture is not the problem. And driver 591.86 is a Windows driver (Linux drivers follow a different numbering, e.g.
  535.xx, 550.xx, 565.xx). So the machine is a Windows workstation running Docker via WSL2.

  This changes the diagnosis entirely — it is not a CUDA version incompatibility. The failure mode on Windows + WSL2 is:

  ┌──────────────────────────┬────────────────────────────────────────────────────────────────┐
  │        Component         │                             Issue                              │
  ├──────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ CUDA compute (inference) │ Generally works through WSL2 CUDA passthrough                  │
  ├──────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ GStreamer RTSP capture   │ No access to physical camera hardware through WSL2             │
  ├──────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ /dev/video USB cameras   │ Not accessible in WSL2                                         │
  ├──────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Network (RTSP streams)   │ Accessible but through WSL2 NAT, not the host network directly │
  └──────────────────────────┴────────────────────────────────────────────────────────────────┘

  ---
  Practical recommendation

  For mAP evaluation on this machine — skip Docker entirely. You already have a working conda environment (trt_y7) on this machine and just ran python test.py natively. Docker adds no value here and WSL2
  adds friction. Just run:

  cd ~/Work/spacenorm_yolov7   # or equivalent Windows path in WSL2
  python test.py \
    --weights spacenorm_yolov7.pt \
    --data data/map_test.yaml \
    --img-size 1280 \
    --device 0 \
    --task val \
    --verbose

  For production deployment — the RTX 4080 Windows machine is not a target for the Docker Swarm service anyway. The edge nodes are Linux machines. The Docker image compatibility issue only matters for
  those Linux nodes.

✻ Baked for 36s

❯ If the docker image uses host PC's GPU driver, why is there a problem about the NVIDIA GPU architecture change, e.g. from Ada Lovelace to Blackwell ?

● Excellent question. There are two separate and distinct layers involved, and it is easy to confuse them.

  The two independent layers

  Layer 1 — Driver API (host-side)

  libcuda.so lives on the host and is shared into the container. This handles communication between CUDA code and the GPU hardware. It is backward compatible — a newer driver can run older CUDA runtimes.
  This is what my earlier statement was about.

  Layer 2 — GPU compute kernels (inside the container)

  PyTorch, cuBLAS, cuDNN and other CUDA libraries ship pre-compiled GPU kernels (cubins/SASS) for specific GPU compute capabilities (sm_xx). These are baked into the libraries at build time, not provided
  by the host driver.

  Inside the container:
    PyTorch 2.1.2 ships kernels for:
      sm_37, sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_89, sm_90

    Blackwell = sm_120  ← not in this list

  When PyTorch runs on a Blackwell GPU it:
  1. Looks for a native cubin for sm_120 → not found
  2. Tries PTX JIT compilation as fallback → PTX in PyTorch 2.1.2 also does not cover sm_120
  3. Fails with CUDA error: no kernel image is available for execution on the device

  The host driver (which fully supports Blackwell) is irrelevant here — the problem is that the GPU kernels inside the container were never compiled for that architecture.

  Summary

  ┌───────────────────────────────────┬────────────────────────────────────────┬──────────────────┐
  │              Problem              │               Root cause               │  Where it lives  │
  ├───────────────────────────────────┼────────────────────────────────────────┼──────────────────┤
  │ Old driver, new CUDA runtime      │ Driver API version too low             │ Host             │
  ├───────────────────────────────────┼────────────────────────────────────────┼──────────────────┤
  │ New GPU architecture, old PyTorch │ Missing compiled kernels for new sm_xx │ Inside container │
  └───────────────────────────────────┴────────────────────────────────────────┴──────────────────┘

  This is exactly why Dockerfile.Blackwell exists — it uses a newer PyTorch version that includes pre-compiled kernels for sm_120, not because Blackwell needs a different driver.

❯ Then, will docker image built by Dockerfile.Blackwell also run on machines with older NVIDIA GPUs since the new PyTorch have all the previous sm_nn as well as newer sm_120 ?

● Let me check what the Blackwell Dockerfile actually uses:

  Searched for 1 pattern, read 1 file (ctrl+o to expand)

● Yes, mostly — but the constraint shifts from GPU architecture to driver version.

  The Blackwell image uses nvidia/cuda:12.8.0 with PyTorch from cu128. PyTorch built against CUDA 12.8 ships pre-compiled kernels for all generations:

  sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_120

  So an RTX 3080 (sm_86) or RTX 4080 (sm_89) has no missing-kernel problem with this image. However, CUDA 12.8 inside the container requires the host driver to be recent enough to support it:

  ┌───────────────────────────┬────────────────────────────┬─────────────────────────────┐
  │                           │ Standard image (CUDA 12.1) │ Blackwell image (CUDA 12.8) │
  ├───────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ Min Linux driver          │ ≥ 525                      │ ≥ 570                       │
  ├───────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ GPU kernel coverage       │ up to sm_90                │ up to sm_120                │
  ├───────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ Runs on RTX 3080 (sm_86)  │ Yes (if driver ≥ 525)      │ Yes (if driver ≥ 570)       │
  ├───────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ Runs on RTX 4080 (sm_89)  │ Yes                        │ Yes                         │
  ├───────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ Runs on RTX 5060 (sm_120) │ No (missing kernels)       │ Yes                         │
  └───────────────────────────┴────────────────────────────┴─────────────────────────────┘

  The key insight: driver ≥ 570 is available for older GPU generations too. NVIDIA provides recent drivers for GPUs going back many years. So updating the driver on an older machine is generally feasible
  without changing the hardware.

  This means consolidating onto the Blackwell image alone is viable if you can bring all deployment node drivers up to ≥ 570 — which eliminates the need to maintain two separate images. The only machines
  where this won't work are those where the driver cannot be updated (e.g., locked-down enterprise systems or very old GPUs that NVIDIA no longer provides new drivers for).

  ❯ How to upgrade the device driver ? Should the CUDA runtime in the host side be upgraded when I upgrade the driver ?

● Driver upgrade on Linux (Ubuntu/Debian edge nodes)

  Method 1 — apt (recommended)
  # See what is currently installed and what is available
  nvidia-smi
  apt list --installed 2>/dev/null | grep nvidia-driver

  # Install the target driver version
  sudo apt update
  sudo apt install nvidia-driver-570   # or whatever version you need

  sudo reboot
  nvidia-smi  # verify

  Method 2 — ubuntu-drivers (auto-selects recommended)
  sudo apt install ubuntu-drivers-common
  ubuntu-drivers devices     # lists recommended driver for your GPU
  sudo ubuntu-drivers autoinstall
  sudo reboot

  Method 3 — NVIDIA's official apt repository (most up-to-date)
  # Add NVIDIA's package repository, then:
  sudo apt install cuda-drivers-570
  sudo reboot

  ---
  Does the host CUDA runtime need upgrading?

  No. This is an important distinction:

  ┌───────────────────────────────────┬───────────────────────────────────────────┬─────────────────────────────┐
  │             Component             │                 Lives on                  │  Required for Docker GPU?   │
  ├───────────────────────────────────┼───────────────────────────────────────────┼─────────────────────────────┤
  │ NVIDIA kernel driver (nvidia.ko)  │ Host                                      │ Yes — must match libcuda.so │
  ├───────────────────────────────────┼───────────────────────────────────────────┼─────────────────────────────┤
  │ libcuda.so (driver-side library)  │ Host (injected into container at runtime) │ Yes                         │
  ├───────────────────────────────────┼───────────────────────────────────────────┼─────────────────────────────┤
  │ CUDA toolkit (nvcc, libcudart.so) │ Inside container                          │ No                          │
  ├───────────────────────────────────┼───────────────────────────────────────────┼─────────────────────────────┤
  │ nvidia-container-toolkit          │ Host                                      │ Yes                         │
  └───────────────────────────────────┴───────────────────────────────────────────┴─────────────────────────────┘

  The "CUDA Version: X.Y" shown by nvidia-smi on the host is the maximum CUDA version the driver supports — it does not mean CUDA toolkit is installed on the host. Containers bring their own libcudart.so
  (the CUDA runtime). The host only needs the driver and the container toolkit.

  So the upgrade sequence is just:

  # 1. Upgrade the driver
  sudo apt install nvidia-driver-570
  sudo reboot

  # 2. Ensure nvidia-container-toolkit is up to date
  sudo apt install nvidia-container-toolkit
  sudo systemctl restart docker

  # 3. Verify
  nvidia-smi                          # check driver version
  docker run --rm --gpus all \
    nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi   # verify GPU visible inside container

  No CUDA toolkit installation on the host is needed.

