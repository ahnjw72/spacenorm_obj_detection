# Licensing Guide

> ⚠️ **Disclaimer:** This document is a technical summary for engineering reference,
> not legal advice. Consult a lawyer before making compliance decisions.

---

## Overview

This project uses **Ultralytics YOLO11**, which is licensed under
**GNU Affero General Public License v3.0 (AGPL-3.0)** for open-source use.
A commercial **Enterprise license** is required for production use if you do not
comply with AGPL-3.0 terms. Ultralytics does not publicly list Enterprise license
pricing — contact [https://ultralytics.com/license](https://ultralytics.com/license)
for a quote.

---

## What AGPL-3.0 Requires

AGPL-3.0 is GPL-3.0 with one additional clause:

> **Network use counts as distribution.** If users interact with your software
> over a network, you must offer them the complete corresponding source code under
> AGPL-3.0 — even if you never ship a binary.

For this project, "users" include the Spacenorm API server and any client that
receives occupancy data from the detection service.

The two paths to compliance:

| Path | What it means |
|------|--------------|
| **Open source** | Publish all source code under AGPL-3.0 and make it accessible to users of the service |
| **Enterprise license** | Purchase a commercial license from Ultralytics — no open-sourcing required |

---

## What Must Be Open-Sourced (AGPL-3.0 Path)

### Must include

| Item | Examples |
|------|---------|
| All Python source code | `spacenorm_obj_detection/`, `entrypoint.py`, `offline_entrypoint.py`, `offline_spacenorm_obj_detection.py` |
| Docker build files | `docker_build/Dockerfile`, `docker_build/docker_build.sh` |
| Deployment scripts and templates | `docker_swarm/stack.yml.template`, `docker_swarm/deploy_obj_detection.sh` |
| Configuration schemas | Structure and format of `default.json`, `cctv_*.json` |
| AGPL-3.0 `LICENSE` file | Must be present in the repository root |
| Any modifications to Ultralytics code | None in this project currently |

### Does NOT need to be open-sourced

| Item | Reason |
|------|--------|
| Custom model weights (`yolo11x_set01-0148.pt`) | Trained artifact / data — not software |
| CCTV credentials and device keys (`*.keys`) | Secrets — excluded via `.gitignore` |
| Customer-specific RTSP URIs in `cctv_*.json` | Sensitive operational data, not code |
| Spacenorm API server code | Separate project with its own license |

---

## Practical Compliance Procedure

### Step 1 — Add LICENSE file
The AGPL-3.0 `LICENSE` file has been added to the repository root. ✅

### Step 2 — Verify secrets are excluded
Confirm these patterns are in `.gitignore` and have never been committed:
```
spacenorm_cfg/cctv/*.keys
```
Customer-specific `cctv_*.json` files containing RTSP URIs should also be excluded
or have credentials replaced with placeholders before committing.

### Step 3 — Make the repository public
On GitHub: **Settings → Danger Zone → Change visibility → Public**

The repository at `github.com/ahnjw72/spacenorm_obj_detection` is already on GitHub.
Making it public satisfies the "source must be accessible to users" requirement.

### Step 4 — Add a source notice to the running service
AGPL-3.0 requires that users interacting with the service over the network can find
the source code. Add a header to Flask responses in
`spacenorm_obj_detection/spacenorm_obj_detection.py`:

```python
@app.after_request
def add_source_header(response):
    response.headers['X-Source-Code'] = \
        'https://github.com/ahnjw72/spacenorm_obj_detection'
    return response
```

Alternatively, include the repository URL in the web UI footer or API documentation.

### Step 5 — Keep source in sync with deployments
Every time a new image is pushed to ECR and deployed, the corresponding source code
must be on the public branch. Since deployment already uses git, this is naturally
satisfied as long as all changes are committed and pushed before deploying.

---

## Is the Timing of Open-Sourcing Important?

**Yes — legally, the obligation begins the moment the service is deployed.**
Retroactively open-sourcing does not erase the period of non-compliance, but it is
still the right course of action and significantly reduces ongoing legal risk.

### The Cure Clause (AGPL-3.0 Section 8)

AGPL-3.0 has a built-in cure mechanism:

| Scenario | Outcome |
|----------|---------|
| First violation — cured within **30 days of being notified** | License automatically reinstated; no legal action possible |
| First violation — cured without ever being notified | License reinstated after 60 days of compliance |
| Repeat violation (previously notified) | Reinstatement at copyright holder's discretion only |

### How Enforcement Actually Works

```
Ultralytics discovers the violation
        │
        ▼
Sends a written notice / cease-and-desist
        │
        ▼
Company is given a chance to cure (come into compliance)
        │
   ┌────┴────┐
   │         │
Complies   Ignores / refuses
   │         │
License    Legal action
reinstated (rare, costly for both sides)
```

In practice, most enforcement starts with a written notice and a request to comply.
Companies that respond promptly and in good faith are rarely sued. Ultralytics has
a commercial incentive to enforce their license (they sell Enterprise licenses), so
they actively monitor for violations — but litigation against a small company that
immediately complies after notification is rare.

### Practical Risk Factors for This Project

| Factor | Risk level |
|--------|-----------|
| Small company, limited commercial revenue | Lower |
| Ultralytics actively monitors for violations | Higher |
| Project already on GitHub (just private) | Lower — easy to make public quickly |
| Custom model weights are excluded | Neutral — weights not covered by AGPL |
| Service runs on private network / VPN | Lower — harder to discover |

### Recommended Action Regardless of Past Timing

1. **Come into compliance now** — add `LICENSE`, make repo public, add source notice
2. **Document the date of compliance** — starts the 60-day automatic reinstatement clock
3. **Do not wait to be notified** — proactive compliance demonstrates good faith and
   eliminates the 30-day conditional window
4. **Consult a lawyer** if the project has been running commercially for a significant
   time without compliance — they can assess actual financial exposure

---

## Alternative Models with Commercial-Friendly Licenses

If avoiding the AGPL-3.0 obligation entirely is a priority, the options are:

### Option A — Ultralytics Enterprise License
Purchase a commercial license from Ultralytics. No open-sourcing required. Pricing
not publicly listed; contact [https://ultralytics.com/license](https://ultralytics.com/license).

### Option B — Ultralytics-Free Runtime (ONNX/TRT path)
Keep Ultralytics only for the one-time offline export step (PT → ONNX); replace
the runtime inference with TensorRT Python API + custom pre/post-processing.
This removes Ultralytics from the deployed production service.
See [ultralytics_free_inference_feasibility.md](ultralytics_free_inference_feasibility.md)
for the full technical analysis and implementation guide.

### Option C — Retrain on an Apache 2.0 Model

The table below lists genuinely commercial-friendly alternatives.

| Model | Code license | Weights license | Status |
|-------|-------------|----------------|--------|
| YOLOv11 | AGPL-3.0 | AGPL-3.0 | ✅ Actively developed |
| YOLO-NAS | Apache 2.0 | ❌ **Non-commercial** | ⛔ Abandoned (see below) |
| **RT-DETR** | **Apache 2.0** | **Apache 2.0** | ✅ Active (PaddlePaddle + Ultralytics wrapper) |
| **Detectron2** | **Apache 2.0** | **Apache 2.0** | ✅ Active (Meta) |

> ⚠️ **YOLO-NAS is not a viable alternative** despite the Apache 2.0 code license:
>
> - The **pre-trained weights** are governed by a separate non-commercial license
>   ([LICENSE.YOLONAS.md](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md))
>   that explicitly prohibits *"any commercial use, including in connection with any
>   models used in a production environment."*
> - Deci AI was **acquired by NVIDIA in April 2024** (~$300M) and dissolved as an
>   independent entity. The last super-gradients release was v3.7.1 (April 8, 2024);
>   the project is effectively abandoned. Pre-trained model download URLs are broken
>   and documentation redirects to nvidia.com.
>
> YOLO-NAS weights are therefore **more restrictive** than Ultralytics YOLO11
> for commercial use, and the project has no active maintainer.

**RT-DETR** is the most practical Apache 2.0 alternative — comparable accuracy to
YOLOv11, supported by PaddlePaddle (original) and available as an Ultralytics
wrapper (`from ultralytics import RTDETR`). Switching requires retraining the
custom model on the project dataset.

---

## YOLOv11 vs. YOLO-NAS Performance Comparison

Benchmarks on COCO val2017, T4 GPU, TensorRT FP16, input 640×640.
Sources: [Ultralytics YOLO11 docs](https://docs.ultralytics.com/models/yolo11/),
[Deci-AI/super-gradients YOLONAS.md](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md).

| Model | mAP50-95 | T4 TRT FP16 (ms) | Params (M) |
|-------|----------|-----------------|------------|
| YOLO11s | 47.0 | 2.5 | 9.4 |
| YOLO-NAS S | 47.5 | 3.21 | 19.0 |
| YOLO11m | 51.5 | 4.7 | 20.1 |
| YOLO-NAS M | 51.55 | 5.85 | 51.1 |
| YOLO11l | 53.4 | 6.2 | 25.3 |
| YOLO-NAS L | 52.22 | 7.87 | 66.9 |
| **YOLO11x** | **54.7** | **11.3** | **56.9** |

YOLOv11 achieves equal or better mAP with 2–3× fewer parameters and faster TRT
inference at every size tier. YOLO-NAS offers no meaningful advantage for this
project, and its non-commercial weights license rules it out regardless.

> **Note:** YOLO11 latency uses standard Ultralytics TRT export; YOLO-NAS latency
> uses Deci AI's proprietary quantization-aware TRT export. Benchmarks were produced
> by different teams with potentially different TRT versions — treat as approximate.

---

## Summary of Recommended Actions

| Priority | Action | Status |
|----------|--------|--------|
| 1 | Add `LICENSE` (AGPL-3.0) to repo root | ✅ Done |
| 2 | Make GitHub repository public | ⬜ Pending |
| 3 | Add `X-Source-Code` response header to Flask service | ⬜ Pending |
| 4 | Ensure `*.keys` and credential files are in `.gitignore` | ✅ Already excluded |
| 5 | Keep public repo in sync with every production deployment | ✅ Naturally satisfied via git |
