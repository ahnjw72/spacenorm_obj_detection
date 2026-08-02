# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Tools for talking to Hanwha NVRs/cameras over SUNAPI (Hanwha's `stw-cgi` REST
API). Two independent implementations live side by side:

- `SunapiClipPy/sunapi_clip.py` — the actively used tool. Downloads a recorded
  video clip for a given channel and time range from an NVR.
- `SunapiSampleCpp/` — Hanwha's original C++ sample REST client (libcurl-based),
  kept as a reference for the SUNAPI request/auth pattern. Not otherwise
  developed.
- `doc/*.pdf` — the full SUNAPI 2.6.2 API reference, split by subsystem
  (`video_audio`, `recording`, `network`, `transfer`, `event`, ...) plus
  `SUNAPI_Application_Programmers_Guide_2.6.2.pdf`, which has worked end-to-end
  examples (playback/backup RTSP session, RTSP-over-HTTP, etc.) that the
  per-subsystem docs don't spell out. These are the only spec for the API —
  there is no other schema/reference in the repo. To search them:
  `pdftotext -layout doc/SUNAPI_xxx.pdf - | grep -n -i <term>`.

## Commands

Python clip tool (no build step; needs `requests` — already present via
system `python3-requests` — and an `ffmpeg` binary on PATH):

```
python3 SunapiClipPy/sunapi_clip.py --host <nvr-host> --username <u> --password <p> --list-channels

python3 SunapiClipPy/sunapi_clip.py --host <nvr-host> --username <u> --password <p> \
  --channel <0-based-id> --start <YYYY-MM-DDTHH:MM:SS> --end <YYYY-MM-DDTHH:MM:SS> \
  --output clip.mp4
```

C++ sample (reference only):

```
cd SunapiSampleCpp && make        # produces ./testClient
./testClient <IP> <username> <password>
```

There is no automated test suite. The tool is verified by running it against
a live NVR (see "Known-good / known-bad targets" below) and checking the
resulting channel list / MP4 plays back correctly.

## Architecture of `sunapi_clip.py`

Two-phase design, both phases hit the same NVR host but different
protocols/ports:

1. **REST discovery** (`stw-cgi/media.cgi`, HTTP/HTTPS): calls
   `msubmenu=streamuri&action=view&MediaType=Search` to get the NVR's
   playback RTSP URI template (`rtsp://<host>:<port>/PlaybackChannel/<ch>/media.smp`).
   This is the only reliable way to learn the NVR's actual RTSP port — it is
   not necessarily the SUNAPI default (554/558).
2. **RTSP backup session** (via `ffmpeg`, not implemented in Python): the
   playback URI's `PlaybackChannel` is rewritten to `BackupChannel` and
   `/start=<ts>&end=<ts>` is appended to the path (this is a SUNAPI-specific
   URL convention documented only in the Application Programmer's Guide,
   chapter 10.3 "Backup Session" — *not* in `SUNAPI_video_audio_2.6.2.pdf`).
   A "backup" RTSP session streams the recording as fast as possible instead
   of in real time, which is what makes this a clip *download* rather than a
   live-speed playback capture. `ffmpeg -c copy` remuxes that stream straight
   to MP4 with no re-encoding.

Other implementation details worth knowing before changing this file:

- **Auth**: SUNAPI devices challenge with Digest, but `sunapi_get()` tries
  Digest then falls back to Basic — mirrors the C++ sample's
  `CURLAUTH_ANY`, since not all deployments behave the same.
- **Always requests JSON** (`Accept: application/json` header) — SUNAPI can
  return `text/plain` key=value pairs instead, which would need different
  parsing; forcing JSON keeps `sunapi_get()` simple.
- **TLS verification defaults off** (`--verify-ssl` to turn on) — NVRs
  typically serve self-signed certs.
- **Channel IDs are 0-based** and match `cameraregister`'s `Channel` field
  directly — same numbering is reused as the SUNAPI `Channel` param and in
  the `BackupChannel/<ch>/` RTSP path segment.
- **Time formats**: NVR local time is `YYYYMMDDTHHMMSS`; UTC is
  `YYYYMMDDTHHMMSSZ` (trailing `Z`, detected in `parse_sunapi_time()` from
  whether the user's `--start`/`--end` value ends in `Z`). Camera-only
  deployments use a different, `T`-less format — not handled here since this
  tool targets NVRs.

## Known-good / known-bad targets

- `yujin.spacenorm.com` — works end-to-end on default `https://` (443).
  64 channels (QNO-6022R / XNO-L6080R / QND-6022R).
- `cheilacc-ansung.spacenorm.com` — port 443 timed out (not refused), so its
  web UI is very likely on a non-default port behind NAT/DDNS. Needs the
  actual port from whoever manages that NVR before it'll work; pass it as
  `--host https://cheilacc-ansung.spacenorm.com:<port>`. If the RTSP port
  SUNAPI reports isn't reachable either (common behind NAT — it may be the
  NVR's internal LAN port, not what's forwarded), use `--rtsp-port` to
  override, or `--rtsp-transport http` to tunnel RTSP over the web port
  (SUNAPI's `RTSPOverHTTP` mechanism).
