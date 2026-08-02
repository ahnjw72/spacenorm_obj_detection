#!/usr/bin/env python3
"""Download a recorded video clip for one channel/time-range from a Hanwha
(SUNAPI) NVR.

Two steps:
  1. REST call (stw-cgi/media.cgi, msubmenu=streamuri) to learn the NVR's
     playback RTSP host/port.
  2. Connect to the SUNAPI "backup" RTSP session for that channel with the
     requested start/end time embedded in the URL, and let ffmpeg remux the
     stream straight to a file (no re-encoding).

Requires the `requests` Python package and an `ffmpeg` binary on PATH.
"""
import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime
from urllib.parse import quote, urlsplit

import requests
import urllib3
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

STW_CGI = "/stw-cgi"


def normalize_base_url(host: str) -> str:
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", host):
        host = "https://" + host
    return host.rstrip("/")


def sunapi_get(base_url: str, cgi: str, params: dict, username: str, password: str,
               verify: bool, timeout: float) -> dict:
    url = f"{base_url}{STW_CGI}/{cgi}"
    headers = {"Accept": "application/json"}
    last_resp = None
    for auth in (HTTPDigestAuth(username, password), HTTPBasicAuth(username, password)):
        resp = requests.get(url, params=params, auth=auth, headers=headers,
                             verify=verify, timeout=timeout)
        last_resp = resp
        if resp.status_code != 401:
            break
    if last_resp.status_code == 401:
        raise SystemExit("Authentication failed (checked both Digest and Basic auth). "
                          "Verify username/password.")
    if not last_resp.ok:
        raise SystemExit(f"SUNAPI request failed: {last_resp.status_code} {last_resp.text[:500]}")
    try:
        return last_resp.json()
    except ValueError:
        raise SystemExit(f"Unexpected non-JSON response from {url}:\n{last_resp.text[:500]}")


def get_registered_channels(base_url, username, password, verify, timeout):
    """Return the NVR's RegisteredCameras list (structured), each a dict with
    keys like 'Channel' (0-based int), 'Model', 'IPAddress', 'Status'."""
    data = sunapi_get(base_url, "media.cgi",
                       {"msubmenu": "cameraregister", "action": "view"},
                       username, password, verify, timeout)
    return data.get("RegisteredCameras", [])


def list_channels(base_url, username, password, verify, timeout):
    cameras = get_registered_channels(base_url, username, password, verify, timeout)
    if not cameras:
        print("No registered cameras found.")
        return
    for cam in cameras:
        print(f"Channel {cam.get('Channel')}: {cam.get('Model', '?')} "
              f"@ {cam.get('IPAddress', '?')} (status: {cam.get('Status', '?')})")


def get_playback_stream_uri(base_url, username, password, channel, verify, timeout):
    data = sunapi_get(base_url, "media.cgi",
                       {"msubmenu": "streamuri", "action": "view", "Channel": channel,
                        "MediaType": "Search", "Mode": "Full", "ClientType": "PC"},
                       username, password, verify, timeout)
    uri = data.get("URI")
    if not uri:
        raise SystemExit(f"SUNAPI did not return a stream URI: {data}")
    return uri


def parse_sunapi_time(value: str):
    """Returns (datetime, SUNAPI-formatted timestamp string)."""
    is_utc = value.endswith(("Z", "z"))
    raw = value[:-1] if is_utc else value
    dt = datetime.fromisoformat(raw)
    stamp = dt.strftime("%Y%m%dT%H%M%S") + ("Z" if is_utc else "")
    return dt, stamp


def build_backup_url(playback_uri: str, channel: int, start_stamp: str, end_stamp: str,
                      username: str, password: str, rtsp_port: int) -> str:
    parsed = urlsplit(playback_uri)
    host = parsed.hostname
    if host is None:
        raise SystemExit(f"Could not parse host out of stream URI: {playback_uri}")
    port = rtsp_port or parsed.port or 554
    backup_path = parsed.path.replace("PlaybackChannel", "BackupChannel", 1)
    if "BackupChannel" not in backup_path:
        backup_path = f"/BackupChannel/{channel}/media.smp"
    userinfo = f"{quote(username, safe='')}:{quote(password, safe='')}@"
    return f"rtsp://{userinfo}{host}:{port}{backup_path}/start={start_stamp}&end={end_stamp}"


def redact(url: str) -> str:
    return re.sub(r"//[^@/]+@", "//****:****@", url)


def download_clip(rtsp_url: str, output_path: str, transport: str, duration_seconds: float,
                  hard_timeout: float = None):
    # Wall-clock cap: a SUNAPI backup session streams "as fast as possible", so a
    # healthy 60s clip downloads in seconds. If the session connects but the NVR
    # never streams packets (dead channel / no recording for the window), `-t`
    # (which bounds STREAM time, not real time) never fires and ffmpeg hangs
    # forever. This timeout kills it so one bad clip can't freeze the whole sweep.
    if hard_timeout is None:
        hard_timeout = int(duration_seconds) + 60
    cmd = [
        "ffmpeg", "-y",
        "-rtsp_transport", transport,
        "-i", rtsp_url,
        "-t", str(int(duration_seconds) + 10),
        "-c", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    print(f"Running: {' '.join(redact(a) for a in cmd)}")
    try:
        result = subprocess.run(cmd, timeout=hard_timeout)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"ffmpeg timed out after {hard_timeout:.0f}s "
                         "(stalled RTSP backup session — connected but no data)")
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg exited with code {result.returncode}")


def download_channel_clip(base_url, username, password, channel, start_iso, end_iso,
                           output_path, verify=False, timeout=10.0,
                           rtsp_transport="tcp", rtsp_port=None, verbose=True,
                           download_timeout=None):
    """Download one channel's recorded clip for [start_iso, end_iso] to output_path.

    Wraps the full two-phase flow (REST stream-uri discovery -> RTSP backup
    session remuxed by ffmpeg). Times are ISO strings as accepted by
    parse_sunapi_time (trailing 'Z' => UTC). Returns output_path.
    """
    start_dt, start_stamp = parse_sunapi_time(start_iso)
    end_dt, end_stamp = parse_sunapi_time(end_iso)
    duration = (end_dt - start_dt).total_seconds()
    if duration <= 0:
        raise SystemExit("end must be after start")

    playback_uri = get_playback_stream_uri(base_url, username, password,
                                            channel, verify, timeout)
    if verbose:
        print(f"Playback stream template: {playback_uri}")

    backup_url = build_backup_url(playback_uri, channel, start_stamp, end_stamp,
                                   username, password, rtsp_port)
    if verbose:
        print(f"Backup RTSP URL: {redact(backup_url)}")

    download_clip(backup_url, output_path, rtsp_transport, duration,
                  hard_timeout=download_timeout)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True,
                         help="NVR host, e.g. cheilacc-ansung.spacenorm.com "
                              "or https://host:port (default scheme: https)")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--channel", type=int,
                         help="0-based SUNAPI channel ID (see --list-channels)")
    parser.add_argument("--start", help="Clip start time, e.g. 2026-07-14T09:00:00 "
                                         "(local NVR time) or ...T09:00:00Z (UTC)")
    parser.add_argument("--end", help="Clip end time, same format as --start")
    parser.add_argument("--output", help="Output file path (default: auto-generated .mp4)")
    parser.add_argument("--rtsp-transport", choices=["tcp", "udp", "http"], default="tcp",
                         help="RTSP transport ffmpeg should use. Use 'http' if the NVR "
                              "is only reachable through its web port (RTSP-over-HTTP "
                              "tunneling), e.g. behind NAT/DDNS.")
    parser.add_argument("--rtsp-port", type=int,
                         help="Override the RTSP port instead of using the one SUNAPI "
                              "reports (useful when the externally forwarded port differs "
                              "from the NVR's internal RTSP port).")
    parser.add_argument("--verify-ssl", action="store_true",
                         help="Verify the NVR's TLS certificate (default: off, since NVRs "
                              "commonly use self-signed certs).")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP request timeout (s)")
    parser.add_argument("--list-channels", action="store_true",
                         help="List registered channels/cameras and exit.")
    args = parser.parse_args()

    if not args.list_channels and shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH. Install it first (e.g. `sudo apt install ffmpeg`).")

    base_url = normalize_base_url(args.host)
    verify = args.verify_ssl
    if not verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if args.list_channels:
        list_channels(base_url, args.username, args.password, verify, args.timeout)
        return

    missing = [name for name, val in (("--channel", args.channel), ("--start", args.start),
                                       ("--end", args.end)) if val is None]
    if missing:
        raise SystemExit(f"Missing required argument(s): {', '.join(missing)}")

    _, start_stamp = parse_sunapi_time(args.start)
    _, end_stamp = parse_sunapi_time(args.end)
    output_path = args.output or f"clip_ch{args.channel}_{start_stamp}_{end_stamp}.mp4"

    download_channel_clip(base_url, args.username, args.password, args.channel,
                          args.start, args.end, output_path,
                          verify=verify, timeout=args.timeout,
                          rtsp_transport=args.rtsp_transport, rtsp_port=args.rtsp_port)
    print(f"Saved clip to {output_path}")


if __name__ == "__main__":
    main()
