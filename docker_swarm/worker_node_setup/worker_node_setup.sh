#!/usr/bin/env bash
# spacenorm_worker_join_swarm.sh
#
# What it does (in order):
#  1) Checks Docker installed + daemon running
#  2) Checks there is a UID=1000 and GID=1000 on the system (as requested)
#  3) Checks manager reachability on tcp/2377
#  4) Ensures firewall allows: 7946/tcp, 7946/udp, 4789/udp (best-effort for ufw/firewalld; otherwise prints guidance)
#  5) Ensures /var/log/spacenorm_obj_detection exists
#  6) Edits /etc/docker/daemon.json to embed NVIDIA runtime + default-runtime=nvidia (JSON-safe merge)
#  7) Verifies GPU via Docker + PyTorch (torch.cuda.is_available())
#  8) Joins Docker Swarm (only if not already active)
#
# Run:
#   sudo bash spacenorm_worker_join_swarm.sh
#
# Optional env overrides:
#   MANAGER_ADDR="10.241.87.117:2377" JOIN_TOKEN="..." bash ...
#   SKIP_GPU_TEST=1 bash ...
#   PYTORCH_IMAGE="pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime" bash ...

set -Eeuo pipefail

MANAGER_IP="${MANAGER_IP:-10.241.87.117}"
MANAGER_PORT="${MANAGER_PORT:-2377}"
MANAGER_ADDR="${MANAGER_ADDR:-${MANAGER_IP}:${MANAGER_PORT}}"
JOIN_TOKEN="${JOIN_TOKEN:-SWMTKN-1-257c7lja5lzy47d8z7zryw0lwgff6shmxqtu91aosjl8aiwsyk-0g8971tut6z70xp05mxag61xp}"

LOG_DIR="/var/log/spacenorm_obj_detection"
DAEMON_JSON="/etc/docker/daemon.json"

SKIP_GPU_TEST="${SKIP_GPU_TEST:-0}"
PYTORCH_IMAGE="${PYTORCH_IMAGE:-pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime}"

say()  { printf "\n[%s] %s\n" "$(date '+%F %T')" "$*"; }
die()  { printf "\nERROR: %s\n" "$*" >&2; exit 1; }
need_root() { [[ "${EUID}" -eq 0 ]] || die "Please run as root (use: sudo bash $0)"; }

cmd_exists() { command -v "$1" >/dev/null 2>&1; }

check_uid_gid_1000() {
  say "Checking UID/GID of 1000 exist (as requested)..."
  if ! getent passwd 1000 >/dev/null; then
    die "No user with UID 1000 exists. Create a user with UID=1000 first (or adjust this requirement)."
  fi
  if ! getent group 1000 >/dev/null; then
    die "No group with GID 1000 exists. Create a group with GID=1000 first (or adjust this requirement)."
  fi
  local user_1000 group_1000
  user_1000="$(getent passwd 1000 | cut -d: -f1)"
  group_1000="$(getent group 1000 | cut -d: -f1)"
  say "OK: UID 1000 user='${user_1000}', GID 1000 group='${group_1000}'"
}

check_docker_installed_running() {
  say "Checking Docker installation..."
  cmd_exists docker || die "docker is not installed."

  say "Checking Docker daemon is running..."
  if ! systemctl is-active --quiet docker; then
    die "docker service is not active. Try: sudo systemctl enable --now docker"
  fi

  # quick sanity
  docker info >/dev/null 2>&1 || die "docker info failed. Docker daemon may be unhealthy."
  say "OK: Docker installed and running."
}

check_manager_port_2377() {
  say "Checking network access to manager ${MANAGER_ADDR} (tcp/${MANAGER_PORT})..."
  if cmd_exists nc; then
    nc -z -w 3 "${MANAGER_IP}" "${MANAGER_PORT}" \
      || die "Cannot reach manager ${MANAGER_ADDR} on TCP. Check routing/security groups/firewall."
  elif cmd_exists timeout && cmd_exists bash; then
    timeout 3 bash -c ">/dev/tcp/${MANAGER_IP}/${MANAGER_PORT}" \
      || die "Cannot reach manager ${MANAGER_ADDR} on TCP (bash /dev/tcp)."
  else
    die "Neither 'nc' nor 'timeout+bash /dev/tcp' available to test connectivity."
  fi
  say "OK: Manager reachable on tcp/${MANAGER_PORT}."
}

ensure_firewall_ports() {
  say "Ensuring firewall allows required ports: 7946/tcp, 7946/udp, 4789/udp..."

  # Docker Swarm requirements (on all nodes):
  # - 2377/tcp: swarm management (worker -> manager; manager listens)
  # - 7946/tcp+udp: node discovery
  # - 4789/udp: overlay network (VXLAN)
  #
  # We do best-effort automated rules for ufw/firewalld.
  # If neither is used, we print guidance.

  if cmd_exists ufw && ufw status >/dev/null 2>&1; then
    if ufw status | grep -qi "Status: active"; then
      say "UFW is active. Adding allow rules (idempotent)..."
      ufw allow 7946/tcp >/dev/null || true
      ufw allow 7946/udp >/dev/null || true
      ufw allow 4789/udp >/dev/null || true
      # (Optional) allow 2377/tcp inbound on managers; worker nodes don’t strictly need inbound 2377.
      say "UFW rules ensured."
      return 0
    fi
  fi

  if cmd_exists firewall-cmd && systemctl is-active --quiet firewalld; then
    say "firewalld is active. Adding permanent rules (idempotent)..."
    firewall-cmd --permanent --add-port=7946/tcp >/dev/null || true
    firewall-cmd --permanent --add-port=7946/udp >/dev/null || true
    firewall-cmd --permanent --add-port=4789/udp >/dev/null || true
    firewall-cmd --reload >/dev/null || true
    say "firewalld rules ensured."
    return 0
  fi

  say "No active ufw/firewalld detected. If you use iptables/nftables or cloud security groups, ensure these are allowed:"
  echo "  - TCP/7946, UDP/7946, UDP/4789 between all swarm nodes"
  echo "  - TCP/2377 from workers to manager(s)"
}

make_log_dir() {
  say "Creating log directory: ${LOG_DIR}"
  mkdir -p "${LOG_DIR}"
  chmod 0755 "${LOG_DIR}"
  say "OK: ${LOG_DIR} exists."
}

merge_daemon_json_nvidia_runtime() {
  say "Embedding NVIDIA runtime settings into ${DAEMON_JSON}..."

  local tmp
  tmp="$(mktemp)"

  # Read existing JSON if present/non-empty; otherwise start from {}
  if [[ -s "${DAEMON_JSON}" ]]; then
    cp -f "${DAEMON_JSON}" "${tmp}.orig"
  else
    echo '{}' > "${tmp}.orig"
  fi

  # Merge using python3 to preserve JSON validity.
  cmd_exists python3 || die "python3 is required to safely edit ${DAEMON_JSON}."

  python3 - <<'PY' "${tmp}.orig" "${tmp}"
import json, sys
src = sys.argv[1]
dst = sys.argv[2]

with open(src, "r", encoding="utf-8") as f:
    try:
        data = json.load(f)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON in {src}: {e}")

if not isinstance(data, dict):
    raise SystemExit(f"{src} must contain a JSON object at top-level.")

data.setdefault("runtimes", {})
if not isinstance(data["runtimes"], dict):
    raise SystemExit('"runtimes" exists but is not an object.')

data["runtimes"]["nvidia"] = {
    "path": "nvidia-container-runtime",
    "runtimeArgs": []
}
data["default-runtime"] = "nvidia"

with open(dst, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=False)
    f.write("\n")
PY

  # Install updated file (only if different)
  if [[ -f "${DAEMON_JSON}" ]] && cmp -s "${DAEMON_JSON}" "${tmp}"; then
    say "No change needed in ${DAEMON_JSON}."
  else
    install -m 0644 -o root -g root "${tmp}" "${DAEMON_JSON}"
    say "Updated ${DAEMON_JSON}. Restarting Docker..."
    systemctl restart docker || die "Failed to restart docker after updating daemon.json."
    systemctl is-active --quiet docker || die "docker is not active after restart."
    say "OK: Docker restarted."
  fi

  rm -f "${tmp}" "${tmp}.orig" || true
}

gpu_test_pytorch_in_docker() {
  if [[ "${SKIP_GPU_TEST}" == "1" ]]; then
    say "Skipping GPU test (SKIP_GPU_TEST=1)."
    return 0
  fi

  say "Checking NVIDIA GPU visibility on host..."
  if ! cmd_exists nvidia-smi; then
    die "nvidia-smi not found. Install NVIDIA driver on the host first."
  fi
  nvidia-smi >/dev/null || die "nvidia-smi failed. Host GPU driver may be broken."

  say "Testing GPU inside Docker using PyTorch image: ${PYTORCH_IMAGE}"
  # Requires NVIDIA Container Toolkit to be installed on host.
  # This test will pull a large image if not present.
  docker run --rm --gpus all "${PYTORCH_IMAGE}" \
    python - <<'PY'
import torch
print("torch.__version__ =", torch.__version__)
print("torch.cuda.is_available() =", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("GPU not available to PyTorch inside container.")
print("CUDA device count =", torch.cuda.device_count())
print("Device 0 =", torch.cuda.get_device_name(0))
PY

  say "OK: PyTorch can use GPU inside Docker."
}

join_swarm_if_needed() {
  say "Checking swarm status..."
  local state
  state="$(docker info --format '{{.Swarm.LocalNodeState}}' 2>/dev/null || echo "unknown")"

  if [[ "${state}" == "active" ]]; then
    say "This node is already part of a swarm (LocalNodeState=active). Skipping join."
    return 0
  fi

  say "Joining swarm at ${MANAGER_ADDR}..."
  docker swarm join --token "${JOIN_TOKEN}" "${MANAGER_ADDR}" \
    || die "docker swarm join failed."

  say "OK: swarm join succeeded."
  docker info --format 'Swarm: {{.Swarm.LocalNodeState}}, NodeID: {{.Swarm.NodeID}}'
}

main() {
  need_root
  check_uid_gid_1000
  check_docker_installed_running
  check_manager_port_2377
  ensure_firewall_ports
  make_log_dir
  merge_daemon_json_nvidia_runtime
  gpu_test_pytorch_in_docker
  join_swarm_if_needed
  say "All checks completed successfully."
}

main "$@"
