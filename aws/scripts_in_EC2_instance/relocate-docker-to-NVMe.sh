#!/usr/bin/env bash
set -euo pipefail

# ===== Configuration =====
NVME_MOUNT="/opt/dlami/nvme"
DOCKER_ROOT="${NVME_MOUNT}/docker"
CONTAINERD_ROOT="${NVME_MOUNT}/containerd"

DOCKER_CONFIG="/etc/docker/daemon.json"
CONTAINERD_OVERRIDE="/etc/systemd/system/containerd.service.d/override.conf"

echo "=== Docker + containerd NVMe Relocation Script ==="

# 1. Must be root
if [[ $EUID -ne 0 ]]; then
  echo "ERROR: Run this script with sudo or as root."
  exit 1
fi

# 2. Verify NVMe mount
if ! mountpoint -q "$NVME_MOUNT"; then
  echo "ERROR: $NVME_MOUNT is not a mounted filesystem."
  echo "       Mount your NVMe disk first."
  exit 1
fi

echo "NVMe mount verified at $NVME_MOUNT"
df -h "$NVME_MOUNT"

# 3. Stop services
echo "Stopping Docker and containerd..."
systemctl stop docker || true
systemctl stop containerd || true

# 4. Prepare directories
echo "Preparing directories..."
mkdir -p "$DOCKER_ROOT" "$CONTAINERD_ROOT"
chmod 711 "$DOCKER_ROOT" "$CONTAINERD_ROOT"

# 5. Migrate Docker data
if [[ -d /var/lib/docker && "$(ls -A /var/lib/docker)" ]]; then
  echo "Migrating Docker data..."
  rsync -aHAX --numeric-ids /var/lib/docker/ "$DOCKER_ROOT/"
else
  echo "No Docker data to migrate."
fi

# 6. Migrate containerd data
if [[ -d /var/lib/containerd && "$(ls -A /var/lib/containerd)" ]]; then
  echo "Migrating containerd data..."
  rsync -aHAX --numeric-ids /var/lib/containerd/ "$CONTAINERD_ROOT/"
else
  echo "No containerd data to migrate."
fi

# 7. Configure Docker daemon
echo "Configuring Docker daemon..."
mkdir -p /etc/docker
cat > "$DOCKER_CONFIG" <<EOF
{
  "data-root": "$DOCKER_ROOT"
}
EOF

# 8. Configure containerd systemd override
echo "Configuring containerd systemd override..."
mkdir -p "$(dirname "$CONTAINERD_OVERRIDE")"
cat > "$CONTAINERD_OVERRIDE" <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/containerd --root $CONTAINERD_ROOT --state /run/containerd
EOF

# 9. Reload systemd and restart services
echo "Reloading systemd and restarting services..."
systemctl daemon-reexec
systemctl daemon-reload
systemctl start containerd
systemctl start docker

# 10. Verification
echo "=== Verification ==="
docker info | grep "Docker Root Dir"

echo "containerd process:"
ps aux | grep containerd | grep root || true

# 11. Cleanup old directories (safe after verification)
echo "Cleaning up old directories..."
rm -rf /var/lib/docker
rm -rf /var/lib/containerd

echo "=== SUCCESS: Docker and containerd are now using NVMe ==="
