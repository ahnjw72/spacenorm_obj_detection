#!/usr/bin/env bash
set -euo pipefail

############################################
# CONFIG (override via environment variables)
############################################
AWS_REGION="${AWS_REGION:-ap-northeast-2}"
AZ="${AZ:-ap-northeast-2a}"

# EBS identity
VOL_NAME="${VOL_NAME:-spacenorm-dataset}"

# EBS create parameters (USED ONLY ON FIRST CREATION)
VOL_TYPE="${VOL_TYPE:-gp3}"
SIZE_GB="${SIZE_GB:-150}"     # Default: 150GB

# Mounting
MOUNT_POINT="${MOUNT_POINT:-/mnt/spacenorm_data}"
OWNER_USER="${OWNER_USER:-ubuntu}"
OWNER_GROUP="${OWNER_GROUP:-ubuntu}"

# Nitro device hint
ATTACH_DEVICE="${ATTACH_DEVICE:-/dev/sdf}"

# Filesystem safety
FSTYPE="ext4"
SENTINEL_FILE=".DO_NOT_FORMAT"

############################################
# S3 DATASET SYNC CONFIG
############################################
S3_DATA_URI="s3://spacenorm-cctv-ml-data/spacenorm_yolov7/data/"
LOCAL_DATA_DIR="${MOUNT_POINT}/spacenorm_yolov7/data"

############################################
# Helpers
############################################
need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing command: $1" >&2; exit 1; }; }

imds() {
  local path="$1"
  local token
  token="$(curl -fsS -m 2 -X PUT http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' || true)"
  if [[ -n "${token:-}" ]]; then
    curl -fsS -m 2 -H "X-aws-ec2-metadata-token: $token" \
      "http://169.254.169.254/latest/${path}"
  else
    curl -fsS -m 2 "http://169.254.169.254/latest/${path}"
  fi
}

wait_for_volume_state() {
  local vol_id="$1"
  local wanted="$2"
  while true; do
    local st
    st="$(aws ec2 describe-volumes \
      --region "$AWS_REGION" \
      --volume-ids "$vol_id" \
      --query 'Volumes[0].State' \
      --output text)"
    [[ "$st" == "$wanted" ]] && break
    sleep 2
  done
}

wait_for_attachment() {
  local vol_id="$1"
  local inst_id="$2"
  while true; do
    local st
    st="$(aws ec2 describe-volumes \
      --region "$AWS_REGION" \
      --volume-ids "$vol_id" \
      --query "Volumes[0].Attachments[?InstanceId=='${inst_id}'].State | [0]" \
      --output text)"
    [[ "$st" == "attached" ]] && break
    sleep 2
  done
}

ensure_nvme_cli() {
  command -v nvme >/dev/null 2>&1 && return
  sudo apt-get update -y >/dev/null
  sudo apt-get install -y nvme-cli >/dev/null
}

find_nvme_by_volume_id() {
  local vol_id="$1"
  local short="${vol_id#vol-}"

  ensure_nvme_cli

  for _ in {1..40}; do
    local dev
    dev="$(sudo nvme list 2>/dev/null | awk -v v="$short" 'tolower($0) ~ tolower(v) {print $1; exit}')"
    [[ -n "${dev:-}" ]] && { echo "$dev"; return; }
    sleep 2
  done

  echo "ERROR: NVMe device not found for $vol_id" >&2
  sudo nvme list >&2
  exit 1
}

has_fs() {
  sudo blkid "$1" >/dev/null 2>&1
}

ensure_fstab() {
  local dev="$1"
  local mp="$2"
  local uuid
  uuid="$(sudo blkid -s UUID -o value "$dev")"
  sudo mkdir -p "$mp"
  sudo grep -q "$uuid" /etc/fstab || \
    echo "UUID=$uuid $mp $FSTYPE defaults,nofail 0 2" | sudo tee -a /etc/fstab >/dev/null
}

############################################
# Preconditions
############################################
need aws
need curl
need awk

INSTANCE_ID="$(imds meta-data/instance-id)"

echo "Region:     $AWS_REGION"
echo "AZ:         $AZ"
echo "VolumeTag:  $VOL_NAME"
echo "Mount:      $MOUNT_POINT"
echo "InstanceId: $INSTANCE_ID"
echo

############################################
# 1) Find or create EBS volume
############################################
VOLUME_ID="$(aws ec2 describe-volumes \
  --region "$AWS_REGION" \
  --filters \
    "Name=tag:Name,Values=${VOL_NAME}" \
    "Name=availability-zone,Values=${AZ}" \
    "Name=status,Values=available,in-use" \
  --query 'Volumes[].VolumeId | [0]' \
  --output text)"

if [[ "$VOLUME_ID" == "None" || -z "$VOLUME_ID" ]]; then
  echo "No existing volume found. Creating EBS volume (${SIZE_GB}GB)..."
  VOLUME_ID="$(aws ec2 create-volume \
    --region "$AWS_REGION" \
    --availability-zone "$AZ" \
    --size "$SIZE_GB" \
    --volume-type "$VOL_TYPE" \
    --tag-specifications \
      "ResourceType=volume,Tags=[{Key=Name,Value=${VOL_NAME}}]" \
    --query 'VolumeId' --output text)"
  echo "Created volume: $VOLUME_ID"
else
  echo "Reusing existing volume: $VOLUME_ID"
fi

############################################
# 2) Attach if needed
############################################
ATTACHED="$(aws ec2 describe-volumes \
  --region "$AWS_REGION" \
  --volume-ids "$VOLUME_ID" \
  --query "Volumes[0].Attachments[?InstanceId=='${INSTANCE_ID}'].State | [0]" \
  --output text)"

if [[ "$ATTACHED" != "attached" ]]; then
  wait_for_volume_state "$VOLUME_ID" "available"
  aws ec2 attach-volume \
    --region "$AWS_REGION" \
    --volume-id "$VOLUME_ID" \
    --instance-id "$INSTANCE_ID" \
    --device "$ATTACH_DEVICE" >/dev/null
  wait_for_attachment "$VOLUME_ID" "$INSTANCE_ID"
else
  echo "Volume already attached."
fi

############################################
# 3) NVMe mapping
############################################
DEV_PATH="$(find_nvme_by_volume_id "$VOLUME_ID")"
echo "Mapped $VOLUME_ID -> $DEV_PATH"

############################################
# 4) SAFE filesystem handling
############################################
sudo mkdir -p "$MOUNT_POINT"

if has_fs "$DEV_PATH"; then
  echo "Filesystem detected on $DEV_PATH"
else
  if [[ -e "${MOUNT_POINT}/${SENTINEL_FILE}" ]]; then
    echo "ERROR: Sentinel exists but filesystem missing — refusing to format." >&2
    exit 1
  fi
  echo "Blank volume detected — creating $FSTYPE filesystem"
  sudo mkfs.ext4 -F "$DEV_PATH"
fi

############################################
# 5) Mount + persist
############################################
mount | grep -q "$MOUNT_POINT" || sudo mount "$DEV_PATH" "$MOUNT_POINT"
ensure_fstab "$DEV_PATH" "$MOUNT_POINT"
sudo chown -R "${OWNER_USER}:${OWNER_GROUP}" "$MOUNT_POINT"

############################################
# 6) Create filesystem sentinel (once)
############################################
if [[ ! -e "${MOUNT_POINT}/${SENTINEL_FILE}" ]]; then
  touch "${MOUNT_POINT}/${SENTINEL_FILE}"
  echo "Filesystem sentinel created: ${MOUNT_POINT}/${SENTINEL_FILE}"
fi

############################################
# 7) S3 → EBS DATASET SYNC (ALWAYS, WITH PROGRESS)
############################################
if [ $# -eq 0 ]; then
  PARAM=""
else
  PARAM="$1"
fi

if [ "$PARAM" == "--sync" ]; then
  echo
  echo "Synchronizing dataset from S3 (progress enabled)..."
  echo "  Source:      ${S3_DATA_URI}"
  echo "  Destination: ${LOCAL_DATA_DIR}"

  sudo -u "${OWNER_USER}" mkdir -p "${LOCAL_DATA_DIR}"

  aws s3 sync \
    "${S3_DATA_URI}" \
    "${LOCAL_DATA_DIR}" \

  echo "S3 dataset sync completed."
else
  echo
  echo "Skipping S3 dataset sync. To enable, run the script with the --sync parameter."
fi

############################################
# Done
############################################
echo
echo "SUCCESS"
echo "VolumeId:    $VOLUME_ID"
echo "Device:      $DEV_PATH"
echo "Mounted at:  $MOUNT_POINT"
echo "Dataset dir: $LOCAL_DATA_DIR"
