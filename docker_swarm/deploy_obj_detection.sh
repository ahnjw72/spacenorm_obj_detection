#!/usr/bin/env bash
# After running this script, verify deployment with:
#  $ docker service ls
#  $ docker service ps spacenorm_detector
#  $ docker service logs -f spacenorm_detector
#
# Remove the container on the worker node
#  $ docker service rm spacenorm_detector
#
# Remove the whole stack
#  $ docker stack rm spacenorm
# 
# Restart the stack
# $ docker service update --force spacenorm_detector

set -euo pipefail

SERVICE_GROUP1=("cym" "kumho" "kumho_car" "yujin" "jaeil_io")
SERVICE_GROUP2=("jaeil_cr")

if [ $# -eq 0 ]; then
    echo "Usage: $0 [service_name]"
    exit 1
fi

export SERVICE_NAME=$1

# Validate server name against known groups
if ! [[ " ${SERVICE_GROUP1[*]} ${SERVICE_GROUP2[*]} " =~ " ${SERVICE_NAME} " ]]; then
    echo "Error: Server '$SERVICE_NAME' is not in any known group."
    echo "Valid servers in GROUP1: ${SERVICE_GROUP1[*]}"
    echo "Valid servers in GROUP2: ${SERVICE_GROUP2[*]}"
    exit 1
fi

AWS_REGION="ap-northeast-2"
#IMAGE_TAG="latest"
IMAGE_TAG="cu128"

LOGIN_PASSWORD=$(aws ecr get-login-password --region "${AWS_REGION}")
if [ $? -ne 0 ]; then
    echo "Failed to retrieve ECR login password for ${AWS_REGION}" >&2
    exit 1
fi

echo "$LOGIN_PASSWORD" | docker login --username AWS --password-stdin 159552820182.dkr.ecr.${AWS_REGION}.amazonaws.com
if [ $? -ne 0 ]; then
    echo "Docker login to ECR failed" >&2
    exit 1
fi
# echo "ECR login succeeded"

if [[ " ${SERVICE_GROUP1[*]} " =~ " ${SERVICE_NAME} " ]]; then
    REPOSITORY_NAME="spacenorm_obj_detection"
    IMG_DIGEST=$(aws ecr describe-images \
      --region "${AWS_REGION}" \
      --repository-name "$REPOSITORY_NAME" \
      --image-ids imageTag="$IMAGE_TAG" \
      --query 'imageDetails[0].imageDigest' \
      --output text)
    export DOCKER_IMAGE="159552820182.dkr.ecr."${AWS_REGION}".amazonaws.com/${REPOSITORY_NAME}@${IMG_DIGEST}"
    #export DOCKER_IMAGE="ahnjw72/spacenorm_obj_detection:cu128"
    echo "DOCKER_IMAGE is set to: ${DOCKER_IMAGE}"
elif [[ " ${SERVICE_GROUP2[*]} " =~ " ${SERVICE_NAME} " ]]; then
    # export DOCKER_IMAGE="ahnjw72/spacenorm_obj_detection_blackwell:latest"

    # REPOSITORY_NAME="spacenorm_obj_detection_blackwell"
    # IMAGE_TAG="latest"

    REPOSITORY_NAME="spacenorm_obj_detection"

    IMG_DIGEST=$(aws ecr describe-images \
      --region "${AWS_REGION}" \
      --repository-name "$REPOSITORY_NAME" \
      --image-ids imageTag="$IMAGE_TAG" \
      --query 'imageDetails[0].imageDigest' \
      --output text)
    export DOCKER_IMAGE="159552820182.dkr.ecr."${AWS_REGION}".amazonaws.com/${REPOSITORY_NAME}@${IMG_DIGEST}"
    #export DOCKER_IMAGE="ahnjw72/spacenorm_obj_detection:cu128"
    echo "DOCKER_IMAGE is set to: ${DOCKER_IMAGE}"
else
    echo "Error: '${SERVICE_NAME}' is not a valid server name for spacenorm_obj_detection service."
    exit 1
fi

CFG_BASE_DIR="/home/cym/Work/spacenorm_obj_detection/spacenorm_cfg"
export CFG_BASE_DIR_IN_CONTAINER="/app/spacenorm_obj_detection/spacenorm_cfg"
STACK_NAME="spacenorm_obj_detection"
TEMPLATE="stack.yml.template"
RENDERED="stack_${SERVICE_NAME}.yml"

# Configuration files for a server ========================================
# export SERVER="${SERVICE_NAME}"
DEFAULT_CFG="${CFG_BASE_DIR}/behavior/default.json"
OVERRIDE_CFG="${CFG_BASE_DIR}/behavior/overrides/${SERVICE_NAME}.json"
CCTV_CFG="${CFG_BASE_DIR}/cctv/cctv_${SERVICE_NAME}.json"
CCTV_KEY="${CFG_BASE_DIR}/cctv/cctv_${SERVICE_NAME}.keys"
# ==========================================================================

# Compute hash for each config file
DEFAULT_CFG_HASH="$(sha256sum "${DEFAULT_CFG}" | awk '{print substr($1,1,12)}')"
export DEFAULT_CFG_NAME="default_cfg_${DEFAULT_CFG_HASH}"

OVERRIDE_CFG_HASH="$(sha256sum "${OVERRIDE_CFG}" | awk '{print substr($1,1,12)}')"
export OVERRIDE_CFG_NAME="${SERVICE_NAME}_override_cfg_${OVERRIDE_CFG_HASH}"

CCTV_CFG_HASH="$(sha256sum "${CCTV_CFG}" | awk '{print substr($1,1,12)}')"
export CCTV_CFG_NAME="${SERVICE_NAME}_cctv_cfg_${CCTV_CFG_HASH}"

CCTV_KEY_HASH="$(sha256sum "${CCTV_KEY}" | awk '{print substr($1,1,12)}')"
export CCTV_KEY_NAME="${SERVICE_NAME}_cctv_key_${CCTV_KEY_HASH}"
# ==========================================================================

# Create config objects only if missing ============================================
if ! docker config inspect "${DEFAULT_CFG_NAME}" >/dev/null 2>&1; then
  docker config create "${DEFAULT_CFG_NAME}" "${DEFAULT_CFG}" >/dev/null
  echo "[OK] Created config: ${DEFAULT_CFG_NAME}"
else
  echo "[OK] Config exists: ${DEFAULT_CFG_NAME}"
fi

if ! docker config inspect "${OVERRIDE_CFG_NAME}" >/dev/null 2>&1; then
    docker config create "${OVERRIDE_CFG_NAME}" "${OVERRIDE_CFG}" >/dev/null
    echo "[OK] Created config: ${OVERRIDE_CFG_NAME}"
else
    echo "[OK] Config exists: ${OVERRIDE_CFG_NAME}"
fi

if ! docker config inspect "${CCTV_CFG_NAME}" >/dev/null 2>&1; then
  docker config create "${CCTV_CFG_NAME}" "${CCTV_CFG}" >/dev/null
  echo "[OK] Created config: ${CCTV_CFG_NAME}"
else
  echo "[OK] Config exists: ${CCTV_CFG_NAME}"
fi

if ! docker config inspect "${CCTV_KEY_NAME}" >/dev/null 2>&1; then
  docker config create "${CCTV_KEY_NAME}" "${CCTV_KEY}" >/dev/null
  echo "[OK] Created config: ${CCTV_KEY_NAME}"
else
  echo "[OK] Config exists: ${CCTV_KEY_NAME}"
fi

# ==========================================================================

# Special service for car detection for "kumho" --> this is temporary before api based config retrieval is implemented
if [ "${SERVICE_NAME}" == "kumho_car" ]; then
    export SERVER="kumho"
    export LOG_DIR_NAME="spacenorm_obj_detection_car"
    export PORT1="8082"
    export PORT2="9003"
else
    export SERVER="${SERVICE_NAME}"
    export LOG_DIR_NAME="spacenorm_obj_detection"
    export PORT1="8081"
    export PORT2="9000"
fi

envsubst < "${TEMPLATE}" > "${RENDERED}"

# The stack bind-mounts /var/lib/spacenorm_obj_detection on each worker node
# for caching TensorRT engine files. Docker Swarm requires the source path to
# exist on the host before the container starts.
# Run this once on every worker node if the directory is not yet present:
#   sudo mkdir -p /var/lib/spacenorm_obj_detection
echo "[INFO] Ensure /var/lib/spacenorm_obj_detection exists on each worker node."
echo "       If not, run on the worker: sudo mkdir -p /var/lib/spacenorm_obj_detection"

# docker stack deploy -c "${RENDERED}" "${STACK_NAME}"
docker stack deploy --with-registry-auth -c "${RENDERED}" "${STACK_NAME}"

echo "[OK] Deployed stack '${STACK_NAME}'"
