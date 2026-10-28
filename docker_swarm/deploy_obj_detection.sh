#!/usr/bin/env bash
# After running this script, verify deployment with:
#  $ docker stack services spacenorm_obj_detection
#  $ docker service ps spacenorm_obj_detection_<service_name>
#  $ docker service logs -f spacenorm_obj_detection_<service_name>
#
# Remove one service from the stack
#  $ docker service rm spacenorm_obj_detection_<service_name>
#
# Remove the whole stack
#  $ docker stack rm spacenorm_obj_detection
#
# Restart a service
#  $ docker service update --force spacenorm_obj_detection_<service_name>

set -euo pipefail

VALID_SERVICES=("cym" "kumho" "kumho_car" "yujin" "jaeil_io" "jaeil_cr")

if [ $# -eq 0 ]; then
    echo "Usage: $0 <service_name|all>"
    echo "Valid services: ${VALID_SERVICES[*]}"
    exit 1
fi

TARGET=$1

if [ "${TARGET}" == "all" ]; then
    SERVICES_TO_DEPLOY=("${VALID_SERVICES[@]}")
elif [[ " ${VALID_SERVICES[*]} " =~ " ${TARGET} " ]]; then
    SERVICES_TO_DEPLOY=("${TARGET}")
else
    echo "Error: Server '${TARGET}' is not in VALID_SERVICES."
    echo "Valid services: ${VALID_SERVICES[*]}"
    exit 1
fi

AWS_REGION="ap-northeast-2"
IMAGE_TAG="cu128"
REPOSITORY_NAME="spacenorm_obj_detection"

LOGIN_PASSWORD=$(aws ecr get-login-password --region "${AWS_REGION}") || {
    echo "Failed to retrieve ECR login password for ${AWS_REGION}" >&2
    exit 1
}

echo "${LOGIN_PASSWORD}" | docker login --username AWS --password-stdin "159552820182.dkr.ecr.${AWS_REGION}.amazonaws.com" || {
    echo "Docker login to ECR failed" >&2
    exit 1
}

IMG_DIGEST=$(aws ecr describe-images \
  --region "${AWS_REGION}" \
  --repository-name "${REPOSITORY_NAME}" \
  --image-ids imageTag="${IMAGE_TAG}" \
  --query 'imageDetails[0].imageDigest' \
  --output text)
export DOCKER_IMAGE="159552820182.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPOSITORY_NAME}@${IMG_DIGEST}"
echo "[INFO] DOCKER_IMAGE is set to: ${DOCKER_IMAGE}"

CFG_BASE_DIR="/home/ahnjw/Work/spacenorm_obj_detection/spacenorm_cfg"
export CFG_BASE_DIR_IN_CONTAINER="/app/spacenorm_obj_detection/spacenorm_cfg"
STACK_NAME="spacenorm_obj_detection"
TEMPLATE="stack.yml.template"

# Computes a content-hash config name, creates the docker config if missing,
# and prints the resulting name on stdout (progress lines go to stderr so
# they don't pollute the captured name).
register_config() {
    local name_prefix=$1 file_path=$2
    local hash name
    hash="$(sha256sum "${file_path}" | awk '{print substr($1,1,12)}')"
    name="${name_prefix}_${hash}"
    if ! docker config inspect "${name}" >/dev/null 2>&1; then
        docker config create "${name}" "${file_path}" >/dev/null
        echo "[OK] Created config: ${name}" >&2
    else
        echo "[OK] Config exists: ${name}" >&2
    fi
    echo "${name}"
}

deploy_service() {
    local service_name=$1
    export SERVICE_NAME="${service_name}"
    local rendered="stack_${service_name}.yml"

    local default_cfg="${CFG_BASE_DIR}/behavior/default.json"
    local override_cfg="${CFG_BASE_DIR}/behavior/overrides/${service_name}.json"
    local cctv_cfg="${CFG_BASE_DIR}/cctv/cctv_${service_name}.json"
    local cctv_key="${CFG_BASE_DIR}/cctv/cctv_${service_name}.keys"

    export DEFAULT_CFG_NAME CCTV_CFG_NAME CCTV_KEY_NAME OVERRIDE_CFG_NAME
    DEFAULT_CFG_NAME=$(register_config "default_cfg" "${default_cfg}")
    OVERRIDE_CFG_NAME=$(register_config "${service_name}_override_cfg" "${override_cfg}")
    CCTV_CFG_NAME=$(register_config "${service_name}_cctv_cfg" "${cctv_cfg}")
    CCTV_KEY_NAME=$(register_config "${service_name}_cctv_key" "${cctv_key}")

    # Special service for car detection for "kumho" --> this is temporary
    # before api based config retrieval is implemented
    if [ "${service_name}" == "kumho_car" ]; then
        export SERVER="kumho"
        export LOG_DIR_NAME="spacenorm_obj_detection_car"
        export PORT1="8082"
        export PORT2="9003"
    else
        export SERVER="${service_name}"
        export LOG_DIR_NAME="spacenorm_obj_detection"
        export PORT1="8081"
        export PORT2="9000"
    fi

    # Derive SPACENORM_MODEL_PT / SPACENORM_IMGSZ / SPACENORM_USE_TENSORRT from
    # the server override config (falling back to default.json) so servers
    # with a different model, resolution, or TensorRT setting (e.g. jaeil_cr
    # uses yolo11x.pt at imgsz 1280; kumho opts out of TensorRT) get the
    # correct TRT engine built.
    local cfg_values
    mapfile -t cfg_values < <(python3 - "${override_cfg}" "${default_cfg}" <<'PY'
import json
import sys


def load(path):
    try:
        return json.load(open(path))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


override, default = load(sys.argv[1]), load(sys.argv[2])

model = get(override, 'detector', 'model', 'value') or get(default, 'detector', 'model', 'value')
if not model:
    raise SystemExit('Could not determine model path from config files')

imgsz = get(override, 'detector', 'img_size', 'value') or get(default, 'detector', 'img_size', 'value')
if not imgsz:
    raise SystemExit('Could not determine img_size from config files')

use_trt = get(override, 'detector', 'use_tensorrt', 'value')
if use_trt is None:
    use_trt = get(default, 'detector', 'use_tensorrt', 'value')
if use_trt is None:
    use_trt = True

print(model)
print(imgsz)
print(str(use_trt).lower())
PY
)
    export SPACENORM_MODEL_PT="/app/${cfg_values[0]}"
    export SPACENORM_IMGSZ="${cfg_values[1]}"
    export SPACENORM_USE_TENSORRT="${cfg_values[2]}"
    echo "[INFO] ${service_name}: SPACENORM_MODEL_PT=${SPACENORM_MODEL_PT} SPACENORM_IMGSZ=${SPACENORM_IMGSZ} SPACENORM_USE_TENSORRT=${SPACENORM_USE_TENSORRT}"

    envsubst < "${TEMPLATE}" > "${rendered}"
    docker stack deploy --with-registry-auth -c "${rendered}" "${STACK_NAME}"
    echo "[OK] Deployed service '${service_name}' to stack '${STACK_NAME}'"
}

# The stack bind-mounts /var/lib/spacenorm_obj_detection on each worker node
# for caching TensorRT engine files. Docker Swarm requires the source path to
# exist on the host before the container starts.
# Run this once on every worker node if the directory is not yet present:
#   sudo mkdir -p /var/lib/spacenorm_obj_detection
echo "[INFO] Ensure /var/lib/spacenorm_obj_detection exists on each worker node."
echo "       If not, run on the worker: sudo mkdir -p /var/lib/spacenorm_obj_detection"

DEPLOYED_SERVICES=()
FAILED_SERVICES=()

for svc in "${SERVICES_TO_DEPLOY[@]}"; do
    echo "=== Deploying ${svc} ==="
    # Each service deploys in its own subshell so a failure partway through
    # (e.g. a missing override config) aborts only that service's deploy
    # instead of leaving it half-configured. `set -e` is toggled off in the
    # parent around the call (not inside an `if`/`&&` condition, where bash
    # silently ignores -e even inside a nested subshell) so the subshell's
    # failure is captured instead of aborting the rest of the loop.
    set +e
    ( set -euo pipefail; deploy_service "${svc}" )
    status=$?
    set -e
    if [ "${status}" -eq 0 ]; then
        DEPLOYED_SERVICES+=("${svc}")
    else
        echo "[ERROR] Failed to deploy '${svc}' (exit ${status})." >&2
        FAILED_SERVICES+=("${svc}")
    fi
done

echo
echo "[SUMMARY] Deployed: ${DEPLOYED_SERVICES[*]:-(none)}"
if [ "${#FAILED_SERVICES[@]}" -gt 0 ]; then
    echo "[SUMMARY] Failed:   ${FAILED_SERVICES[*]}" >&2
    exit 1
fi
