#! /bin/bash
set -e

echo "============================================"
echo "[1] Checking spacenorm_obj_detection local clone..."
echo "============================================"

if [ ! -d "spacenorm_obj_detection" ]; then
    echo "→ spacenorm_obj_detection not found. Cloning..."
    git clone git@github.com:ahnjw72/spacenorm_obj_detection.git
else
    echo "→ spacenorm_obj_detection already exists. Pulling latest changes..."
    (cd spacenorm_obj_detection && git pull)
fi

# spacenorm_cfg is injected at runtime by Docker Swarm
rm -rf ./spacenorm_obj_detection/spacenorm_cfg

echo
echo "============================================"
echo "[2] Copying model files..."
echo "============================================"

# Dereference the symlink so docker COPY gets the actual file, not a broken link.
cp -L spacenorm_obj_detection.pt spacenorm_obj_detection_build.pt

echo "→ Model file copied (dereferenced)"

echo
echo "============================================"
echo "[3] Starting Docker build"
echo "============================================"

#DOCKER_REPOSITORY_TAG='spacenorm_obj_detection:latest'
DOCKER_REPOSITORY_TAG='spacenorm_obj_detection:cu128'
DOCKER_IMAGE_NAME_DOCKER_HUB="ahnjw72/$DOCKER_REPOSITORY_TAG"
DOCKER_IMAGE_NAME_AWS_ECR="159552820182.dkr.ecr.ap-northeast-2.amazonaws.com/$DOCKER_REPOSITORY_TAG"

docker build -f Dockerfile -t $DOCKER_IMAGE_NAME_DOCKER_HUB .
docker tag $DOCKER_IMAGE_NAME_DOCKER_HUB $DOCKER_IMAGE_NAME_AWS_ECR

rm -f spacenorm_obj_detection_build.pt

echo
echo "=========================================================="
echo "[DONE] Docker image built successfully!"
echo "Image name for Docker Hub : $DOCKER_IMAGE_NAME_DOCKER_HUB"
echo "Image name for AWS ECR    : $DOCKER_IMAGE_NAME_AWS_ECR"
echo "=========================================================="
