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
# yolo11x_cctv.pt / yolo11x_coco.pt already exist here as symlinks; if cp writes
# through an existing destination symlink it overwrites whatever the symlink
# points to (verified: GNU cp -L follows an existing dest symlink and clobbers
# its target in place). yolo11x_cctv.pt currently resolves into the sibling
# yolov11_training_aws training-results directory, so removing the symlinks
# first is required to avoid overwriting that file.
rm -f ./yolo11x_cctv.pt ./yolo11x_coco.pt
cp -L ./yolo_weights/yolo11x_set01-0151.pt ./yolo11x_cctv.pt
cp -L ./yolo_weights/yolo11x.pt ./yolo11x_coco.pt

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

# Remove the dereferenced build-context copies created in step [2] (~114MB each)
rm -f ./yolo11x_cctv.pt ./yolo11x_coco.pt

echo
echo "=========================================================="
echo "[DONE] Docker image built successfully!"
echo "Image name for Docker Hub : $DOCKER_IMAGE_NAME_DOCKER_HUB"
echo "Image name for AWS ECR    : $DOCKER_IMAGE_NAME_AWS_ECR"
echo "=========================================================="
