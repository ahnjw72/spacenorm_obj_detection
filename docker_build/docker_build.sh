#! /bin/bash
set -e

echo "============================================"
echo "[1] Checking spacenorm_yolov7 local clone..."
echo "============================================"

if [ ! -d "spacenorm_yolov7" ]; then
    echo "→ spacenorm_yolov7 not found. Cloning..."
    git clone git@github.com:ahnjw72/spacenorm_yolov7.git
else
    echo "→ spacenorm_yolov7 already exists. Pulling latest changes..."
    (cd spacenorm_yolov7 && git pull)
fi

# spacenorm_cfg is injected at runtime by Docker Swarm
rm -rf ./spacenorm_yolov7/spacenorm_cfg

echo
echo "============================================"
echo "[2] Copying model files..."
echo "============================================"

cp -L ./yolov7_weights/spacenorm_yolov7.pt .
cp -L ./yolov7_weights/yolov7-w6.pt .

echo "→ Model files copied"

echo
echo "============================================"
echo "[3] Starting Docker build"
echo "============================================"

#DOCKER_REPOSITORY_TAG='spacenorm_yolov7:latest'
DOCKER_REPOSITORY_TAG='spacenorm_yolov7:cu128'
DOCKER_IMAGE_NAME_DOCKER_HUB="ahnjw72/$DOCKER_REPOSITORY_TAG"
DOCKER_IMAGE_NAME_AWS_ECR="159552820182.dkr.ecr.ap-northeast-2.amazonaws.com/$DOCKER_REPOSITORY_TAG"

docker build -f Dockerfile -t $DOCKER_IMAGE_NAME_DOCKER_HUB .
docker tag $DOCKER_IMAGE_NAME_DOCKER_HUB $DOCKER_IMAGE_NAME_AWS_ECR

echo
echo "=========================================================="
echo "[DONE] Docker image built successfully!"
echo "Image name for Docker Hub : $DOCKER_IMAGE_NAME_DOCKER_HUB"
echo "Image name for AWS ECR    : $DOCKER_IMAGE_NAME_AWS_ECR"
echo "=========================================================="
