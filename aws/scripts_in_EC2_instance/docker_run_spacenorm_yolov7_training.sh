#! /bin/bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 {g5.xlarge|g5.12xlarge}"
  exit 1
fi

INSTANCE_TYPE="$1"
if [ "${INSTANCE_TYPE}" != "g5.xlarge" ] && [ "${INSTANCE_TYPE}" != "g5.12xlarge" ]; then
  echo "Error: Invalid instance type. Please choose either 'g5.xlarge' or 'g5.12xlarge'."
  exit 1
fi

workers=${workers:-8}
batch_size=${batch_size:-8}
epochs=${epochs:-50}
hyperparameter_file=${hyperparameter_file:-data/hyp.scratch.p6.yaml}
cfg_file=${cfg_file:-cfg/training/yolov7-w6-cctv.yaml}
data_file=${data_file:-data/cctv.yaml}
training_name=${training_name:-yolov7_w6_cctv}
training_pt_file=${training_pt_file:-yolov7-w6_training.pt}
log_file=${log_file:-docker_shared/train.log}
root_dir=${root_dir:-/mnt/spacenorm_data/spacenorm_yolov7}
container_name=${container_name:-yolov7_training}
docker_image_name=${docker_image_name:-159552820182.dkr.ecr.ap-northeast-2.amazonaws.com/spacenorm_without_trt:latest}
docker_workspace_dir=${docker_workspace_dir:-/workspace/spacenorm_yolov7}

case "$INSTANCE_TYPE" in
  g5.xlarge)
    GPUS=1
    DEVICE="0"
    LAUNCHER="python"
    shm_size=8g
    ;;
  g5.12xlarge)
    GPUS=4
    DEVICE="0,1,2,3"
    # LAUNCHER="torchrun --nproc_per_node=4"  <-- torchrun is not compatible with older PyTorch versions
    LAUNCHER="python -m torch.distributed.launch --nproc_per_node=$GPUS"
    workers=16
    batch_size=128
    shm_size=32g
    ;;
  *)
    echo "ERROR: Unsupported instance type: $INSTANCE_TYPE"
    echo "Supported values: g5.xlarge, g5.12xlarge"
    exit 1
    ;;
esac

# -----------------------------
# Info output
# -----------------------------
echo "========================================"
echo " Instance type : $INSTANCE_TYPE"
echo " Num GPUs      : $GPUS"
echo " GPU IDs       : $DEVICE"
echo " Shared Mem    : $shm_size"
echo " Launcher      : $LAUNCHER"
echo " Batch size    : $batch_size (TOTAL)"
echo "========================================"

command_to_run="cd ${docker_workspace_dir} && ${LAUNCHER} train_aux.py --workers ${workers} --device ${DEVICE} --batch-size ${batch_size} --epochs ${epochs} --hyp ${hyperparameter_file} --cfg ${cfg_file} --data ${data_file} --name ${training_name} --weights ${training_pt_file} 1> /dev/null 2> ${log_file}"

docker_run_command="docker run --gpus all -d -e TZ=Asia/Seoul --shm-size=${shm_size} -v ${root_dir}/runs:${docker_workspace_dir}/runs -v ${root_dir}/data:${docker_workspace_dir}/data -v ${root_dir}/docker_shared:${docker_workspace_dir}/docker_shared -it --name ${container_name} ${docker_image_name} bash -c \"${command_to_run}\""

echo "Docker run command: ${docker_run_command}"

eval ${docker_run_command}