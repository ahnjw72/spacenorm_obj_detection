#! /bin/bash
# Usage: sh spacenorm_yolov7_training_docker.sh ahnjw72/spacenorm:latest yolov7_training ./doker_shared/yolov7_training.pt
root_dir=/opt/dlami/nvme/spacenorm_yolov7
docker_image_name=ahnjw72/spacenorm_without_trt:latest
training_name=yolov7_training
training_pt_file=yolov7-w6_training.pt
batch_size=16
epochs=6


# docker run --restart unless-stopped --gpus all -d -e TZ=Asia/Seoul -v ${root_dir}/runs:/workspace/spacenorm_yolov7/runs -v ${root_dir}/data/cctv_train_data:/workspace/spacenorm_yolov7/data/cctv_train_data -v ${root_dir}/docker_shared:/workspace/spacenorm_yolov7/docker_shared -it --name yolov7_training ${docker_image_name} /workspace/spacenorm_yolov7/spacenorm_yolov7_training_in_docker.sh ${training_name} ${training_pt_file}
docker run --restart unless-stopped --gpus all -d -e TZ=Asia/Seoul --shm-size=8g -v ${root_dir}/runs:/workspace/spacenorm_yolov7/runs -v ${root_dir}/data/cctv_train_data:/workspace/spacenorm_yolov7/data/cctv_train_data -v ${root_dir}/docker_shared:/workspace/spacenorm_yolov7/docker_shared -it --name yolov7_training ${docker_image_name}
