# some more ls aliases
alias ll='ls -alFh'
alias la='ls -A'
alias l='ls -CF'

alias v.c='vi ~/.bashrc'
alias s.c='source ~/.bashrc'
alias v.a='vi ~/.bash_aliases'
alias s.a='source ~/.bash_aliases'
#alias s.t='conda activate TensorRT'
alias s.t='conda activate TensorRT'
#alias s.y='conda activate yolov7; cd /home/cym/Work/yolov7'
alias s.y='conda activate trt_y7; cd /home/cym/Work/spacenorm_yolov7'
alias s.m='conda activate SmartMirror; cd /home/cym/Work/Smart-Mirror'
alias s.h='conda activate hailo; cd /home/cym/Work/tests/HAILO'
alias s.8='conda activate yolov8; cd /home/cym/Work/tests/yolov8_test'
alias c.n='cat ./.NOTE'
alias v.n='vi ./.NOTE'
alias cdt='cd ~/Work/TensorRT-7.2.1.6/'
#alias cds='cd /home/cym/Work/spacenorm_tensorrt'
#alias cds='cd /home/cym/Work/spacenorm_yolov7/scripts/cctv'
alias cds='cd /home/cym/Work/spacenorm_yolov7/docker_swarm'
#alias cdsy='cd /home/cym/Work/spacenorm_yolov7'
alias cdf='cd ~/Work/spacenorm_yolov7/scripts/cctv/make_FP_training_set'
alias cdm='cd ~/Work/mAP'
alias cdms='cd ~/Work/mAP/scripts/extra'
#alias cdd='cd ~/Work/darknet'
alias cdd='cd /home/cym/Work/spacenorm_yolov7/data/cctv_train_data'
alias cddc='cd ~/Work/darknet/data/cctv'
alias cdds='cd ~/Work/darknet/scripts/cctv'
alias cdmt='cd ~/Work/darknet/scripts/cctv/mturk_utils/boto3_test'
alias cdq='cd ~/Work/amazon_mturk/create_qual_test'
#alias cdw='cd ~/Work/WSGI/test'
alias cdy='cd ~/Work/spacenorm_yolov7'
alias cda='cd ~/Work/spacenorm_anomaly_detect'
alias cdw='cd ~/Work/GTEC-related/webcrawl'
#alias cdi='cd /home/cym/Work/spacenorm_yolov7/data/cctv_train_data/inspection'
alias cdi='cd /home/cym/Work/yolov11_training_aws/infra'
alias cdv='cd /home/cym/Work/tests/vibration_autoencoder_test'
alias cdl='cd /home/cym/Work/spacenorm_yolov7/docker_shared/logs'
alias cdla='cd /home/cym/Work/spacenorm_anomaly_detect/docker_shared/logs'
alias cdb='cd /home/cym/Work/spacenorm_yolov7/docker_build'


#alias v.t='vi ./trt_yolo_spacenorm.py'
alias v.s='vi ./spacenorm_yolov7.py'
#alias t.y='tail -f /home/cym/Work/spacenorm_yolov7/docker_shared/logs/cym.log'
#alias dlog='docker logs -f --tail 100 yolov7_cym'
alias t.a='tail -f /home/cym/Work/spacenorm_anomaly_detect/docker_shared/logs/cym.log'
alias docker_rm='docker rm `docker ps -a -q`'
#alias docker_run='docker run --gpus all --name spacenorm_yolov7 --rm -p 8081:8081 -v ~/Work/spacenorm_yolov7:/workspace/spacenorm_yolov7 -it ahnjw72/spacenorm:tensorrt22.07-py3_spacenorm_yolov7 bash'

#alias docker_run='docker run --gpus all --name spacenorm_yolov7 -e TZ=Asia/Seoul --rm -p 8081:8081 -it ahnjw72/spacenorm:beta1.0 bash'
docker_image="ahnjw72/spacenorm:20230923"
alias docker_run="docker run --gpus all --name spacenorm -e TZ=Asia/Seoul --rm -p 8081:8081 -it ${docker_image} bash"
#alias docker_exec='docker exec -it spacenorm_yolov7 bash'
#alias docker_commit='docker commit spacenorm_yolov7 ahnjw72/spacenorm:tensorrt22.07-py3_spacenorm_yolov7'
alias dey='docker exec -it -e LC_ALL=C.UTF-8 yolov7_cym bash'
alias dea='docker exec -it -e LC_ALL=C.UTF-8 anomaly_cym bash'
alias drmy='docker rm -f yolov7_cym'
alias drmy_car='docker stop yolov7_cym_car; docker rm yolov7_cym_car'
alias drma='docker stop anomaly_cym; docker rm anomaly_cym'
alias dps='docker ps -a'
alias di='docker images'
alias dls='docker service ls'

#alias dpush='docker push ahnjw72/spacenorm:latest'
#alias dpush='docker push ahnjw72/spacenorm_without_trt:latest'
#alias dpush='docker push ahnjw72/spacenorm_compatibility_test:latest'
alias dpush='docker push ahnjw72/spacenorm_yolov7:latest'
#alias dpusha='docker push ahnjw72/spacenorm_anomaly_detect:latest'
alias dpusha='docker push 159552820182.dkr.ecr.ap-northeast-2.amazonaws.com/spacenorm_anomaly_detect:latest'

alias diprune='docker system prune -a -f'
#alias dbuild='cd /home/cym/Work/spacenorm_yolov7/docker_build; ./docker_build.sh ahnjw72/spacenorm:latest'
#alias dbuild='cd /home/cym/Work/spacenorm_yolov7/docker_build; ./docker_build_without_TRT.sh'
alias dbuild='cd /home/cym/Work/spacenorm_yolov7/docker_build; ./docker_build_robust_rtsp.sh'

alias dlabel="docker node ls -q | xargs -n1 docker node inspect --format '{{ .Description.Hostname }}  {{ .Spec.Labels }}'"

alias jpt='jupyter notebook --ip=0.0.0.0 --port=8083 --allow-root'
#alias jpt='jupyter lab --ip=0.0.0.0 --port=8083 --allow-root'
#alias gpush='git push --force --set-upstream origin master'
#alias gpush='git push --force https://ahnjw72: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@github.com/ahnjw72/spacenorm_yolov7.git --all'
#alias gpush='git push --force git@github.com:ahnjw72/spacenorm_yolov7.git --all'
alias gpush='git push origin master'
#alias gpusha='git push --force https://ahnjw72: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@github.com/ahnjw72/spacenorm_anomaly_detect.git --all'
alias gpusha='git push --force git@github.com:ahnjw72/spacenorm_anomaly_detect.git --all'
#alias gclone='git clone https://github.com/ahnjw72/spacenorm_yolov7.git'
alias gclone='git clone git@github.com:ahnjw72/spacenorm_yolov7.git'

alias gits='git status'
alias gminor='git commit -m "minor modification"'
gitrmall() {
  # 1. Stage all deletions (git rm)
  git ls-files --deleted -z | xargs -0 git rm

  # 2. Commit the changes
  # Uses your message if provided, otherwise defaults to "minor modification"
  git commit -m "${1:-remove unused files}"
}

alias dus='du -h --max-depth=1 | sort -hr'


#alias restart_spacenorm='sudo systemctl restart spacenorm_cctv_detection.service'
#alias start_spacenorm='sudo systemctl start spacenorm_cctv_detection.service'
#alias stop_spacenorm='sudo systemctl stop spacenorm_cctv_detection.service'
#alias restart_spacenorm='sudo systemctl restart spacenorm_cctv_yolov7_detection.service'
#alias start_spacenorm='sudo systemctl start spacenorm_cctv_yolov7_detection.service'
#alias stop_spacenorm='sudo systemctl stop spacenorm_cctv_yolov7_detection.service'
#alias restart_anomaly_detect='sudo systemctl restart spacenorm_anomaly_detect.service'
#alias start_anomaly_detect='sudo systemctl start spacenorm_anomaly_detect.service'
#alias stop_anomaly_detect='sudo systemctl stop spacenorm_anomaly_detect.service'
alias sty='echo cym08552; cd /home/cym/Work/spacenorm_yolov7; sudo systemctl start spacenorm_cctv_yolov7_detection.service'
alias sta='echo cym08552; cd /home/cym/Work/spacenorm_anomaly_detect; sudo systemctl start spacenorm_anomaly_detect.service'

alias tfy='tail -f /home/cym/Work/spacenorm_yolov7/docker_shared/cym.log'
alias tfa='tail -f /home/cym/Work/spacenorm_anomaly_detect/docker_shared/cym.log'

alias tempa='python -u mqtt_spacenorm_anomaly_detect.py --cfg company_cfg/cym.json --log test.log'

#alias nonzero='find ./ -type f -size +0 -print0 | tr -d -c '\0' | wc -c'
alias nonzero='find ./ -type f -size +0 | wc -l'
#alias gpuname='lspci | grep -i VGA'
alias gpuname='nvidia-smi --query-gpu=name --format=csv,noheader'
alias cpuname='cat /proc/cpuinfo | grep name'

# prometheus reload
alias preload='curl -X POST http://127.0.0.1:9090/-/reload'

# aws related
#alias ecr_login='aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 159552820182.dkr.ecr.ap-northeast-2.amazonaws.com'
#alias dpush_ecr='docker push 159552820182.dkr.ecr.ap-northeast-2.amazonaws.com/spacenorm_without_trt:latest'
#dpush_ecr() {
    #aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 159552820182.dkr.ecr.ap-northeast-2.amazonaws.com
    #docker push $1 
#}
dpush_ecr() {
    local FULL_IMAGE=$1

    # 1. Extract metadata from the image string
    # String format: 123456789012.dkr.ecr.region.amazonaws.com/repo_name:tag
    local REGISTRY=$(echo $FULL_IMAGE | cut -d'/' -f1)
    local REPO_WITH_TAG=$(echo $FULL_IMAGE | cut -d'/' -f2)
    local REPO_NAME=$(echo $REPO_WITH_TAG | cut -d':' -f1)
    local REGION=$(echo $REGISTRY | cut -d'.' -f4)

    echo "Targeting Region: $REGION"
    echo "Targeting Repo: $REPO_NAME"

    # 2. Login to ECR
    aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "$REGISTRY"

    # 3. Create the repository if it doesn't exist
    # --status-code 200/400 check is handled by '|| true' to keep it simple
    echo "Ensuring repository '$REPO_NAME' exists..."
    aws ecr create-repository --repository-name "$REPO_NAME" --region "$REGION" > /dev/null 2>&1 || \
    echo "Repository already exists or skipped creation."

    # 4. Push the image
    docker push "$FULL_IMAGE"
}

nsml_address="223.130.132.43"
gpunsml_address="175.45.205.187"
#alias nsml='ssh nsml@175.45.202.84 -p 15001 -o StrictHostKeyChecking=no'
alias nsml='ssh nsml@${nsml_address} -p 15001 -o StrictHostKeyChecking=no'
alias gnsml='ssh nsml@${gpunsml_address} -p 15001 -o StrictHostKeyChecking=no'

# be sure to set this port and address as correct ones (its lifetime is very short) 
KT_port="10980"
KT_address="proxy1.aitrain.ktcloud.com"
tokt() {
    scp -i ~/Work/id_container -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P ${KT_port} -rp $1 work@${KT_address}:$2
}
fromkt() {
    scp -i ~/Work/id_container -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P ${KT_port} -rp work@${KT_address}:$1 $2
}

yujin_port="22"
yujin_address="10.241.147.61"
toyujin() {
    scp -i ~/Work/id_container -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P ${yujin_port} -rp $1 spacenorm@${yujin_address}:$2
}
fromyujin() {
    scp -i ~/Work/id_container -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P ${yujin_port} -rp spacenorm@${yujin_address}:$1 $2
}

tonsml() {
    scp -P 15001 $1 nsml@${nsml_address}:$2
}

tonsmlr() {
    scp -P 15001 -r $1 nsml@${nsml_address}:$2
}

fromnsml() {
    scp -P 15001 nsml@${nsml_address}:$1 $2
}

fromnsmlr() {
    scp -P 15001 -r nsml@${nsml_address}:$1 $2
}

#alias docker_run_anomaly_detect='docker run --gpus all -d -e TZ=Asia/Seoul -it --name spacenorm_anomaly_detect ${docker_image} /workspace/spacenorm_anomaly_detect/spacenorm_anomaly_detect_in_docker.sh'
# Ex)) $ docker_run_anomaly_detect <COMPANY_NAME> 
docker_run_anomaly_detect() {
    docker run --gpus all -d -e TZ=Asia/Seoul -it --name anomaly_$1 ${docker_image} /workspace/spacenorm_anomaly_detect/spacenorm_anomaly_detect_in_docker.sh company_cfg/$1.json $1.log
}

#alias docker_run_yolov7='docker run --gpus all -d -e TZ=Asia/Seoul -p 8081:8081 -it --name spacenorm_yolov7 ${docker_image} /workspace/spacenorm_yolov7/spacenorm_yolov7_in_docker.sh'
#docker_run_yolov7() {
    #docker run --gpus all -d -e TZ=Asia/Seoul -p 8081:8081 -it --name yolov7_$1 ${docker_image} /workspace/spacenorm_yolov7/spacenorm_yolov7_in_docker.sh $1 
#}

# $1 : xx.service
install_service() {
    sudo cp $1 /etc/systemd/system/$1
    sudo systemctl daemon-reload
    sudo systemctl enable $1 
}

yolov7_detect() {
    python detect.py --weights spacenorm_yolov7.pt --iou-thres 0.65 --conf-thres 0.452 --source $1 --no-trace --save-txt --nosave --img-size 1280
}

# $1 : 10.241.87.117 (cym) or 10.241.147.61 (yujin) 
# $2 : spacenorm/efa1d3bdf973/CD
mqtt_test() {
    echo "mosquitto_sub -h $1 -p 1883 -t $2 -d"
    mosquitto_sub -h $1 -p 1883 -t $2 -d
    #mosquitto_sub -h 10.241.147.61 -p 1883 -t $1 -d
}

grab_image() {
    #ffmpeg -rtsp_transport tcp -i "rtsp://space:spacenorm12%21%40%23@192.168.7.211:554/0/profile2/media.smp" -frames:v 1 -q:v 2 frame.jpg
    ffmpeg -rtsp_transport tcp -i $1 -frames:v 1 -q:v 2 $2 
}

rc() {
    echo ""
    echo "==============================================================="
    echo "/home/cym/Work/spacenorm_yolov7/spacenorm_cfg --> 금호정공"
    rsync -avz --delete /home/cym/Work/spacenorm_yolov7/spacenorm_cfg spacenorm@10.241.24.21:/home/spacenorm/Work/spacenorm_yolov7
    rsync -avz --include="spacenorm_yolov7_*.sh" --exclude="*" ./ spacenorm@10.241.24.21:/home/spacenorm/Work/spacenorm_yolov7

    echo "==============================================================="
    echo "/home/cym/Work/spacenorm_yolov7/spacenorm_cfg --> 유진판지"
    rsync -avz --delete /home/cym/Work/spacenorm_yolov7/spacenorm_cfg spacenorm@10.241.147.61:/home/spacenorm/Work/spacenorm_yolov7
    rsync -avz --include="spacenorm_yolov7_*.sh" --exclude="*" ./ spacenorm@10.241.147.61:/home/spacenorm/Work/spacenorm_yolov7

    echo "==============================================================="
    echo "/home/cym/Work/spacenorm_yolov7/spacenorm_cfg --> 아세아_안성"
    rsync -avz --delete /home/cym/Work/spacenorm_yolov7/spacenorm_cfg spacenorm@10.241.34.204:/home/spacenorm/Work/spacenorm_yolov7
    rsync -avz --include="spacenorm_yolov7_*.sh" --exclude="*" ./ spacenorm@10.241.34.204:/home/spacenorm/Work/spacenorm_yolov7

    echo "==============================================================="
    echo "/home/cym/Work/spacenorm_yolov7/spacenorm_cfg --> 제일산업_로봇"
    rsync -avz --delete /home/cym/Work/spacenorm_yolov7/spacenorm_cfg spacenorm@10.241.232.189:/home/spacenorm/backup/Work/spacenorm_yolov7
    rsync -avz --include="spacenorm_yolov7_*.sh" --exclude="*" ./ spacenorm@10.241.232.189:/home/spacenorm/backup/Work/spacenorm_yolov7
    echo "==============================================================="
    echo ""
}

show_labels() {
    docker node ls -q | xargs -n1 docker node inspect \
  --format '{{ .Description.Hostname }}  {{ .Spec.Labels }}'
}

dlog() {
    docker service logs -f --tail 100 spacenorm_yolov7_$1 2>&1
}
num_grab() {
    docker service logs -f --tail 100 spacenorm_yolov7_$1 2>&1 | grep "grab_img threads running"
}

dlogout() {
    docker service logs spacenorm_yolov7_$1
}

dloga() {
    docker service logs -f --tail 100 spacenorm_anomaly_detect_$1 2>&1
}

dsrm() {
    docker service rm spacenorm_yolov7_$1
}

dsrma() {
    docker service rm spacenorm_anomaly_detect_$1
}

ecr_login() {
    aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 159552820182.dkr.ecr.ap-northeast-2.amazonaws.com
}

dtail() {
    LOGPATH=$(docker inspect --format='{{.LogPath}}' $1)
    echo $LOGPATH
    sudo tail -f $LOGPATH
}

dy7() {
    cd /home/cym/Work/spacenorm_yolov7/docker_swarm/
    /home/cym/Work/spacenorm_yolov7/docker_swarm/deploy_yolov7.sh $1
}

dexec() {
    docker exec -it $1 bash
}

ddigest() {
    docker inspect $1 --format='{{index .Image}}'
}
alias spot_status='aws s3 cp s3://spacenorm-cctv-ml-data/yolo_training_aws/logs/training.log - | grep -E "SPOT_TERMINATION|Uploaded last.pt|Restored checkpoint|Training command"'
alias spot_ssh='ssh -i ~/.ssh/yolov11-training.pem ubuntu@43.203.168.46'
