# some more ls aliases
alias ll='ls -alFh'
alias la='ls -A'
alias l='ls -CF'

alias v.c='vi ~/.bashrc'
alias s.c='source ~/.bashrc'
alias v.a='vi ~/scripts_in_EC2_instance/.bash_aliases'
alias s.a='source ~/scripts_in_EC2_instance/.bash_aliases'
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
alias cds='cd /home/cym/Work/spacenorm_yolov7/scripts/cctv'
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
alias cdy='cd /mnt/spacenorm_data/spacenorm_yolov7'
alias cda='cd ~/Work/spacenorm_anomaly_detect'
alias cdw='cd ~/Work/GTEC-related/webcrawl'
alias cdi='cd /home/cym/Work/spacenorm_yolov7/data/cctv_train_data/inspection'
alias cdv='cd /home/cym/Work/tests/vibration_autoencoder_test'
alias cdl='cd /mnt/spacenorm_data/spacenorm_yolov7/docker_shared'


#alias v.t='vi ./trt_yolo_spacenorm.py'
alias v.s='vi ./spacenorm_yolov7.py'
alias t.s='tail -f spacenorm_yolov7.log'
alias docker_rm='docker rm `docker ps -a -q`'
#alias docker_run='docker run --gpus all --name spacenorm_yolov7 --rm -p 8081:8081 -v ~/Work/spacenorm_yolov7:/workspace/spacenorm_yolov7 -it ahnjw72/spacenorm:tensorrt22.07-py3_spacenorm_yolov7 bash'

#alias docker_run='docker run --gpus all --name spacenorm_yolov7 -e TZ=Asia/Seoul --rm -p 8081:8081 -it ahnjw72/spacenorm:beta1.0 bash'
docker_image="ahnjw72/spacenorm:20230923"
alias docker_run="docker run --gpus all --name spacenorm -e TZ=Asia/Seoul --rm -p 8081:8081 -it ${docker_image} bash"
#alias docker_exec='docker exec -it spacenorm_yolov7 bash'
#alias docker_commit='docker commit spacenorm_yolov7 ahnjw72/spacenorm:tensorrt22.07-py3_spacenorm_yolov7'
alias dey='docker exec -it -e LC_ALL=C.UTF-8 yolov7_training bash'
alias dea='docker exec -it -e LC_ALL=C.UTF-8 anomaly_cym bash'
alias drmy='docker stop yolov7_training; docker rm yolov7_training'
alias drmy_car='docker stop yolov7_cym_car; docker rm yolov7_cym_car'
alias drma='docker stop anomaly_cym; docker rm anomaly_cym'
alias dps='docker ps -a'
#alias dpush='docker push ahnjw72/spacenorm:latest'
alias dpush='docker push ahnjw72/spacenorm_without_trt:latest'
alias dpull='docker pull 159552820182.dkr.ecr.ap-northeast-2.amazonaws.com/spacenorm_without_trt:latest'
alias diprune='docker image prune'
#alias dbuild='cd /home/cym/Work/spacenorm_yolov7/docker_build; ./docker_build.sh ahnjw72/spacenorm:latest'
alias dbuild='cd /home/cym/Work/spacenorm_yolov7/docker_build; ./docker_build_without_TRT.sh'

alias jpt='jupyter notebook --ip=0.0.0.0 --port=8083 --allow-root'
#alias jpt='jupyter lab --ip=0.0.0.0 --port=8083 --allow-root'
#alias gpush='git push --force --set-upstream origin master'
#alias gpush='git push --force https://ahnjw72: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@github.com/ahnjw72/spacenorm_yolov7.git --all'
alias gpush='git push --force git@github.com:ahnjw72/spacenorm_yolov7.git --all'
#alias gpusha='git push --force https://ahnjw72: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@github.com/ahnjw72/spacenorm_anomaly_detect.git --all'
alias gpusha='git push --force git@github.com:ahnjw72/spacenorm_anomaly_detect.git --all'
#alias gclone='git clone https://github.com/ahnjw72/spacenorm_yolov7.git'
alias gclone='git clone git@github.com:ahnjw72/spacenorm_yolov7.git'

alias gits='git status'

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
alias gpuname='lspci | grep -i VGA'
alias cpuname='cat /proc/cpuinfo | grep name'

# prometheus reload
alias preload='curl -X POST http://127.0.0.1:9090/-/reload'


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
    sudo cp $1 /lib/systemd/system/$1
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
