#!/usr/bin/env bash

aws configure set aws_access_key_id AKIASKJQW2PLNPCHNUHG \
&& aws configure set aws_secret_access_key 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 \
&& aws configure set region ap-northeast-2 \
&& aws configure set output json \
&& aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 159552820182.dkr.ecr.ap-northeast-2.amazonaws.com \
&& aws s3 sync s3://spacenorm-cctv-ml-data/spacenorm_yolov7/others/ ~/scripts_in_EC2_instance/ \
&& cd ~/scripts_in_EC2_instance \
&& chmod +x ./*.sh \
&& source ./.bash_aliases