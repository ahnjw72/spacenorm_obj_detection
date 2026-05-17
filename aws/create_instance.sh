#! /bin/bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: <instance_type> (ex: {t3.small|t3.xlarge|g5.xlarge|g5.12xlarge}"
  exit 1
fi

INSTANCE_TYPE="$1"
#if [ "${INSTANCE_TYPE}" != "t3.small" ] && [ "${INSTANCE_TYPE}" != "g5.xlarge" ] && [ "${INSTANCE_TYPE}" != "g5.12xlarge" ] && [ "${INSTANCE_TYPE}" != "t3.xlarge" ]; then
  #echo "Error: Invalid instance type. Please choose either 't3.small', 'g5.xlarge' or 'g5.12xlarge'."
  #exit 1
#fi

IMG_ID=ami-08381cc5143a321b7
KEY_NAME=ML_spot_instance
SECURITY_GROUP_ID=sg-090b8ab1ad20e88bc
SUBNET_ID=subnet-000d47e7bdaeb91d6
AVAILABILITY_ZONE=ap-northeast-2a
OWNER_TAG=ahnjaewoo
NAME_TAG=yolo_training_spot

AWS_COMMAND="aws ec2 run-instances \
  --image-id ${IMG_ID} \
  --instance-type ${INSTANCE_TYPE} \
  --key-name ${KEY_NAME} \
  --security-group-ids ${SECURITY_GROUP_ID} \
  --subnet-id ${SUBNET_ID} \
  --instance-market-options '{\"MarketType\":\"spot\"}' \
  --placement AvailabilityZone=${AVAILABILITY_ZONE} \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Owner,Value=${OWNER_TAG}},{Key=Name,Value=${NAME_TAG}}]' \
  --associate-public-ip-address \
  --count 1"

echo "Running AWS command: ${AWS_COMMAND}"

eval ${AWS_COMMAND}
