INSTANCE_IDS=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=yolo_training_spot" \
           "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' \
  --output text)


echo "aws ec2 terminate-instances --instance-ids $INSTANCE_IDS"
aws ec2 terminate-instances --instance-ids $INSTANCE_IDS
