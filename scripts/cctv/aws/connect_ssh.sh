IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=yolo_training_spot" \
  --query 'Reservations[].Instances[].PublicIpAddress' \
  --output text)

echo "ssh -i ML_spot_instance.pem ubuntu@$IP"

ssh -i ML_spot_instance.pem ubuntu@$IP