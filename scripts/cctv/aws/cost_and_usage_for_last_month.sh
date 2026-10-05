#!/bin/sh
# AWS CLI for checking billing status for a last month

START_DATE=$(date -d "last month" +%Y-%m-01)

# 현재 날짜 기준으로 해당 월의 1일 날짜(YYYY-MM-01 형식)
#END_DATE=$(date +%Y-%m-01)

# 그냥 오늘 날짜
END_DATE=$(date +%Y-%m-%d)

echo $START_DATE "~" $END_DATE

aws ce get-cost-and-usage \
    --time-period Start=$START_DATE,End=$END_DATE \
    --granularity MONTHLY \
    --metrics "UnblendedCost" "UsageQuantity" \
    --filter '{ "Dimensions": { "Key": "INSTANCE_TYPE", "Values": ["g5.xlarge", "g5.2xlarge", "g5.4xlarge", "g5.8xlarge", "g5.16xlarge", "g5.12xlarge"] }}' \
    --group-by Type=DIMENSION,Key=INSTANCE_TYPE
