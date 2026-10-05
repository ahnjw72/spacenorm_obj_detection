# Sets END_DATE to today (e.g., 2026-01-03)
END_DATE=$(date +%Y-%m-%d)

# Sets START_DATE to exactly one month ago (e.g., 2025-12-03)
START_DATE=$(date -d "1 month ago" +%Y-%m-%d)

aws ce get-cost-and-usage \
    --time-period Start=$START_DATE,End=$END_DATE \
    --granularity DAILY \
    --metrics "UnblendedCost" "UsageQuantity" \
    --filter '{ "Dimensions": { "Key": "INSTANCE_TYPE", "Values": ["g5.xlarge", "g5.2xlarge", "g5.4xlarge", "g5.8xlarge", "g5.16xlarge", "g5.12xlarge"] }}' \
    --group-by Type=DIMENSION,Key=INSTANCE_TYPE
