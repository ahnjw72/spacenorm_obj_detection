#!/usr/bin/env bash
# Fetch `docker service logs` for a spacenorm_obj_detection service within a local-time
# window, and filter down to one camera's reporting-related lines.
#
# Keeps lines mentioning "report" or "AR coment with files" (yes, "coment" is a typo
# in the source logging code — matched verbatim on purpose), then drops the noisier
# "inference", "report_period", and "__report():" lines.
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") -s SERVICE -c SITE -f FROM -t TO [-z TZ_OFFSET] [-o OUTFILE]

  -s SERVICE     Docker service name (e.g. spacenorm_obj_detection_jaeil_cr)
  -c SITE        Camera/site name as it appears in log lines
                 (e.g. "Jaeil_Robot_EVOL_08x21#1_R1")
  -f FROM        Start of window, local time: "YYYY-MM-DD HH:MM:SS"
  -t TO          End of window, local time: "YYYY-MM-DD HH:MM:SS"
  -z TZ_OFFSET   Offset from UTC for FROM/TO (default: +09:00)
  -o OUTFILE     Write result to this file instead of stdout

Example:
  $(basename "$0") -s spacenorm_obj_detection_jaeil_cr \\
    -c "Jaeil_Robot_EVOL_08x21#1_R1" \\
    -f "2026-07-16 10:00:00" -t "2026-07-16 10:10:00" \\
    -o /tmp/filtered.log
EOF
  exit 1
}

SERVICE=""
SITE=""
FROM=""
TO=""
TZ_OFFSET="+09:00"
OUTFILE=""

while getopts "s:c:f:t:z:o:h" opt; do
  case "$opt" in
    s) SERVICE="$OPTARG" ;;
    c) SITE="$OPTARG" ;;
    f) FROM="$OPTARG" ;;
    t) TO="$OPTARG" ;;
    z) TZ_OFFSET="$OPTARG" ;;
    o) OUTFILE="$OPTARG" ;;
    h|*) usage ;;
  esac
done

[[ -z "$SERVICE" || -z "$SITE" || -z "$FROM" || -z "$TO" ]] && usage

SINCE_UTC=$(date -u -d "${FROM} ${TZ_OFFSET}" +"%Y-%m-%dT%H:%M:%SZ")
UNTIL_UTC=$(date -u -d "${TO} ${TZ_OFFSET}" +"%Y-%m-%dT%H:%M:%SZ")
# Append max fractional seconds so the boundary second is included in full
# (docker's --since/--until aren't both supported on `service logs`, so the
# upper bound is enforced client-side against the --timestamps column).
UNTIL_BOUND="${UNTIL_UTC%Z}.999999999"

fetch_and_bound() {
  # docker service logs has no --until; stop reading as soon as a line's
  # timestamp passes the bound (safe because docker stamps lines in read order).
  # awk's early exit sends docker SIGPIPE, so isolate the pipeline's exit
  # status to awk's alone -- otherwise `pipefail` would treat that as failure.
  local awk_status
  set +o pipefail
  docker service logs --timestamps --since "$SINCE_UTC" "$SERVICE" 2>&1 \
    | awk -v until="$UNTIL_BOUND" '{ ts=$1; gsub(/Z$/,"",ts); if (ts <= until) print; else exit }'
  awk_status=${PIPESTATUS[1]}
  set -o pipefail
  return "$awk_status"
}

filter_pipeline() {
  grep -F "$SITE" \
    | grep -iE "report|AR coment with files|AR comment with files" \
    | grep -v "inference" \
    | grep -v "report_period" \
    | grep -v -- "__report():"
}

if [[ -n "$OUTFILE" ]]; then
  fetch_and_bound | filter_pipeline > "$OUTFILE"
  echo "Wrote $(wc -l < "$OUTFILE") lines to $OUTFILE" >&2
else
  fetch_and_bound | filter_pipeline
fi
