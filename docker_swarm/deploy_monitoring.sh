#!/usr/bin/env bash
# Deploy/redeploy the spacenorm_monitoring stack (Prometheus + Grafana +
# cAdvisor + node-exporter + the nginx gateway).
#
# Unlike deploy_obj_detection.sh, there is no per-site image to resolve from
# ECR -- every image here is a fixed public upstream version pinned directly
# in stack_monitoring.yml. What this script automates instead is the config
# hash/rebuild dance: Swarm config objects are immutable once created, so
# every config file below needs a fresh content-hash-named object whenever
# its content changes, with stack_monitoring.yml's `name:` field updated to
# match. This is the same procedure documented by hand in that file's own
# header comment -- this script just does it for all six configs at once.
#
# One-time setup, NOT handled by this script (run once per manager node):
#   1. Create the Grafana admin password secret:
#        printf '%s' '<a real password>' | docker secret create grafana_admin_password -
#   2. Ensure the bind-mount source directories exist on the manager node:
#        sudo mkdir -p /var/lib/spacenorm_monitoring/prometheus /var/lib/spacenorm_monitoring/grafana
#        sudo chown -R 65534:65534 /var/lib/spacenorm_monitoring/prometheus
#        sudo chown -R 472:472 /var/lib/spacenorm_monitoring/grafana
#
# After running this script, verify deployment with:
#  $ docker service ls | grep spacenorm_monitoring
#  $ docker service logs -f spacenorm_monitoring_<service>
#
# Remove the whole stack:
#  $ docker stack rm spacenorm_monitoring
#
# Restart one service without a config change:
#  $ docker service update --force spacenorm_monitoring_<service>

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

STACK_NAME="spacenorm_monitoring"
STACK_FILE="stack_monitoring.yml"

# config_key (matches the `configs:` section in stack_monitoring.yml) -> source file
declare -A CONFIG_FILES=(
  [prometheus_cfg]="monitoring/prometheus.yml"
  [grafana_datasource_cfg]="monitoring/grafana/provisioning/datasources/prometheus.yml"
  [grafana_dashboard_provider_cfg]="monitoring/grafana/provisioning/dashboards/dashboards.yml"
  [grafana_dashboard_fleet]="monitoring/grafana/provisioning/dashboards/json/fleet_overview.json"
  [grafana_dashboard_per_camera]="monitoring/grafana/provisioning/dashboards/json/per_camera.json"
  [grafana_dashboard_cctv_config]="monitoring/grafana/provisioning/dashboards/json/cctv_configuration.json"
  [gateway_nginx_cfg]="monitoring/nginx/nginx.conf"
)

for CONFIG_KEY in "${!CONFIG_FILES[@]}"; do
  CONFIG_FILE="${CONFIG_FILES[$CONFIG_KEY]}"
  CONFIG_HASH="$(sha256sum "${CONFIG_FILE}" | awk '{print substr($1,1,12)}')"
  CONFIG_NAME="${CONFIG_KEY}_${CONFIG_HASH}"

  # Create config objects only if missing -- Swarm config objects are
  # immutable, so a content-hash name that already exists means this exact
  # file content was already registered; nothing to do.
  if ! docker config inspect "${CONFIG_NAME}" >/dev/null 2>&1; then
    docker config create "${CONFIG_NAME}" "${CONFIG_FILE}" >/dev/null
    echo "[OK] Created config: ${CONFIG_NAME}"
  else
    echo "[OK] Config exists: ${CONFIG_NAME}"
  fi

  # Update stack_monitoring.yml's `name:` for this config key in place.
  # Matches the stable `<key>_<12-hex-hash>` pattern, not the previous
  # literal value, so this is safe even if the file was hand-edited since
  # the last run.
  sed -i -E "s/${CONFIG_KEY}_[0-9a-f]{12}/${CONFIG_NAME}/" "${STACK_FILE}"
done

echo "[INFO] Ensure /var/lib/spacenorm_monitoring/{prometheus,grafana} exist on the manager node."
echo "       If not, see the one-time setup steps in this script's header comment."

docker stack deploy --with-registry-auth -c "${STACK_FILE}" "${STACK_NAME}"

echo "[OK] Deployed stack '${STACK_NAME}'"
