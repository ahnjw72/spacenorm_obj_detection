#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# ECR → Swarm "Track tag, pin digest" updater
#
# Behavior:
#  - If service image is REGISTRY/repo:tag
#       → resolve tag to digest in ECR
#       → update service to REGISTRY/repo@sha256:<digest> (pin it)
#  - If service image is REGISTRY/repo@sha256:<digest>
#       → determine track tag (label ecr.track-tag or DEFAULT_TRACK_TAG)
#       → resolve tag to digest in ECR
#       → if differs, update service to new digest (re-pin)
#
# Notes:
#  - Uses docker service update --with-registry-auth so workers need no AWS creds
#  - Requires AWS CLI creds on the manager (IAM role or env-based credentials)
###############################################################################

# ---- Configuration (edit as needed) -----------------------------------------
REGION="${REGION:-ap-northeast-2}"
REGISTRY="${REGISTRY:-159552820182.dkr.ecr.ap-northeast-2.amazonaws.com}"

# Default tag to track if service has no label override
DEFAULT_TRACK_TAG="${DEFAULT_TRACK_TAG:-latest}"

# Optional: only act on services that have this label key (set empty to disable)
# Example: REQUIRE_LABEL_KEY="ecr.auto-update"
REQUIRE_LABEL_KEY="${REQUIRE_LABEL_KEY:-}"

# Per-service tag override label:
#   docker service update --label-add ecr.track-tag=stable <svc>
TRACK_TAG_LABEL_KEY="${TRACK_TAG_LABEL_KEY:-ecr.track-tag}"

HEALTH_FILE="${HEALTH_FILE:-/var/log/ecr-refresh-status.json}"
LOCK_FILE="${LOCK_FILE:-/var/lock/ecr-swarm-digest-refresh.lock}"

# If true: when ECR lookup fails for a service, count as failed and continue
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
# ----------------------------------------------------------------------------

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

update_health() {
  local status="$1"
  local message="$2"
  local checked="${3:-0}"
  local updated="${4:-0}"
  local pinned="${5:-0}"
  local skipped="${6:-0}"
  local failed="${7:-0}"
  local duration_s="${8:-0}"

  mkdir -p "$(dirname "$HEALTH_FILE")" 2>/dev/null || true
  printf '{"last_run":"%s","status":"%s","message":"%s","checked":%s,"updated":%s,"pinned":%s,"skipped":%s,"failed":%s,"duration_s":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$status" \
    "$(echo "$message" | sed 's/"/\\"/g')" \
    "$checked" "$updated" "$pinned" "$skipped" "$failed" "$duration_s" \
    > "$HEALTH_FILE"
}

# ---- AWS ECR login ----------------------------------------------------------
ecr_login() {
  local token
  if ! token="$(aws ecr get-login-password --region "$REGION" 2>/dev/null)"; then
    return 1
  fi
  echo "$token" | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null 2>&1
}

# ---- Docker helpers ---------------------------------------------------------
get_service_image_spec() {
  local svc="$1"
  docker service inspect "$svc" --format '{{.Spec.TaskTemplate.ContainerSpec.Image}}' 2>/dev/null || true
}

get_service_label_value() {
  local svc="$1"
  local key="$2"
  docker service inspect "$svc" --format "{{ index .Spec.Labels \"$key\" }}" 2>/dev/null || true
}

service_has_required_label() {
  local svc="$1"
  local key="$2"
  [[ -z "$key" ]] && return 0
  local v
  v="$(get_service_label_value "$svc" "$key")"
  [[ -n "$v" && "$v" != "<no value>" && "$v" != "null" ]]
}

# Parse repo from:
#   REGISTRY/repo:tag
#   REGISTRY/repo@sha256:...
parse_repo_from_image() {
  local image="$1"
  local without_registry="${image#${REGISTRY}/}"     # repo:tag OR repo@sha256:...
  # strip tag or digest suffix
  without_registry="${without_registry%@sha256:*}"   # remove @sha256:* if present
  without_registry="${without_registry%:*}"          # remove :tag if present
  echo "$without_registry"
}

# Parse tag from REGISTRY/repo:tag (empty if digest form)
parse_tag_from_image() {
  local image="$1"
  if [[ "$image" == *@sha256:* ]]; then
    echo ""
    return 0
  fi
  local without_registry="${image#${REGISTRY}/}"
  if [[ "$without_registry" != *:* ]]; then
    echo ""
    return 0
  fi
  echo "${without_registry##*:}"
}

# Parse digest from:
#  - REGISTRY/repo@sha256:...
#  - optionally from running tasks (best-effort fallback)
parse_digest_from_spec() {
  local image="$1"
  if [[ "$image" == *@sha256:* ]]; then
    echo "${image##*@}"   # sha256:...
  else
    echo ""
  fi
}

get_current_digest_from_tasks() {
  local svc="$1"
  local img
  img="$(docker service ps "$svc" --no-trunc --format '{{.Image}}' 2>/dev/null | head -n1 || true)"
  if [[ "$img" == *@sha256:* ]]; then
    echo "${img##*@}"
  else
    echo ""
  fi
}

# ---- ECR helper -------------------------------------------------------------
get_remote_digest_for_tag() {
  local repo="$1"
  local tag="$2"
  aws ecr describe-images \
    --region "$REGION" \
    --repository-name "$repo" \
    --image-ids "imageTag=$tag" \
    --query 'imageDetails[0].imageDigest' \
    --output text 2>/dev/null || true
}

# ---- Main -------------------------------------------------------------------
main() {
  local start_ts end_ts duration_s
  start_ts="$(date +%s)"

  log "RUNNING" "ECR refresh in progress..."

  # single-run lock
  mkdir -p "$(dirname "$LOCK_FILE")" 2>/dev/null || true
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    update_health "SKIPPED" "Another run is in progress (lock held)." 0 0 0 0 0 0
    exit 0
  fi

  if ! ecr_login; then
    update_health "ERROR" "ECR login failed (check IAM/network/region/registry)." 0 0 0 0 1 0
    exit 1
  fi

  local SERVICES_CHECKED=0
  local SERVICES_UPDATED=0   # digest changed -> repin update
  local SERVICES_PINNED=0    # tag form -> pinned for first time
  local SERVICES_SKIPPED=0
  local SERVICES_FAILED=0

  while IFS= read -r svc; do
    log "Processing service: $svc"

    [[ -z "$svc" ]] && continue

    # Optional gating label
    if ! service_has_required_label "$svc" "$REQUIRE_LABEL_KEY"; then
      log "SKIP: $svc: missing required label $REQUIRE_LABEL_KEY"
      ((++SERVICES_SKIPPED))
      continue
    fi

    local image repo spec_tag track_tag remote_digest current_digest desired_image
    image="$(get_service_image_spec "$svc")"
    log "DEBUG: $svc: image spec -> ${image:-<none>}"
    [[ -z "$image" ]] && { ((++SERVICES_FAILED)); continue; }

    log "DEBUG: $svc: checking if image starts with target registry $REGISTRY/"
    # only manage images in target registry
    if [[ "$image" != "$REGISTRY/"* ]]; then
      log "SKIP: $svc: image $image does not start with target registry $REGISTRY/"
      ((++SERVICES_SKIPPED))
      continue
    fi

    ((++SERVICES_CHECKED))

    repo="$(parse_repo_from_image "$image")"
    log "DEBUG: $svc: parsed repo -> ${repo:-<none>}"
    if [[ -z "$repo" ]]; then
      log "WARN: $svc: cannot parse repo from image spec: $image"
      ((++SERVICES_FAILED))
      continue
    fi

    # Determine which tag to track
    track_tag="$(get_service_label_value "$svc" "$TRACK_TAG_LABEL_KEY")"
    if [[ -z "$track_tag" || "$track_tag" == "<no value>" || "$track_tag" == "null" ]]; then
      track_tag="$DEFAULT_TRACK_TAG"
      log "INFO: $svc: using default track tag '$track_tag'"
    fi

    # If service is tag-based, respect that tag as the track tag (unless label overrides)
    spec_tag="$(parse_tag_from_image "$image")"
    if [[ -n "$spec_tag" && ( "$(get_service_label_value "$svc" "$TRACK_TAG_LABEL_KEY")" == "<no value>" || -z "$(get_service_label_value "$svc" "$TRACK_TAG_LABEL_KEY")" || "$(get_service_label_value "$svc" "$TRACK_TAG_LABEL_KEY")" == "null" ) ]]; then
      track_tag="$spec_tag"
      log "INFO: $svc: using track tag '$track_tag' from image spec"
    fi

    remote_digest="$(get_remote_digest_for_tag "$repo" "$track_tag")"
    log "DEBUG: $svc: resolved remote digest for ${repo}:${track_tag} -> ${remote_digest}"
    if [[ -z "$remote_digest" || "$remote_digest" == "None" || "$remote_digest" == "null" ]]; then
      log "WARN: $svc: cannot resolve ECR digest for $repo:$track_tag"
      ((++SERVICES_FAILED))
      [[ "$CONTINUE_ON_ERROR" == "true" ]] && continue || break
    fi

    # Determine current digest
    current_digest="$(parse_digest_from_spec "$image")"
    log "DEBUG: $svc: current digest from spec -> ${current_digest:-<none>}"
    if [[ -z "$current_digest" ]]; then
      # fallback: tasks might show repo:tag@sha256:...
      current_digest="$(get_current_digest_from_tasks "$svc")"
    fi

    desired_image="${REGISTRY}/${repo}@${remote_digest}"
    log "DEBUG: $svc: desired image with digest -> ${desired_image}"

    # If not pinned yet (tag-based spec), pin it now
    if [[ "$image" != *@sha256:* ]]; then
      log "PIN: $svc: ${REGISTRY}/${repo}:${track_tag} -> ${desired_image}"
      if docker service update --with-registry-auth --image "$desired_image" "$svc" >/dev/null 2>&1; then
        ((++SERVICES_PINNED))
      else
        log "ERROR: $svc: docker service update failed (pin)"
        ((++SERVICES_FAILED))
      fi
      continue
    fi

    # Already pinned; update only if digest differs
    log "DEBUG: $svc: comparing current digest ${current_digest:-<none>} with remote digest ${remote_digest}"
    if [[ -z "$current_digest" ]]; then
      # rare: spec digest missing? repin anyway
      log "UPDATE: $svc: current digest unknown; repin to ${desired_image}"
      if docker service update --with-registry-auth --image "$desired_image" "$svc" >/dev/null 2>&1; then
        ((++SERVICES_UPDATED))
      else
        log "ERROR: $svc: docker service update failed (repin)"
        ((++SERVICES_FAILED))
      fi
      continue
    fi

    if [[ "$current_digest" != "$remote_digest" ]]; then
      log "UPDATE: $svc: ${current_digest} -> ${remote_digest} (track ${repo}:${track_tag})"
      if docker service update --with-registry-auth --image "$desired_image" "$svc" >/dev/null 2>&1; then
        ((++SERVICES_UPDATED))
      else
        log "ERROR: $svc: docker service update failed (digest change)"
        ((++SERVICES_FAILED))
      fi
    else
      log "OK: $svc: current digest matches remote digest; no update needed"
      ((++SERVICES_SKIPPED))
    fi
    
  done < <(docker service ls --format '{{.Name}}')

  end_ts="$(date +%s)"
  duration_s="$((end_ts - start_ts))"

  update_health "SUCCESS" \
    "Checked=$SERVICES_CHECKED, Pinned=$SERVICES_PINNED, Updated=$SERVICES_UPDATED, Skipped=$SERVICES_SKIPPED, Failed=$SERVICES_FAILED" \
    "$SERVICES_CHECKED" "$SERVICES_UPDATED" "$SERVICES_PINNED" "$SERVICES_SKIPPED" "$SERVICES_FAILED" "$duration_s"

  log "DONE: Checked=$SERVICES_CHECKED Pinned=$SERVICES_PINNED Updated=$SERVICES_UPDATED Skipped=$SERVICES_SKIPPED Failed=$SERVICES_FAILED duration_s=$duration_s"
}

main "$@"
