# Prometheus + Grafana Fleet Monitoring — Plan

## Context

`spacenorm_obj_detection` runs as one Docker Swarm service per client site
(`spacenorm_obj_detection_cym`, `_jaeil_cr`, `_jaeil_io`, `_kumho`,
`_kumho_car`), each scheduled onto a different physical gateway node. There
is currently no unified way to see, in one place:

- Whether each site's service is actually healthy (up, restarting, crash
  history, resource pressure).
- Whether a given camera's RTSP connection is up, or stuck reconnecting.
- Whether the dynamic-config method (`feature/dynamic-cctv-config`) is
  actually re-fetching from the vision_nodes API on schedule
  (`config_renew_period`).
- What a camera is currently seeing, with its ROI and detection boxes drawn.

Today, checking any of this means SSH-less CLI spelunking through
`docker service ps` / `docker service logs` per site (what this session has
been doing by hand for `jaeil_cr`) — there is no dashboard.

This plan was prompted directly by investigating a real incident: jaeil_cr's
production service crashed three times (SIGSEGV / GLib abort, "Too many open
files") over several hours. The root cause — a file-descriptor leak in
`RTSP_Camera.release()` triggered by chronically-reconnecting cameras — was
only found by manually reading crash logs after the fact. A
`process_open_fds` trend line and a per-camera reconnect-attempt counter
would have surfaced this hours before the crash. That incident is the
concrete justification for several of the metrics below, not a hypothetical.

## Current state (audited, not assumed)

- Each service's Flask app (`spacenorm_obj_detection.py`) exposes
  `/video_feed` on `:8081`, but only for **one** camera at a time — whichever
  `selected_key` was last chosen via `?cctv_key=` on `/`. No grid view, no
  per-camera direct link.
- A `:9000` port is published in every site's stack file for Prometheus, but
  **nothing listens on it** — `prometheus_client.start_http_server()` is
  never called anywhere in the codebase. The one `Gauge('thread_count', ...)`
  object that exists is updated in memory but was never actually exposed to
  a scraper. This is a dead/half-built feature, not a working one.
- The ROI-snapshot writer added for dynamic-mode config (requirement 7 of
  `vision_node_requirements.md`) writes one annotated `.jpg` per camera, but
  only on `config_renew_period` (minutes), and only for dynamic-mode sites.
  It is not continuous and does not cover static-mode sites.
- All 5 `spacenorm_obj_detection_*` services already share one Swarm overlay
  network (`spacenorm_obj_detection_default`), because they're deployed
  under the same stack name. **A new service attached to that network can
  reach every site's container port by Swarm-internal DNS**
  (`spacenorm_obj_detection_jaeil_cr:8081`, etc.) regardless of which
  physical node each one lands on. This is what makes a central dashboard
  possible without SSH/node access to every site.
- No monitoring stack (Portainer/Prometheus/Grafana) is deployed anywhere in
  this swarm today — confirmed via `docker stack ls`.

## Decision

- **Status/health monitoring**: Prometheus + Grafana (not Portainer,
  not Uptime Kuma) — chosen for richer historical dashboards and alerting,
  and because the image-viewing requirement needs a dashboard surface
  regardless.
- **Important correction baked into this plan**: Prometheus is a numeric
  time-series store; it cannot hold or serve image bytes, and Grafana cannot
  pull a live JPEG "from Prometheus." Grafana's role is to be the single
  viewing surface — it embeds an image URL via its built-in HTML/Text panel
  (no plugin required) — but the actual bytes must be served by something
  else. That "something else" is a new, deliberately minimal component
  (§4), not a revival of the old Flask app.
- The existing Flask web-streaming app and the dead `Gauge`-only metric are
  removed, not kept alongside the new stack.

## 1. Remove

From `spacenorm_obj_detection.py`:

- The Flask `app` object, `index()`, `video_feed()`, `generate()` routes, and
  the `index.html` template. (Port `:8081` itself is **kept** — see §4, which
  repurposes it for one new, much smaller route. Only this specific
  implementation — templating, MJPEG streaming, single-selected-camera
  state — goes away.)
- `outputFrame`, `lock`, `selected_key`, `cctv_space_name` globals and all
  code that writes to them inside `detector_per_cam()`.
- The `web_streaming_port` branch in `main()` (the `app.run(...)` call and
  its `get_default_host_ip()` helper) — replaced by §4's route registration.
- The existing `Gauge('thread_count', ...)` in `detect_multithreaded()`
  (superseded by the richer metric set in §2).
- The `web_streaming_port` config key's old meaning (gate for the MJPEG
  streaming UI) from `default.json`'s `miscellaneous` section. `PORT1`/
  port-8081 wiring in `deploy_obj_detection.sh` / `stack.yml.template` stays
  — same port, new purpose.

## 2. Add — Prometheus instrumentation

One `start_http_server(METRICS_PORT)` call at startup (fixing the currently
dead `:9000`), plus:

| Metric | Type | Labels | Rationale |
|---|---|---|---|
| `spacenorm_detector_threads_active` | Gauge | — | Live detector thread count |
| `spacenorm_detector_threads_expected` | Gauge | — | Sensor count from config; gap vs. active = stuck/failed cameras |
| `spacenorm_camera_connected` | Gauge (0/1) | `camera` | Per-camera RTSP connection state |
| `spacenorm_camera_last_frame_timestamp` | Gauge | `camera` | Epoch of last successfully grabbed frame — staleness alerting, stronger signal than "thread alive" |
| `spacenorm_camera_reconnect_attempts_total` | Counter | `camera` | The signal that was missing before the fd-leak crash — a fast-climbing counter on one camera is the early warning |
| `spacenorm_detections_total` | Counter | `camera` | Cumulative occupancy-report events |
| `spacenorm_mqtt_connected` | Gauge (0/1) | `camera` | Mirrors existing per-camera MQTT client state |
| `spacenorm_dynamic_config_last_fetch_timestamp` | Gauge | — (dynamic method only) | Direct, dashboard-visible answer to "is `config_renew_period` actually firing" |
| `spacenorm_dynamic_config_sensors_active` | Gauge | — (dynamic method only) | Sensor count from the last successful API fetch; tracks add/remove events over time |
| `process_open_fds` | Gauge | — | Free via `prometheus_client`'s default `ProcessCollector`, enabled automatically once `start_http_server()` runs. This is the metric that would have shown the leak climbing toward the crash. |
| `process_resident_memory_bytes` | Gauge | — | Same free collector |
| `spacenorm_camera_info` | Gauge, always `1` | `camera`, `device_id`, `company_name`, `config_method`, `monitor_id`, `mqtt_broker_addr`, `mqtt_broker_port`, `min_obj_size_ratio`, `has_roi` | The standard Prometheus "info metric" pattern (same idea as `kube_pod_info`): configuration as labels on a constant-value gauge, queryable as a table. Backs the configuration table in §6. Sourced from the same `cctv_ID` dict both static and dynamic methods already populate at runtime, so one metric covers both with no special-casing. |

**Deliberately excluded from `spacenorm_camera_info`**: `access_token` /
`refresh_token` (bearer credentials — never belong in a label, which is
plaintext in both the scrape endpoint and Prometheus's own storage), and the
raw `uri` / `private_url` (the dynamic-mode RTSP URI embeds the hardcoded
`space:spacenorm12!@#` credentials — exposing it as a label would leak them
through `:9000` and any Grafana table built on this metric). If the URI's
host/path (no userinfo) turns out to be useful in the table later, it gets
added back with credentials stripped, not as the raw string.

Infra-level, zero custom code, one per node:

- **cAdvisor** — per-container CPU/memory/restart-count.
- **node-exporter** — host-level disk/network/fd-limit pressure.

## 3. Add — Prometheus + Grafana deployment

- Prometheus deployed as a Swarm service, scrape config targets each site's
  `spacenorm_obj_detection_<site>:<METRICS_PORT>` by Swarm DNS (no new
  networking — reuses `spacenorm_obj_detection_default`).
- Grafana deployed as a Swarm service, Prometheus as its datasource.
- **Fleet dashboard**: one row per site — up/down, active vs. expected
  threads, open-fd trend, restart count, dynamic-config freshness.
- **Per-camera dashboard**: reconnect-attempt rate, last-frame age,
  detection history, templated by a `camera` dashboard variable.
- **Live imagery row**: Grafana Text panel (HTML mode), camera-selector
  template variable, embedding
  `<img src="http://spacenorm_obj_detection_<site>:8081/snapshot/<camera>.jpg?t=$__now">`.
  Dashboard auto-refresh is left **off** — bytes are only (re-)fetched on
  initial dashboard load and when the user clicks Grafana's own refresh
  button, both of which change `$__now` and so bust the browser's image
  cache. No custom "refresh button" code on either side; this is Grafana's
  native refresh behavior, just with auto-refresh disabled so nothing
  polls in between. Bytes come from §4, never from Prometheus.

## 4. Add — on-demand snapshot endpoint (reuses `:8081`)

**Capture happens on request, not continuously** — no background writer
thread, no periodic disk writes. `:8081` (the old Flask video-streaming
port) is repurposed for one lightweight route, `GET /snapshot/<camera>`,
added back into the *same* per-site process (not a new service, not a
separate container):

1. `cam.read()` on the live `RTSP_Camera` for that key — the freshest raw
   frame, already held in-process by the running detector thread.
2. Draw the ROI overlay (`draw_roi_overlay()`, already exists).
3. Draw the most recently computed detection boxes from a small per-camera
   cache (`boxes`/`confs`/`clss`/`types`/`motionesses`), updated once at the
   end of each `detector_per_cam()` loop iteration — negligible cost, it's
   just stashing values already computed for the existing report/log path,
   not an extra inference run. The overlay can be up to one `report_period`
   (2–3s) stale relative to the brand-new frame it's drawn on; acceptable
   for a human glancing at a dashboard, not used for anything time-critical.
4. Encode as JPEG and return it directly as the HTTP response
   (`Content-Type: image/jpeg`) — no disk write, no stored file.

This replaces the old Flask app's role (image serving) with something much
smaller: one route, no templating, no per-client stream state, no
selected-camera global. Request frequency is bounded by how often someone
actually has the dashboard open and refreshing it — not continuous, which
was the explicit requirement this design is built around.

## 5. Persistence and auth (resolved)

Prometheus and Grafana hold two different kinds of state with two different
costs when lost. Prometheus's TSDB (`/prometheus`) is passively-collected
history — losing it means losing the trend line that would have shown the
jaeil_cr fd-leak building over hours, but the dashboard still works
going forward. Grafana's own database (`/var/lib/grafana`: dashboards, users,
datasource configs, alert rules) is authored configuration — losing it means
rebuilding every dashboard by hand, a strictly worse loss.

**Node placement.** Neither a bind mount nor a named volume with Docker's
default `local` driver follows a service across nodes if Swarm reschedules
it — both are tied to whichever single node they're created on, and this
project already has first-hand evidence of what happens when that's not
accounted for (`deploy_obj_detection.sh`'s note that `/var/lib/spacenorm_obj_detection`,
the TRT engine cache, had to be manually `mkdir -p`'d per node because Swarm
doesn't reliably auto-create a missing bind source). **Decision: pin
Prometheus and Grafana to the manager node (`ahnjw-GTEC-Ubuntu`) via a
placement constraint**, the same pattern already used for every per-site
service (`node.labels.server == <site>`) — chosen over any client site's
on-site gateway node because it's the one node not subject to a client
site's own power/network interruptions. Once pinned to one node, bind mount
vs. named volume is a style choice; **decision: bind mount**, for
consistency with every other persistence decision already made in this
codebase, not because it's technically superior here.

**Dashboards as code, not as database rows.** Grafana supports loading
dashboards and datasources from version-controlled JSON/YAML at container
startup (`provisioning/dashboards/`, `provisioning/datasources/`) instead of
depending on its SQLite database alone. **Decision: dashboard definitions are
checked into this repo and provisioned this way**, so a full Grafana
container/volume wipe loses zero dashboard configuration — it's
re-created from git on every start, the same way `spacenorm_cfg` already
works for behavior config. Only Prometheus's collected metrics and Grafana's
user/session database still depend on the bind mount actually persisting.

**Retention.** Prometheus's default `--storage.tsdb.retention.time` is 15
days. **Decision: set this explicitly to 60 days** — cheap at this project's
metric cardinality (5 sites, a handful of cameras each), and long enough to
diagnose a slow-building issue like the fd leak without having already
aged out of the window.

**Auth.** This dashboard shows live camera imagery of people on active
factory floors — a materially different risk than a toy metrics board left
on default credentials. Decisions:

- Grafana's default `admin`/`admin` is never left in place — the admin
  password is set via `GF_SECURITY_ADMIN_PASSWORD` from a Swarm secret at
  deploy time, not hardcoded in the stack file.
- Viewer and Admin are separate accounts. Anyone who only needs to glance at
  fleet health gets a Viewer account, not the account that can edit
  dashboards, add datasources, or run raw Prometheus queries (itself a
  secondary privacy surface, since camera-presence-correlated metrics are
  queryable even without the image panels).
- **No published host port.** Grafana does not need to be reachable outside
  the swarm to do its job, given the existing `spacenorm_obj_detection_default`
  overlay network already reaches every site. Access is via SSH tunnel /
  VPN into the manager node, not a `mode: host` port publish. If routine
  multi-person access later makes a published port worth it, a reverse
  proxy (Traefik/nginx) terminating TLS in front of it becomes mandatory at
  that point, not optional — Grafana serves plain HTTP by default, and
  publishing it directly would put admin credentials and camera frames in
  cleartext on whatever network reaches that port.
- OAuth/SSO (Grafana supports Google/GitHub/LDAP out of the box) is not
  adopted for the initial rollout — local accounts plus the two points above
  are proportionate for a small, solo-operated fleet. Revisit if more than
  one or two people end up needing routine access.

## 6. CCTV configuration table and per-camera drill-down

New requirement: every site's dashboard shows a structured table of its
CCTV configuration — covering **both** static and dynamic config method with
no special-casing, since both already populate the same `cctv_ID` dict at
runtime (this is exactly why `spacenorm_camera_info`, §2, is modeled as one
metric regardless of which method sourced a given entry).

- **Table panel**, querying `spacenorm_camera_info`, one row per camera:
  `camera`, `device_id`, `company_name`, `config_method`, `monitor_id`,
  `mqtt_broker_addr`/`port`, `min_obj_size_ratio`, `has_roi`. Credentials and
  raw URIs are never in this table — see the exclusion list in §2.
- **Click-through**: the `camera` column is configured as a Grafana **data
  link** (built into the Table panel, no plugin) that navigates to the
  per-camera dashboard already specified in §3, passing the clicked value
  through as that dashboard's `camera` template variable
  (`/d/<per-camera-dashboard-uid>?var-camera=${__data.fields.camera}`).
  No new dashboard is built for this — it reuses §3's existing per-camera
  dashboard (reconnect-attempt rate, last-frame age, detection history, plus
  the on-demand snapshot image from §4) as the click target.

## 7. Rollout (decided: staged)

jaeil_cr first — it's already the dynamic-method pilot site from
`feature/dynamic-cctv-config`, so it's the site this session has the deepest
live operational visibility into if something goes wrong. Each of the other
4 sites (cym, jaeil_io, kumho, kumho_car) follows individually once jaeil_cr
has run clean for a reasonable observation period, rather than cutting all 5
over on one image rebuild. This plan does not require a config flag to gate
the rollout per site — `spacenorm_obj_detection.py` is shared code, but
since the old Flask/metrics paths are being removed outright (§1) rather
than kept behind a toggle, "staged" here means staging the **image
rebuild + redeploy** per site (same mechanism already used for the
fd-leak fix rollout to jaeil_cr earlier in this session), not maintaining
two code paths simultaneously.

## Deferred (explicitly, not blocking initial rollout)

- **Alerting scope**: this plan specifies the metrics and dashboards; actual
  Grafana alert rules (e.g., `process_open_fds` approaching ulimit,
  `spacenorm_camera_reconnect_attempts_total` rate threshold) are a
  follow-up decision made after the dashboards exist and there's a baseline
  to alert against.

## Explicitly out of scope for this plan

- Migrating detection itself to a third-party NVR/detection stack (e.g.
  Frigate) — this plan only adds observability around the existing
  YOLO/RTSP pipeline, it does not replace it.
- MQTT-based dashboards (Node-RED / Home Assistant) for occupancy counts —
  a separate, smaller idea surfaced during investigation, not part of this
  plan.
