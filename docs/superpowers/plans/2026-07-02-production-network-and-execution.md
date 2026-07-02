# Production Network And Execution Plan

Date: 2026-07-02
Status: Accepted
Scope: How RAG Agent Backend is exposed, operated, and kept recoverable in production.

This plan records the decided production architecture and the work required to
reach it. It follows the constraints in `skills.md` (Hardware And Hosting
Context, Architecture Rules) and does not change any application architecture:
API and worker stay separate processes, Redis Streams stays the only queue,
retrieval stays on `rag1.HeaderInContentTrial`.

## 1. Goals And Constraints

- The chat frontend is a static browser app on cPanel (`chat.rag-agent.example`).
  Its users' browsers need an HTTPS API origin that reaches the backend.
- cPanel remains authoritative for `ziyutec.com` DNS and continues hosting the
  static website.
- More than 6 people must be able to use the app, from devices where we cannot
  require a VPN client.
- The backend host is a private Windows PC (RTX 5090, WSL2). It accepts no
  inbound connections. All backend services stay bound to loopback or the
  internal Docker network.
- No new paid services. Anything introduced must be free and removable.
- Tailscale remains the private access layer for admin and CI, and Tailscale
  Funnel provides the public HTTPS URL for the API.
- Introduce the minimum number of new moving parts.

## 2. Decision Record: Public API URL

The browser-to-API path is handled by Tailscale Funnel on the production host.
The compose stack does not run a public tunnel or reverse proxy container.

Options considered:

| Option | Free | Host stays private | Works for users without installs | Verdict |
| --- | --- | --- | --- | --- |
| Tailscale on every user device | No | Yes | No | Rejected: per-user tailnet membership and installs |
| VPS reverse proxy over the tailnet | No | Yes | Yes | Rejected: recurring cost, extra host to patch |
| Router port-forward plus reverse proxy on the host | Yes | No | Yes | Rejected: violates the no-inbound-exposure rule; may be impossible under CGNAT |
| cPanel as API reverse proxy | Yes | n/a | n/a | Rejected: cPanel cannot reach the private host; streaming, uploads, and auth headers are unproven there |
| Temporary tunnel tools | Yes | Yes | No | Rejected: random URLs, interstitials, session limits; not production |
| Tailscale Funnel | Yes | Yes | Yes | Accepted |

The integration is deliberately minimal and reversible:

- Docker publishes the API only on `127.0.0.1:9229`.
- Tailscale Funnel runs on the production host and forwards public HTTPS
  traffic to that local port.
- The frontend uses the exact Funnel URL printed by `tailscale funnel status`.
- `PRODUCTION_PUBLIC_HEALTH_URL` in GitHub is the exact Funnel URL plus
  `/health`.
- cPanel DNS stays unchanged. A pure Funnel setup uses Tailscale's `*.ts.net`
  hostname rather than a `ziyutec.com` API subdomain.
- Removal path: turn off Funnel on the host and clear the GitHub public health
  URL. No application data or Docker service depends on the public URL.

## 3. Target Network Architecture

Five trust zones:

| Zone | Contents | Reachable from |
| --- | --- | --- |
| Public internet | User browsers, cPanel frontend, Tailscale Funnel URL | Everyone |
| Tailnet | Admin devices, ephemeral CI node (`tag:ci`), production host | Tailnet members only |
| WSL2 Docker network | api, ingestion-worker, Redis, Postgres, Milvus (etcd, MinIO), observability stack | Containers and host loopback only |
| Windows host | Ollama on the RTX 5090 (`:11434`), Tailscale daemon | WSL2/containers via `host.docker.internal`; tailnet for admin/CI |
| cPanel | Static frontend and DNS for `ziyutec.com` | Everyone for the website; DNS remains managed there |

Traffic paths:

- **Users**: browser loads the static site from cPanel and calls the Funnel API
  base URL. Tailscale terminates public HTTPS for the `*.ts.net` hostname and
  forwards to `http://127.0.0.1:9229` on the production host.
- **Admin**: Tailscale SSH to the production host, then the port-forward recipe
  in `skills.md` for Grafana (3001), Prometheus (9090), Milvus (19530), Redis
  (6380), Postgres (5433), collector health (13133).
- **CI**: GitHub Actions joins the tailnet as an ephemeral node, SSHes in,
  rsyncs the source, and runs `infra/deploy.sh`.
- **Inference**: containers reach Ollama at `host.docker.internal:11434`.
  Ollama is never routed through Funnel or bound to a public interface.

Network invariants enforced in review:

- Every `ports:` entry in `infra/docker-compose.yaml` keeps its `127.0.0.1:`
  prefix. `RAG_AGENT_HTTP_BIND` stays `127.0.0.1` in production.
- Compose has no public tunnel service and no public reverse proxy profile.
- Grafana, Attu, Prometheus, MinIO, Milvus, Redis, and Postgres remain
  loopback/tailnet-only.
- The Windows firewall rule for Ollama (11434) is scoped to the WSL2 virtual
  subnet, never "any". Verify the effective `OLLAMA_HOST` rather than changing
  code (`skills.md`).
- The production env file lives outside `PRODUCTION_DEPLOY_PATH` so rsync
  cannot delete it.

## 4. Public URL Implementation

The application repository only owns the local service:

- `infra/docker-compose.yaml` publishes `api` as
  `${RAG_AGENT_HTTP_BIND:-127.0.0.1}:${RAG_AGENT_HTTP_PORT:-9229}:9229`.
- `infra/deploy.sh` reconciles the application, infrastructure, tools, and
  observability profiles. It does not start the public URL.
- `infra/env.production.example` contains application and data-store settings
  only. There is no public tunnel token.

The production host owns Funnel:

```bash
tailscale funnel --bg --yes 9229
tailscale funnel status
```

Run those commands on the machine where `127.0.0.1:9229` reaches the API. The
status output prints the public HTTPS URL. It will look like
`https://<node>.<tailnet>.ts.net`, where `<node>` is the Tailscale machine name
and `<tailnet>` is the tailnet DNS name. Do not type the angle brackets; use the
real URL from the command output.

`infra/deploy.sh` also runs these Funnel checks during deployment:

- verifies Tailscale is running and the host is logged in,
- verifies local API health at `http://127.0.0.1:9229/health`,
- ensures Funnel is enabled for `9229`,
- prints `tailscale funnel status`,
- curls `PUBLIC_HEALTH_URL` when CI passes it to the deploy.

The `--bg` flag makes Funnel persistent across Tailscale restarts and host
reboots. Do not add a Docker service for it unless the production host is
deliberately redesigned around that service boundary.

## 5. Application Hardening

- **CORS**: `CORS_ORIGINS` is comma-separated. Production should include only
  the static frontend origins that must call the API.
- **Secrets**: production env file gets a long random `JWT_SECRET_KEY` and
  non-default Postgres and MinIO passwords, per
  `infra/env.production.example`.
- **Upload limit**: the application enforces 52 428 800 bytes through
  `INGESTION_MAX_UPLOAD_BYTES`. The public URL must be tested with a realistic
  multipart upload before launch.

## 6. Boot Persistence

A consumer Windows PC reboots after power loss or Windows Update. The stack must
recover with no human present. Required chain:

1. **Windows**: Ollama installed to start at boot (service or startup task, not
   tied to interactive logon). Windows Update active hours set; reboots are
   accepted, not fought.
2. **WSL2 boot**: `/etc/wsl.conf` enables systemd (`[boot] systemd=true`). A
   Task Scheduler job runs at system startup ("run whether user is logged on or
   not") and executes a `wsl.exe` keep-alive so the distro boots headless.
3. **Resources**: `.wslconfig` pins a fixed memory budget for WSL2 so Milvus,
   Postgres, Redis, and the observability stack cannot starve Windows or the GPU
   workload.
4. **Docker**: `docker.service` enabled in the distro. Every compose service
   carries `restart: unless-stopped`, so the stack self-assembles in dependency
   order.
5. **Funnel**: Tailscale is logged in on the production host and Funnel is
   configured for local port `9229`.
6. **Acceptance drill**: pull the plug once, on purpose. The system passes when
   `PRODUCTION_PUBLIC_HEALTH_URL` returns 200 within a defined window (target:
   10 minutes) with zero manual steps.

## 7. Deploy And Rollback

The existing pipeline (`.github/workflows/deploy.yml` -> tailnet -> rsync ->
`infra/deploy.sh` with lock, Ollama gate, `--wait`) stays.

- **Image pinning**: the deploy step sets `RAG_AGENT_IMAGE_TAG` to the git SHA.
  Rollback becomes "re-run the deploy workflow from the last good commit".
- **Public smoke test**: the final workflow step curls
  `PRODUCTION_PUBLIC_HEALTH_URL`, so a deploy that breaks the API or the public
  Funnel route fails in CI instead of in front of users.

## 8. Backups And Disaster Recovery

All durable state lives in Docker volumes on one consumer machine. Backup
targets, in priority order:

- **Postgres** (auth, conversations, checkpoints): nightly `pg_dump` via a
  systemd timer in WSL2.
- **Milvus** (838k+ migrated chunks): scheduled backup with the `milvus-backup`
  tool, or cold copies of the `milvus_data`, `etcd_data`, and `minio_data`
  volumes. The preserved legacy Docker volumes plus the checkpoint at
  `.deploy/legacy-milvus-migration.json` remain the rebuild-of-last-resort via
  `tools/migrate_legacy_milvus.py`.
- **Redis**: AOF is already enabled; snapshot `redis_data` with the same timer.
  Losing it costs in-flight jobs only; documents re-ingest.
- **Off-host copy**: encrypted archives pushed to storage we already pay for.
  The cPanel account over SFTP is acceptable zero-cost storage for encrypted
  backup archives. A backup on the same SSD as the database is not a backup.
- **Retention**: observability data (Loki, Tempo, Prometheus) is disposable;
  configure 14-30 day retention instead of backing it up.
- **Acceptance**: one restore rehearsed end-to-end into a scratch environment.
  An untested backup does not count.

## 9. Monitoring And Alerting

Observability (OTel -> Tempo/Loki/Prometheus/Grafana) exists; alerting does not.
Add:

- **External uptime probe** (free tier of UptimeRobot or healthchecks.io)
  against `PRODUCTION_PUBLIC_HEALTH_URL`, the only check that sees the public
  URL the way users do.
- **Prometheus alert rules** on the canonical signals from `skills.md`:
  dead-letter stream growth, `XPENDING` age on `rag-agent-ingestion-workers`,
  API 5xx rate, collector health (`:13133`), and disk usage on the Milvus,
  MinIO, and Loki volumes.
- **Delivery**: Grafana alerting to company email (cPanel SMTP, already paid
  for).
- Debugging stays as documented in `skills.md`: Loki -> TraceID -> Tempo,
  `rag_agent_ingestion_*` metrics.

## 10. Rollout Phases

Phases 1-3 gate production launch. Phases 4-6 follow within days and do not
block it.

1. **Public URL.** Enable Tailscale Funnel for port `9229`, set the frontend API
   base URL to the Funnel URL, and set `PRODUCTION_PUBLIC_HEALTH_URL` in GitHub.
   *Accept when*: the cPanel app logs in, streams a chat response, and uploads a
   40 MB document through the Funnel URL.
2. **App hardening.** Env-driven CORS and strong secrets verified in the server
   env file.
   *Accept when*: production `/health` works and a localhost-origin CORS
   preflight is rejected.
3. **Boot persistence.** wsl.conf, Task Scheduler job, `.wslconfig`, Ollama at
   boot, and Funnel restored after reboot.
   *Accept when*: the pull-the-plug drill passes hands-off.
4. **Backups.** Timers running; one restore rehearsed.
5. **Alerting.** External probe plus the Prometheus rules; a deliberately
   stopped worker confirmed to page.
6. **Deploy polish.** SHA-pinned image tags and the public URL smoke step in CI.

## 11. Standing Rules

- Users never join the tailnet; the public API path is Funnel only.
- cPanel keeps the website and DNS.
- Any future need for a custom API hostname, edge rate limiting, additional
  public hostnames, or a host-level reverse proxy is a change to this plan and
  gets reviewed first.
- Any replacement for Funnel must still satisfy: free, no inbound exposure, no
  per-user installs, and streaming, 50 MB multipart uploads, and
  `Authorization` headers proven.
