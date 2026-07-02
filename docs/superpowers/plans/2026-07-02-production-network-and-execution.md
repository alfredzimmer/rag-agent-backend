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

- The chat frontend is a static browser app on cPanel (`chat.rag-agent.example`). Its
  users' browsers need an HTTPS API origin that reaches the backend.
- More than 6 people must be able to use the app, from devices where we cannot
  require a VPN client.
- The backend host is a private Windows PC (RTX 5090, WSL2). It accepts no
  inbound connections. All backend services stay bound to loopback or the
  internal Docker network.
- No new paid services. Anything introduced must be free and removable.
- Tailscale remains the private access layer for admin and CI only. Its free
  plan (6 users, unlimited devices as of April 2026) covers those identities;
  end users are never added to the tailnet.
- Introduce the minimum number of new components.

## 2. Decision Record: Public API Edge

The only architectural gap is the browser-to-API path. Options considered:

| Option | Free | Host stays private | Works for >6 users without installs | Verdict |
| --- | --- | --- | --- | --- |
| Tailscale on every user device | No (free tier is 6 users) | Yes | No | Rejected: per-seat cost, per-device install |
| VPS reverse proxy over the tailnet | No | Yes | Yes | Rejected: recurring cost, extra box to patch |
| Router port-forward + reverse proxy on the host | Yes | No | Yes | Rejected: violates the no-inbound-exposure rule; may be impossible under CGNAT |
| cPanel as API reverse proxy | Yes | n/a | n/a | Rejected: cPanel cannot reach the private host; streaming/uploads/auth headers unproven there |
| Free tunnel tools (ngrok free, quick tunnels, localtunnel) | Yes | Yes | No | Rejected: random URLs, interstitials, session limits; not production |
| Cloudflare Tunnel | Yes | Yes (outbound-only) | Yes | **Accepted** |

Cloudflare Tunnel is the unique option that is free, keeps the host
un-exposed, needs no client software, and is production-grade. The 50 MB
upload cap (`INGESTION_MAX_UPLOAD_BYTES=52428800`) fits under the free plan's
100 MB request body limit.

The integration is deliberately minimal and reversible:

- One `cloudflared` container, one named tunnel, one ingress rule
  (`api.ziyutec.com -> http://api:9229`), one proxied DNS record.
- `rag-agent.example` nameservers move to Cloudflare (free). cPanel keeps hosting the
  site; existing DNS records are recreated unchanged.
- Explicitly out of scope for launch: WAF ruleset tuning, edge rate limiting,
  Cloudflare Access. JWT auth and the application upload cap remain the
  enforcement points. Edge rules can be added later without architecture
  changes.
- Removal path: delete the container and the env token, move DNS back. No
  data or code depends on Cloudflare.

## 3. Target Network Architecture

Five trust zones:

| Zone | Contents | Reachable from |
| --- | --- | --- |
| Public internet | User browsers, cPanel frontend | Everyone |
| Cloudflare edge | `api.ziyutec.com`, TLS termination | Everyone, HTTPS only |
| Tailnet | Admin devices, ephemeral CI node (`tag:ci`) | Tailnet members only |
| WSL2 Docker network | api, ingestion-worker, cloudflared, Redis, Postgres, Milvus (etcd, MinIO), observability stack | Containers and host loopback only |
| Windows host | Ollama on the RTX 5090 (`:11434`) | WSL2/containers via `host.docker.internal` |

Traffic paths:

- **Users**: browser loads `chat.rag-agent.example` from cPanel, calls
  `https://api.ziyutec.com`. Cloudflare terminates TLS and hands the request to
  the tunnel that `cloudflared` opened outbound from the Docker network.
  `cloudflared` forwards to `http://api:9229` over the compose network.
- **Admin**: Tailscale SSH to `ai-server`, then the port-forward recipe in
  `skills.md` for Grafana (3001), Prometheus (9090), Milvus (19530), Redis
  (6380), Postgres (5433), collector health (13133).
- **CI**: unchanged. GitHub Actions joins the tailnet as an ephemeral node,
  SSHes in, rsyncs the source, runs `infra/deploy.sh`.
- **Inference**: containers reach Ollama at `host.docker.internal:11434`.
  Ollama is never routed through the tunnel or bound to a public interface.

Network invariants (enforce in every review):

- Every `ports:` entry in `infra/docker-compose.yaml` keeps its `127.0.0.1:`
  prefix. `RAG_AGENT_HTTP_BIND` stays `127.0.0.1` in production.
- The tunnel routes exactly one hostname: `api.ziyutec.com`. Grafana, Attu,
  Prometheus, MinIO, Milvus, and Redis never receive an ingress rule.
  Grafana runs with anonymous admin and is only safe because it is
  loopback/tailnet-only.
- The Windows firewall rule for Ollama (11434) is scoped to the WSL2 virtual
  subnet, never "any". Verify the effective `OLLAMA_HOST` rather than
  changing code (`skills.md`).
- The production env file lives outside `PRODUCTION_DEPLOY_PATH` (readme
  requirement) so rsync cannot delete it.

## 4. Edge Implementation

Add a `cloudflared` service to `infra/docker-compose.yaml` behind a new
`edge` profile:

```yaml
cloudflared:
  image: cloudflare/cloudflared:<pinned-version>
  restart: unless-stopped
  profiles: ["edge"]
  command: ["tunnel", "--no-autoupdate", "run"]
  environment:
    TUNNEL_TOKEN: ${CLOUDFLARE_TUNNEL_TOKEN}
  depends_on:
    api:
      condition: service_healthy
```

- No published ports. The container only dials out.
- The tunnel is dashboard-managed; its single ingress rule maps
  `api.ziyutec.com` to `http://api:9229`. The token goes into the production
  env file as `CLOUDFLARE_TUNNEL_TOKEN` (and into
  `infra/env.production.example` as a placeholder).
- `infra/deploy.sh` adds `--profile edge` so deploys reconcile the tunnel
  with the rest of the stack.
- Cloudflare zone settings: SSL mode Full, the one DNS record proxied.
  Nothing else is configured at launch.

## 5. Application Hardening

- **CORS**: the origin list is hardcoded in `src/rag_agent_server/main.py` and
  currently ships `http://localhost:3000` / `http://localhost:5173` to
  production. Move it to a `CORS_ORIGINS` env var (comma-separated).
  Development default: current list. Production env file: only
  `https://chat.rag-agent.example` and `https://pis3.aempro.ca`. With
  `allow_credentials=True`, unused origins are pure attack surface.
- **Secrets**: production env file gets a long random `JWT_SECRET_KEY` and
  non-default Postgres and MinIO passwords, per
  `infra/env.production.example`.
- **Upload limit**: the application enforces 52 428 800 bytes; Cloudflare's
  100 MB body limit backstops it. No edge configuration required.

## 6. Boot Persistence

A consumer Windows PC reboots (power loss, Windows Update). The stack must
recover with no human present. Required chain:

1. **Windows**: Ollama installed to start at boot (service or startup task,
   not tied to interactive logon). Windows Update active hours set; reboots
   are accepted, not fought.
2. **WSL2 boot**: `/etc/wsl.conf` enables systemd (`[boot] systemd=true`).
   A Task Scheduler job runs at system startup ("run whether user is logged
   on or not") and executes a `wsl.exe` keep-alive so the distro boots
   headless.
3. **Resources**: `.wslconfig` pins a fixed memory budget for WSL2 so
   Milvus, Postgres, Redis, and the observability stack cannot starve
   Windows or the GPU workload.
4. **Docker**: `docker.service` enabled in the distro. Every compose service
   already carries `restart: unless-stopped`, so the stack self-assembles in
   dependency order; `cloudflared` re-establishes the tunnel on its own.
5. **Acceptance drill**: pull the plug once, on purpose. The system passes
   when `https://api.ziyutec.com/health` returns 200 within a defined window
   (target: 10 minutes) with zero manual steps. This drill is the acceptance
   test for the whole section.

## 7. Deploy And Rollback

The existing pipeline (`.github/workflows/deploy.yml` -> tailnet -> rsync ->
`infra/deploy.sh` with lock, Ollama gate, `--wait`) stays. Two additions:

- **Image pinning**: the deploy step sets `RAG_AGENT_IMAGE_TAG` to the git SHA.
  Rollback becomes "re-run the deploy workflow from the last good commit".
- **Edge smoke test**: the final workflow step curls
  `https://api.ziyutec.com/health` through the public edge, so a deploy that
  breaks DNS, the tunnel, or the API fails in CI instead of in front of
  users.

## 8. Backups And Disaster Recovery

All durable state lives in Docker volumes on one consumer machine. Backup
targets, in priority order:

- **Postgres** (auth, conversations, checkpoints): nightly `pg_dump` via a
  systemd timer in WSL2.
- **Milvus** (838k+ migrated chunks): scheduled backup with the
  `milvus-backup` tool, or cold copies of the `milvus_data`, `etcd_data`,
  and `minio_data` volumes. The preserved legacy Docker volumes plus the
  checkpoint at `.deploy/legacy-milvus-migration.json` remain the
  rebuild-of-last-resort via `tools/migrate_legacy_milvus.py`.
- **Redis**: AOF is already enabled; snapshot `redis_data` with the same
  timer. Losing it costs in-flight jobs only; documents re-ingest.
- **Off-host copy**: encrypted archives pushed to storage we already pay
  for — the cPanel account over SFTP is an acceptable zero-cost target for
  encrypted backup archives (static storage is within its sanctioned role).
  A backup on the same SSD as the database is not a backup.
- **Retention**: observability data (Loki, Tempo, Prometheus) is disposable;
  configure 14–30 day retention instead of backing it up.
- **Acceptance**: one restore rehearsed end-to-end into a scratch
  environment. An untested backup does not count.

## 9. Monitoring And Alerting

Observability (OTel -> Tempo/Loki/Prometheus/Grafana) exists; alerting does
not. Add:

- **External uptime probe** (free tier of UptimeRobot or healthchecks.io)
  against `https://api.ziyutec.com/health` — the only check that sees DNS, the
  tunnel, and the API the way users do.
- **Prometheus alert rules** on the canonical signals from `skills.md`:
  dead-letter stream growth, `XPENDING` age on `rag-agent-ingestion-workers`,
  API 5xx rate, collector health (`:13133`), and disk usage on the Milvus,
  MinIO, and Loki volumes.
- **Delivery**: Grafana alerting to company email (cPanel SMTP, already
  paid for).
- Debugging stays as documented in `skills.md`: Loki -> TraceID -> Tempo,
  `rag_agent_ingestion_*` metrics.

## 10. Rollout Phases

Phases 1–3 gate production launch. Phases 4–6 follow within days and do not
block it.

1. **DNS + edge.** Move `rag-agent.example` nameservers to Cloudflare, recreate
   records, create the tunnel, add the `cloudflared` compose service and
   deploy-script profile.
   *Accept when*: the cPanel app logs in, streams a chat response, and
   uploads a 40 MB document through `https://api.ziyutec.com`.
2. **App hardening.** Env-driven CORS, strong secrets verified in the
   server env file.
   *Accept when*: production `/health` works and a localhost-origin CORS
   preflight is rejected.
3. **Boot persistence.** wsl.conf, Task Scheduler job, `.wslconfig`, Ollama
   at boot.
   *Accept when*: the pull-the-plug drill passes hands-off.
4. **Backups.** Timers running; one restore rehearsed.
5. **Alerting.** External probe plus the Prometheus rules; a deliberately
   stopped worker confirmed to page.
6. **Deploy polish.** SHA-pinned image tags and the through-the-edge smoke
   step in CI.

## 11. Standing Rules

- Users never join the tailnet; Tailscale stays within its free tier for
  admin and CI identities only.
- Any future need for edge rate limiting, WAF rules, or additional public
  hostnames is a change to this plan and gets reviewed first.
- If Cloudflare is ever removed, the replacement must still satisfy: free,
  no inbound exposure, no per-user installs, streaming and 50 MB multipart
  uploads and `Authorization` headers proven.
