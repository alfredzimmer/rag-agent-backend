#!/usr/bin/env bash
#
# push-deploy.sh — deploy the RAG agent straight to the AI server from your
# workstation, bypassing GitHub Actions / Tailscale Funnel.
#
# It does what the CI "Deploy production" job does, minus the flaky public
# Funnel smoke test:
#   1. rsync this working tree to the server's deploy path
#   2. run infra/deploy.sh on the server (build image + reconcile compose stack)
#   3. health-check the API on the server's loopback
#   4. (optional) restart the host-run API + Streamlit UI so they pick up new code
#
# Run this ON YOUR WORKSTATION (not on the server):
#   ./infra/push-deploy.sh                 # deploy the docker compose stack
#   ./infra/push-deploy.sh --with-ui       # also restart the :9230 API + :8501 UI
#   ./infra/push-deploy.sh --dry-run       # show what rsync would change, deploy nothing
#
# Everything is overridable via environment variables (defaults match the
# current server layout):
#   AI_SERVER            ssh host/alias                (default: ai-server)
#   DEPLOY_PATH          deploy dir on the server      (default: /home/ziyutecc_ai_wsl/rag-agent-backend)
#   RAG_AGENT_ENV_FILE   prod env file on the server   (default: /home/ziyutecc_ai_wsl/.config/rag-agent/production.env)
#   RAG_AGENT_IMAGE_TAG  compose image tag             (default: git HEAD sha, +'-dirty' if uncommitted)
#   OLLAMA_HEALTH_URL    ollama health url on server   (default: http://127.0.0.1:11434/api/tags)
#   UI_API_PORT          host-run API port for --with-ui restart   (default: 9230)
#   UI_STREAMLIT_PORT    Streamlit port for --with-ui restart      (default: 8501)

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_SERVER="${AI_SERVER:-ai-server}"
DEPLOY_PATH="${DEPLOY_PATH:-/home/ziyutecc_ai_wsl/rag-agent-backend}"
RAG_AGENT_ENV_FILE="${RAG_AGENT_ENV_FILE:-/home/ziyutecc_ai_wsl/.config/rag-agent/production.env}"
OLLAMA_HEALTH_URL="${OLLAMA_HEALTH_URL:-http://127.0.0.1:11434/api/tags}"
UI_API_PORT="${UI_API_PORT:-9230}"
UI_STREAMLIT_PORT="${UI_STREAMLIT_PORT:-8501}"

DRY_RUN=0
WITH_UI=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --with-ui) WITH_UI=1 ;;
    -h|--help) grep -E '^#( |$)' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 'Unknown argument: %s (try --help)\n' "$arg" >&2; exit 2 ;;
  esac
done

# Image tag: git HEAD, marked -dirty when the tree has uncommitted changes.
if [[ -z "${RAG_AGENT_IMAGE_TAG:-}" ]]; then
  if sha="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)"; then
    git -C "$REPO_ROOT" diff --quiet && git -C "$REPO_ROOT" diff --cached --quiet \
      || sha="${sha}-dirty"
    RAG_AGENT_IMAGE_TAG="$sha"
  else
    RAG_AGENT_IMAGE_TAG="local"
  fi
fi

SSH=(ssh -o BatchMode=yes "$AI_SERVER")

log() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
[[ -f "$REPO_ROOT/infra/deploy.sh" ]] || die "infra/deploy.sh not found; run from the repo."
command -v rsync >/dev/null || die "rsync is required on this machine."

log "Target        : $AI_SERVER:$DEPLOY_PATH"
log "Prod env file : $RAG_AGENT_ENV_FILE"
log "Image tag     : $RAG_AGENT_IMAGE_TAG"
log "Checking SSH connectivity…"
"${SSH[@]}" -o ConnectTimeout=12 'echo "connected to $(hostname)"' \
  || die "cannot reach $AI_SERVER over SSH."

# Confirm the server has the tools deploy.sh needs and the prod env file exists.
"${SSH[@]}" bash -s -- "$DEPLOY_PATH" "$RAG_AGENT_ENV_FILE" <<'REMOTE' || die "server preflight failed."
set -Eeuo pipefail
install -d -m 0755 "$1"
for c in docker curl flock rsync; do command -v "$c" >/dev/null || { echo "missing: $c" >&2; exit 1; }; done
docker compose version >/dev/null
[[ -r "$2" ]] || { echo "prod env file not readable: $2" >&2; exit 1; }
REMOTE

# ---------------------------------------------------------------------------
# 1. Sync source (same excludes as CI, plus local-only cruft). Never ships the
#    repo .env — the server keeps its own .env for the host-run API.
# ---------------------------------------------------------------------------
RSYNC_OPTS=(--archive --compress --delete-delay --human-readable --prune-empty-dirs
  --exclude=.git/ --exclude=.env --exclude=.venv/ --exclude=.uv-cache/
  --exclude=.ruff_cache/ --exclude=.pytest_cache/ --exclude=.profiling_venv/
  --exclude=.runtime/ --exclude=.deploy/ --exclude=__pycache__/ --exclude='*.pyc'
  --exclude=.DS_Store --exclude=scratch/ --exclude=.claude/
  --filter=':- .gitignore'
  -e "ssh -o BatchMode=yes")
[[ "$DRY_RUN" == 1 ]] && RSYNC_OPTS+=(--dry-run --itemize-changes)

log "Syncing source to server$([[ "$DRY_RUN" == 1 ]] && printf ' (dry run)')…"
rsync "${RSYNC_OPTS[@]}" "$REPO_ROOT/" "$AI_SERVER:$DEPLOY_PATH/"

if [[ "$DRY_RUN" == 1 ]]; then
  log "Dry run complete — no deployment performed."
  exit 0
fi

# ---------------------------------------------------------------------------
# 2. Reconcile the docker compose stack via the server-side deploy.sh
# ---------------------------------------------------------------------------
log "Building image and reconciling the compose stack on the server…"
"${SSH[@]}" bash -s -- "$DEPLOY_PATH" "$RAG_AGENT_ENV_FILE" "$OLLAMA_HEALTH_URL" "$RAG_AGENT_IMAGE_TAG" <<'REMOTE'
set -Eeuo pipefail
cd "$1"
DEPLOY_ROOT="$1" RAG_AGENT_ENV_FILE="$2" OLLAMA_HEALTH_URL="$3" RAG_AGENT_IMAGE_TAG="$4" \
  bash infra/deploy.sh
REMOTE

# ---------------------------------------------------------------------------
# 3. (optional) Restart the host-run API + Streamlit so they run the new code.
#    These are plain background processes (no systemd), so we capture the exact
#    argv/cwd/port env from the running process, stop it, and relaunch it.
# ---------------------------------------------------------------------------
if [[ "$WITH_UI" == 1 ]]; then
  log "Restarting host-run API (:$UI_API_PORT) and Streamlit (:$UI_STREAMLIT_PORT)…"
  "${SSH[@]}" bash -s -- "$DEPLOY_PATH" "$UI_API_PORT" "$UI_STREAMLIT_PORT" <<'REMOTE'
set -Eeuo pipefail
DIR="$1"; API_PORT="$2"; ST_PORT="$3"
cd "$DIR"

# Refresh the host virtualenv so code + deps match the synced tree.
uv sync --group ui --group ingest --group dev

restart_bg() { # name  port  logfile  launch-cmd...
  local name="$1" port="$2" logf="$3"; shift 3
  local pids
  pids="$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u || true)"
  if [[ -n "$pids" ]]; then
    echo "  stopping $name (pids: $pids)"
    # kill the process group (covers the uv-run launcher + child)
    for p in $pids; do kill -TERM "-$(ps -o pgid= -p "$p" | tr -d ' ')" 2>/dev/null || kill -TERM "$p" 2>/dev/null || true; done
    sleep 3
    for p in $pids; do kill -KILL "$p" 2>/dev/null || true; done
  fi
  echo "  starting $name -> $logf"
  nohup "$@" >>"$logf" 2>&1 &
  disown || true
}

# Host API: it reads its port from the deploy-dir .env (RAG_AGENT_PORT); we also
# pass it explicitly so the restart is deterministic.
RAG_AGENT_PORT="$API_PORT" restart_bg "rag-agent-api" "$API_PORT" "$HOME/rag-agent-api-$API_PORT.log" \
  env RAG_AGENT_PORT="$API_PORT" uv run rag-agent-api

restart_bg "streamlit" "$ST_PORT" "$HOME/streamlit-$ST_PORT.log" \
  uv run streamlit run src/streamlit_app.py \
    --server.address 0.0.0.0 --server.port "$ST_PORT" --server.headless true

# Give them a moment and verify the API answers.
for i in $(seq 1 20); do
  if curl --fail --silent --max-time 5 "http://127.0.0.1:$API_PORT/health" >/dev/null; then
    echo "  host API healthy on :$API_PORT"; break
  fi
  sleep 2
  [[ "$i" == 20 ]] && { echo "  WARNING: host API did not become healthy on :$API_PORT" >&2; }
done
REMOTE
fi

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
log "Health summary:"
"${SSH[@]}" bash -s -- "$UI_API_PORT" <<'REMOTE' || true
set -uo pipefail
docker_api="$(curl --fail --silent --max-time 8 http://127.0.0.1:9229/health || echo 'unreachable')"
echo "  docker API  :9229  -> $docker_api"
host_api="$(curl --fail --silent --max-time 8 "http://127.0.0.1:$1/health" || echo 'unreachable')"
echo "  host API    :$1  -> $host_api"
REMOTE

log "Done. Deployed tag: $RAG_AGENT_IMAGE_TAG"
