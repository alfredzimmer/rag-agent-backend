#!/usr/bin/env bash

set -Eeuo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${RAG_AGENT_ENV_FILE:?RAG_AGENT_ENV_FILE is required}"
OLLAMA_HEALTH_URL="${OLLAMA_HEALTH_URL:-http://127.0.0.1:11434/api/tags}"
STATE_DIR="${STATE_DIR:-$DEPLOY_ROOT/.deploy}"
COMPOSE_FILE="$DEPLOY_ROOT/infra/docker-compose.yaml"

compose=(
  docker compose
  --env-file "$ENV_FILE"
  --profile observability
  --profile tools
  -f "$COMPOSE_FILE"
)

infrastructure_services=(
  redis
  postgres
  etcd
  minio
  milvus
  attu
  otel-collector
  prometheus
  tempo
  loki
  grafana
)

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'Required command is missing: %s\n' "$1" >&2
    exit 1
  }
}

env_file_value() {
  local key="$1"
  local line

  line="$(grep -E "^[[:space:]]*${key}=" "$ENV_FILE" | tail -n 1 || true)"
  line="${line#*=}"
  line="${line%$'\r'}"
  line="${line%\"}"
  line="${line#\"}"
  line="${line%\'}"
  line="${line#\'}"
  printf '%s' "$line"
}

require_env_file_value() {
  local key="$1"
  local value

  value="$(env_file_value "$key")"
  case "$value" in
    ""|replace-*|generate-*|change-me|thisisdifferentontheserver|*replace-with-*|*change-me*)
      printf 'Production environment file must set a real %s value.\n' "$key" >&2
      exit 1
      ;;
  esac
}

env_flag_enabled() {
  local key="$1"
  local default="$2"
  local value

  value="${!key:-}"
  [[ -z "$value" ]] && value="$(env_file_value "$key")"
  value="${value:-$default}"

  case "$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      return 0
      ;;
    0|false|no|off)
      return 1
      ;;
    *)
      printf '%s must be true or false, got: %s\n' "$key" "$value" >&2
      exit 1
      ;;
  esac
}

api_port() {
  local value

  value="${RAG_AGENT_HTTP_PORT:-}"
  [[ -z "$value" ]] && value="$(env_file_value RAG_AGENT_HTTP_PORT)"
  printf '%s' "${value:-9229}"
}

tailscale_funnel_target() {
  local value

  value="${TAILSCALE_FUNNEL_TARGET:-}"
  [[ -z "$value" ]] && value="$(env_file_value TAILSCALE_FUNNEL_TARGET)"
  printf '%s' "${value:-$(api_port)}"
}

require_numeric_port() {
  local key="$1"
  local value="$2"

  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    printf '%s must be a numeric TCP port, got: %s\n' "$key" "$value" >&2
    exit 1
  fi
}

ollama_healthy() {
  curl --fail --silent --show-error --max-time 5 -- "$OLLAMA_HEALTH_URL" >/dev/null
}

wait_for_ollama() {
  local attempts="${1:-12}"

  for _ in $(seq 1 "$attempts"); do
    if ollama_healthy; then
      return 0
    fi
    sleep 5
  done

  return 1
}

start_ollama() {
  printf 'Ollama is unavailable at %s; attempting to start it.\n' "$OLLAMA_HEALTH_URL"

  if command -v systemctl >/dev/null 2>&1; then
    if systemctl start ollama >/dev/null 2>&1; then
      printf 'Started Ollama through the system service.\n'
    elif command -v sudo >/dev/null 2>&1 \
      && sudo -n systemctl start ollama >/dev/null 2>&1; then
      printf 'Started Ollama through the system service with sudo.\n'
    elif systemctl --user start ollama >/dev/null 2>&1; then
      printf 'Started Ollama through the user service.\n'
    fi
  fi

  if wait_for_ollama 6; then
    return 0
  fi

  if command -v ollama >/dev/null 2>&1; then
    printf 'Starting Ollama directly with ollama serve.\n'
    nohup ollama serve >"$STATE_DIR/ollama.log" 2>&1 </dev/null &
    disown || true
  else
    printf 'Ollama is not installed and no Ollama service could be started.\n' >&2
    return 1
  fi

  wait_for_ollama 12
}

show_diagnostics() {
  "${compose[@]}" ps || true
  "${compose[@]}" logs --no-color --tail 200 api ingestion-worker || true
}

wait_for_public_health() {
  local url="$1"
  local attempts="${2:-12}"

  for _ in $(seq 1 "$attempts"); do
    if curl --fail --silent --show-error --max-time 10 -- "$url" >/dev/null; then
      return 0
    fi
    sleep 10
  done

  return 1
}

ensure_tailscale_funnel() {
  if ! env_flag_enabled TAILSCALE_FUNNEL_ENABLED true; then
    printf 'Tailscale Funnel is disabled by TAILSCALE_FUNNEL_ENABLED.\n'
    return 0
  fi

  require_command tailscale

  if ! tailscale status >/dev/null 2>&1; then
    printf 'Tailscale is not running or this host is not logged in.\n' >&2
    exit 1
  fi

  local port
  local target
  local local_health_url
  local public_health_url
  local status

  port="$(api_port)"
  require_numeric_port RAG_AGENT_HTTP_PORT "$port"
  target="$(tailscale_funnel_target)"
  local_health_url="http://127.0.0.1:${port}/health"
  public_health_url="${PUBLIC_HEALTH_URL:-}"

  printf 'Checking local API health at %s.\n' "$local_health_url"
  curl --fail --silent --show-error --max-time 10 -- "$local_health_url" >/dev/null

  printf 'Ensuring Tailscale Funnel exposes local API target %s.\n' "$target"
  tailscale funnel --bg --yes "$target"

  status="$(tailscale funnel status 2>&1)"
  printf 'Tailscale Funnel status:\n%s\n' "$status"

  if [[ "$target" == "$port" ]] && ! grep -Eq "(127\\.0\\.0\\.1|localhost):${port}" <<<"$status"; then
    printf 'Tailscale Funnel status does not show the expected local API port %s.\n' "$port" >&2
    exit 1
  fi

  if [[ -n "$public_health_url" ]]; then
    printf 'Checking public API health at %s.\n' "$public_health_url"
    if ! wait_for_public_health "$public_health_url" 12; then
      printf 'Public API health check failed at %s.\n' "$public_health_url" >&2
      exit 1
    fi
  else
    printf 'PUBLIC_HEALTH_URL is not set; skipping public health check in deploy.sh.\n'
    printf 'Run `tailscale funnel status` and set PRODUCTION_PUBLIC_HEALTH_URL to the printed URL plus /health.\n'
  fi
}

require_command curl
require_command docker
require_command flock

[[ -r "$ENV_FILE" ]] || {
  printf 'Production environment file is not readable: %s\n' "$ENV_FILE" >&2
  exit 1
}

require_env_file_value JWT_SECRET_KEY
require_env_file_value POSTGRES_PASSWORD
require_env_file_value PG_URI
require_env_file_value MINIO_ROOT_PASSWORD

mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/deploy.lock"
flock --nonblock 9 || {
  printf 'Another deployment is already running.\n' >&2
  exit 1
}

export DEPLOYMENT_ENVIRONMENT="${DEPLOYMENT_ENVIRONMENT:-production}"
export RAG_AGENT_ENV_FILE="$ENV_FILE"
export RAG_AGENT_IMAGE_TAG="${RAG_AGENT_IMAGE_TAG:-local}"

if ! ollama_healthy && ! start_ollama; then
  printf 'Ollama did not become healthy at %s.\n' "$OLLAMA_HEALTH_URL" >&2
  [[ -f "$STATE_DIR/ollama.log" ]] && tail -n 100 "$STATE_DIR/ollama.log" >&2
  exit 1
fi

printf 'Ollama is healthy at %s.\n' "$OLLAMA_HEALTH_URL"

"${compose[@]}" config --quiet

printf 'Pulling updated infrastructure images.\n'
"${compose[@]}" pull --policy always "${infrastructure_services[@]}"

printf 'Building the application image.\n'
if ! "${compose[@]}" build api; then
  printf 'Application build failed.\n' >&2
  exit 1
fi

printf 'Reconciling the complete stack.\n'
if ! "${compose[@]}" up \
  --detach \
  --no-build \
  --remove-orphans \
  --wait \
  --wait-timeout 600; then
  printf 'Deployment failed.\n' >&2
  show_diagnostics
  exit 1
fi

"${compose[@]}" ps
ensure_tailscale_funnel
printf 'Deployment completed successfully.\n'
