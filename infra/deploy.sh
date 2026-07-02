#!/usr/bin/env bash

set -Eeuo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${EDEMI_ENV_FILE:?EDEMI_ENV_FILE is required}"
OLLAMA_HEALTH_URL="${OLLAMA_HEALTH_URL:-http://127.0.0.1:11434/api/tags}"
STATE_DIR="$DEPLOY_ROOT/.deploy"
COMPOSE_FILE="$DEPLOY_ROOT/infra/docker-compose.yaml"

compose=(
  docker compose
  --env-file "$ENV_FILE"
  --profile edge
  --profile observability
  --profile tools
  -f "$COMPOSE_FILE"
)

infrastructure_services=(
  cloudflared
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

require_command curl
require_command docker
require_command flock

[[ -r "$ENV_FILE" ]] || {
  printf 'Production environment file is not readable: %s\n' "$ENV_FILE" >&2
  exit 1
}

require_env_file_value CLOUDFLARE_TUNNEL_TOKEN
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
export EDEMI_ENV_FILE="$ENV_FILE"
export EDEMI_IMAGE_TAG="${EDEMI_IMAGE_TAG:-local}"

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
printf 'Deployment completed successfully.\n'
