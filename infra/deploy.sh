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
  "${compose[@]}" logs --no-color --tail 200 || true
}

require_command curl
require_command docker
require_command flock

[[ -r "$ENV_FILE" ]] || {
  printf 'Production environment file is not readable: %s\n' "$ENV_FILE" >&2
  exit 1
}

mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/deploy.lock"
flock --nonblock 9 || {
  printf 'Another deployment is already running.\n' >&2
  exit 1
}

export DEPLOYMENT_ENVIRONMENT="${DEPLOYMENT_ENVIRONMENT:-production}"
export EDEMI_ENV_FILE="$ENV_FILE"

if ! ollama_healthy && ! start_ollama; then
  printf 'Ollama did not become healthy at %s.\n' "$OLLAMA_HEALTH_URL" >&2
  [[ -f "$STATE_DIR/ollama.log" ]] && tail -n 100 "$STATE_DIR/ollama.log" >&2
  exit 1
fi

printf 'Ollama is healthy at %s.\n' "$OLLAMA_HEALTH_URL"

"${compose[@]}" config --quiet

printf 'Pulling updated infrastructure images.\n'
"${compose[@]}" pull --policy always "${infrastructure_services[@]}"

printf 'Building the application and reconciling the complete stack.\n'
if ! "${compose[@]}" up \
  --detach \
  --build \
  --remove-orphans \
  --wait \
  --wait-timeout 600; then
  printf 'Deployment failed.\n' >&2
  show_diagnostics
  exit 1
fi

"${compose[@]}" ps
printf 'Deployment completed successfully.\n'
