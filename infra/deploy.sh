#!/usr/bin/env bash

set -Eeuo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${RAG_AGENT_ENV_FILE:?RAG_AGENT_ENV_FILE is required}"
COMPOSE_FILE="$DEPLOY_ROOT/infra/docker-compose.yaml"
STATE_DIR="${STATE_DIR:-$DEPLOY_ROOT/.deploy}"
OLLAMA_HEALTH_URL="${OLLAMA_HEALTH_URL:-http://127.0.0.1:11434/api/tags}"

compose=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE")

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'Required command is missing: %s\n' "$1" >&2
    exit 1
  }
}

env_file_value() {
  local line
  line="$(grep -E "^[[:space:]]*$1=" "$ENV_FILE" | tail -n 1 || true)"
  printf '%s' "${line#*=}"
}

require_command curl
require_command docker
require_command flock

[[ -r "$ENV_FILE" ]] || {
  printf 'Environment file is not readable: %s\n' "$ENV_FILE" >&2
  exit 1
}

case "$(env_file_value MINIO_ROOT_PASSWORD)" in
  ""|change-me|*replace-with-*)
    printf 'Environment file must set a real MINIO_ROOT_PASSWORD value.\n' >&2
    exit 1
    ;;
esac

mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/deploy.lock"
flock --nonblock 9 || {
  printf 'Another deployment is already running.\n' >&2
  exit 1
}

export RAG_AGENT_ENV_FILE="$ENV_FILE"
export RAG_AGENT_IMAGE_TAG="${RAG_AGENT_IMAGE_TAG:-local}"

# The API talks to Ollama on the host; fail before touching the stack if it is down.
curl --fail --silent --show-error --max-time 5 -- "$OLLAMA_HEALTH_URL" >/dev/null || {
  printf 'Ollama is not healthy at %s. Start it and retry.\n' "$OLLAMA_HEALTH_URL" >&2
  exit 1
}

"${compose[@]}" config --quiet

printf 'Pulling infrastructure images.\n'
"${compose[@]}" pull --policy always etcd minio milvus

printf 'Building the application image.\n'
"${compose[@]}" build api

printf 'Reconciling the stack.\n'
if ! "${compose[@]}" up --detach --no-build --remove-orphans --wait --wait-timeout 600; then
  printf 'Deployment failed.\n' >&2
  "${compose[@]}" ps || true
  "${compose[@]}" logs --no-color --tail 200 api || true
  exit 1
fi

"${compose[@]}" ps

port="$(env_file_value RAG_AGENT_HTTP_PORT)"
port="${RAG_AGENT_HTTP_PORT:-${port:-9229}}"
printf 'Checking local API health at http://127.0.0.1:%s/health.\n' "$port"
curl --fail --silent --show-error --max-time 10 -- "http://127.0.0.1:${port}/health" >/dev/null

printf 'Deployment completed successfully.\n'
