#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PORT="${SERVICE_PORT:-8000}"
BIND_ADDRESS=""
GPU_ENABLED=1
DETACH=0
BUILD_ON_UP=1
ACTION="up"
BACKEND="${GFM_BACKEND:-vggt}"
DOCKER_GPUS_VALUE="${DOCKER_GPUS:-all}"
SHM_SIZE_VALUE="${SHM_SIZE:-8gb}"
SERVICE_NAME="gfm-serve"
COMMON_IMAGE="${GFM_COMMON_IMAGE:-gfm-serve-core:latest}"

usage() {
  cat <<'EOF'
Usage:
  scripts/docker_compose.sh [action] [options]

Actions:
  up         Build and start the service. Default action.
  down       Stop the service and remove containers.
  build      Build the service image only.
  logs       Follow service logs.
  ps         Show compose service status.

Options:
  -p, --port PORT     Forward this host/container port. Default: 8000
  --backend ID        Service directory name. Default: vggt
  --common-image TAG  Shared base image tag. Default: gfm-serve-core:latest
  --cpu               Use the base compose file only and skip GPU settings.
  --gpus VALUE        GPU selector passed to compose. Default: all
  --shm-size VALUE    Shared memory size for GPU compose. Default: 8gb
  -d, --detach        Run 'up' in detached mode.
  --no-build          Skip '--build' on 'up'.
  -h, --help          Show this help text.

Examples:
  scripts/docker_compose.sh up --backend vggt --port 9000
  scripts/docker_compose.sh up --backend vggt --port 9000 --cpu
  scripts/docker_compose.sh logs --backend vggt --port 9000
  scripts/docker_compose.sh down
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    up|down|build|logs|ps)
      ACTION="$1"
      shift
      ;;
    -p|--port)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for $1" >&2
        exit 1
      fi
      PORT="$2"
      shift 2
      ;;
    --backend)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for $1" >&2
        exit 1
      fi
      BACKEND="$2"
      shift 2
      ;;
    --common-image)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for $1" >&2
        exit 1
      fi
      COMMON_IMAGE="$2"
      shift 2
      ;;
    --cpu)
      GPU_ENABLED=0
      shift
      ;;
    --gpus)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for $1" >&2
        exit 1
      fi
      DOCKER_GPUS_VALUE="$2"
      shift 2
      ;;
    --shm-size)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for $1" >&2
        exit 1
      fi
      SHM_SIZE_VALUE="$2"
      shift 2
      ;;
    -d|--detach)
      DETACH=1
      shift
      ;;
    --no-build)
      BUILD_ON_UP=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
  echo "Port must be an integer between 1 and 65535: $PORT" >&2
  exit 1
fi

is_ipv4_address() {
  local address="$1"
  local numeric_octet
  local octet
  local IFS=.
  local -a octets

  if ! [[ "$address" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    return 1
  fi

  read -r -a octets <<<"$address"
  if [ "${#octets[@]}" -ne 4 ]; then
    return 1
  fi
  for octet in "${octets[@]}"; do
    numeric_octet=$((10#$octet))
    if [ "$numeric_octet" -gt 255 ]; then
      return 1
    fi
  done
}

resolve_bind_address() {
  local detected_addresses

  if ! command -v tailscale >/dev/null 2>&1; then
    echo "Tailscale is required but the 'tailscale' command was not found." >&2
    echo "Install Tailscale and connect this host to a tailnet, then try again." >&2
    exit 1
  fi

  if ! detected_addresses="$(tailscale ip -4 2>/dev/null)"; then
    echo "Tailscale is installed but is not connected." >&2
    echo "Connect it with 'tailscale up', then try again." >&2
    exit 1
  fi
  BIND_ADDRESS="${detected_addresses%%$'\n'*}"

  if [ -z "$BIND_ADDRESS" ] || ! is_ipv4_address "$BIND_ADDRESS"; then
    echo "Could not detect a Tailscale IPv4 address." >&2
    echo "Connect Tailscale, then try again." >&2
    exit 1
  fi
}

DOCKERFILE_PATH="services/${BACKEND}/Dockerfile"
if [ ! -f "${ROOT_DIR}/${DOCKERFILE_PATH}" ]; then
  echo "Backend Dockerfile not found: ${DOCKERFILE_PATH}" >&2
  exit 1
fi

if [ ! -f "${ROOT_DIR}/services/${BACKEND}/pyproject.toml" ]; then
  echo "Backend package not found: services/${BACKEND}/pyproject.toml" >&2
  exit 1
fi

if [ ! -f "${ROOT_DIR}/services/${BACKEND}/upstream/pyproject.toml" ]; then
  echo "${BACKEND} upstream assets are missing. Run: git submodule update --init --recursive" >&2
  exit 1
fi

COMPOSE_ARGS=(-f docker-compose.yml)
if [ "$GPU_ENABLED" -eq 1 ]; then
  COMPOSE_ARGS+=(-f docker-compose.gpu.yml)
fi

resolve_bind_address

export SERVICE_PORT="$PORT"
export GFM_SERVE_BIND_ADDRESS="$BIND_ADDRESS"
export GFM_BACKEND="$BACKEND"
export GFM_COMMON_IMAGE="$COMMON_IMAGE"
export DOCKERFILE_PATH
export SERVICE_IMAGE="gfm-serve:${BACKEND}"
export DOCKER_GPUS="$DOCKER_GPUS_VALUE"
export SHM_SIZE="$SHM_SIZE_VALUE"

cd "$ROOT_DIR"

build_common_base() {
  docker build \
    -f docker/Dockerfile.common \
    -t "${COMMON_IMAGE}" \
    .
}

case "$ACTION" in
  up)
    echo "Binding GFM Serve to address: http://${BIND_ADDRESS}:${PORT}"
    if [ "$BUILD_ON_UP" -eq 1 ]; then
      build_common_base
    fi
    UP_ARGS=(up)
    if [ "$BUILD_ON_UP" -eq 1 ]; then
      UP_ARGS+=(--build)
    fi
    if [ "$DETACH" -eq 1 ]; then
      UP_ARGS+=(-d)
    fi
    exec docker compose "${COMPOSE_ARGS[@]}" "${UP_ARGS[@]}"
    ;;
  down)
    exec docker compose "${COMPOSE_ARGS[@]}" down
    ;;
  build)
    build_common_base
    exec docker compose "${COMPOSE_ARGS[@]}" build
    ;;
  logs)
    exec docker compose "${COMPOSE_ARGS[@]}" logs -f "${SERVICE_NAME}"
    ;;
  ps)
    exec docker compose "${COMPOSE_ARGS[@]}" ps
    ;;
esac
