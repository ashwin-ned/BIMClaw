#!/usr/bin/env bash
# BIMClaw startup helper.
# Usage:
#   bash start.sh ollama-install   — pull the LLM model (first time)
#   bash start.sh up               — start Ollama + build + run Docker stack
#   bash start.sh down             — stop everything
#   bash start.sh logs             — follow backend logs
#   bash start.sh frontend-logs    — follow frontend logs

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5-coder:32b}"
OLLAMA_DATA="${OLLAMA_DATA:-/mnt/iml/checkpoints/ollama}"
OLLAMA_BIN="${OLLAMA_BIN:-$(which ollama 2>/dev/null || echo /usr/local/bin/ollama)}"
LOG_DIR="${SCRIPT_DIR}/.logs"

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------

ollama_start() {
    if pgrep -x ollama > /dev/null; then
        echo "Ollama already running."
        return
    fi
    echo "=== Starting Ollama (models at $OLLAMA_DATA/models) ==="
    mkdir -p "$OLLAMA_DATA/models"
    OLLAMA_HOST=0.0.0.0 OLLAMA_MODELS="$OLLAMA_DATA/models" nohup "$OLLAMA_BIN" serve \
        > "$OLLAMA_DATA/ollama.log" 2>&1 &
    echo "Ollama PID=$!"
    for i in $(seq 1 30); do
        curl -sf http://localhost:11434/ > /dev/null 2>&1 && echo "Ollama ready." && return
        sleep 2
    done
    echo "WARNING: Ollama didn't respond — check $OLLAMA_DATA/ollama.log"
}

# ---------------------------------------------------------------------------

case "${1:-help}" in

ollama-install)
    if ! command -v "$OLLAMA_BIN" &>/dev/null; then
        echo "ERROR: ollama not found. Install: curl -fsSL https://ollama.com/install.sh | sh"
        exit 1
    fi
    mkdir -p "$OLLAMA_DATA/models"
    if ! pgrep -x ollama > /dev/null; then
        OLLAMA_MODELS="$OLLAMA_DATA/models" "$OLLAMA_BIN" serve &
        sleep 5
    fi
    echo "=== Pulling $OLLAMA_MODEL ==="
    OLLAMA_HOST=http://localhost:11434 "$OLLAMA_BIN" pull "$OLLAMA_MODEL"
    echo "Done. Run 'bash start.sh up' to start BIMClaw."
    ;;

up)
    ollama_start

    echo "=== Building + starting Docker stack ==="
    OLLAMA_MODEL="$OLLAMA_MODEL" docker compose up --build -d

    echo ""
    echo "  Frontend : http://localhost:3000"
    echo "  Backend  : http://localhost:8000"
    echo "  Model    : $OLLAMA_MODEL"
    echo ""
    echo "Logs: bash start.sh logs"
    ;;

down)
    docker compose down 2>/dev/null && echo "Docker stack stopped." || true
    pkill -x ollama 2>/dev/null && echo "Ollama stopped." || true
    ;;

logs)
    docker compose logs -f backend
    ;;

frontend-logs)
    docker compose logs -f frontend
    ;;

restart)
    docker compose restart "${2:-backend}"
    ;;

*)
    cat <<EOF
Usage: bash start.sh <command>

  ollama-install   Pull the LLM model to /mnt/iml/checkpoints/ollama (first time)
  up               Start Ollama + build + run Docker stack
  down             Stop Docker stack + Ollama
  logs             Follow backend logs
  frontend-logs    Follow frontend logs
  restart [svc]    Restart a service (default: backend)

Environment variables:
  OLLAMA_MODEL   (default: qwen2.5-coder:32b)
  OLLAMA_DATA    (default: /mnt/iml/checkpoints/ollama)
EOF
    ;;

esac
