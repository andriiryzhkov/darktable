#!/bin/bash
#
# NOVA memory benchmark.
# Measures RSS of darktable-nova (direct mode) and darktable-server (IPC mode).
#
# Usage:
#   bench_memory.sh [OPTIONS]
#     --build-dir DIR     Build directory (default: ./build)
#     --configdir DIR     Config directory (default: /tmp/dt-bench-config)
#     --settle SECS       Seconds to wait after launch for process to settle (default: 5)
#     --json              Output as JSON
#     --baseline PATH     Compare against baseline JSON file
#     --threshold PCT     Regression threshold percentage (default: 20)

set -euo pipefail

BUILD_DIR="./build"
CONFIGDIR="/tmp/dt-bench-config"
SETTLE=5
JSON_OUTPUT=0
BASELINE=""
THRESHOLD=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)   BUILD_DIR="$2"; shift 2 ;;
    --configdir)   CONFIGDIR="$2"; shift 2 ;;
    --settle)      SETTLE="$2"; shift 2 ;;
    --json)        JSON_OUTPUT=1; shift ;;
    --baseline)    BASELINE="$2"; shift 2 ;;
    --threshold)   THRESHOLD="$2"; shift 2 ;;
    --help)
      sed -n '3,12p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

NOVA_BIN="${BUILD_DIR}/bin/darktable-nova"
SERVER_BIN="${BUILD_DIR}/bin/darktable-server"

# Platform-specific RSS measurement (in KB)
get_rss_kb() {
  local pid=$1
  case "$(uname)" in
    Darwin) ps -o rss= -p "$pid" | tr -d ' ' ;;
    Linux)  awk '/VmRSS/ {print $2}' "/proc/$pid/status" 2>/dev/null || ps -o rss= -p "$pid" | tr -d ' ' ;;
    *)      ps -o rss= -p "$pid" | tr -d ' ' ;;
  esac
}

# CPU usage measurement: sample %CPU over a duration
# Returns average CPU% over the sampling period
get_cpu_pct() {
  local pid=$1
  local duration=${2:-3}
  local samples=0
  local total=0
  for i in $(seq 1 "$duration"); do
    local cpu
    cpu=$(ps -o %cpu= -p "$pid" 2>/dev/null | tr -d ' ')
    if [[ -n "$cpu" ]]; then
      total=$(echo "$total + $cpu" | bc)
      samples=$((samples + 1))
    fi
    sleep 1
  done
  if [[ $samples -gt 0 ]]; then
    echo "scale=1; $total / $samples" | bc
  else
    echo "0"
  fi
}

cleanup() {
  # Kill any processes we started
  [[ -n "${NOVA_PID:-}" ]] && kill "$NOVA_PID" 2>/dev/null && wait "$NOVA_PID" 2>/dev/null || true
  [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null || true
  rm -f /tmp/dt-bench.sock
}
trap cleanup EXIT

echo "NOVA Memory Benchmark"
echo "===================="
echo ""

# --- Check binaries ---
if [[ ! -x "$NOVA_BIN" ]]; then
  echo "ERROR: $NOVA_BIN not found. Build first with: cmake --build build --target darktable-nova"
  exit 1
fi

# --- Direct mode (darktable-nova standalone) ---
echo "=== Direct Mode (darktable-nova) ==="
mkdir -p "$CONFIGDIR"

"$NOVA_BIN" --configdir "$CONFIGDIR" &>/dev/null &
NOVA_PID=$!
sleep "$SETTLE"

CPU_DIRECT="0"
if kill -0 "$NOVA_PID" 2>/dev/null; then
  RSS_DIRECT=$(get_rss_kb "$NOVA_PID")
  echo "  PID: $NOVA_PID"
  echo "  RSS: $((RSS_DIRECT / 1024)) MB ($RSS_DIRECT KB)"
  echo "  Measuring idle CPU (3s)..."
  CPU_DIRECT=$(get_cpu_pct "$NOVA_PID" 3)
  echo "  CPU (idle): ${CPU_DIRECT}%"
  kill "$NOVA_PID" 2>/dev/null
  wait "$NOVA_PID" 2>/dev/null || true
  unset NOVA_PID
else
  echo "  ERROR: darktable-nova exited prematurely"
  RSS_DIRECT=0
  unset NOVA_PID
fi

echo ""

# --- IPC mode (darktable-server + darktable-nova --server) ---
RSS_SERVER=0
RSS_HOST=0
RSS_IPC_TOTAL=0

if [[ -x "$SERVER_BIN" ]]; then
  echo "=== IPC Mode (server + host) ==="
  SOCK_PATH="/tmp/dt-bench.sock"
  rm -f "$SOCK_PATH"

  "$SERVER_BIN" --socket "$SOCK_PATH" --configdir "$CONFIGDIR" --library ":memory:" &>/dev/null &
  SERVER_PID=$!

  # Wait for socket
  for i in $(seq 1 50); do
    [[ -S "$SOCK_PATH" ]] && break
    sleep 0.1
  done

  if [[ -S "$SOCK_PATH" ]]; then
    "$NOVA_BIN" --server --socket "$SOCK_PATH" --configdir "$CONFIGDIR" &>/dev/null &
    NOVA_PID=$!
    sleep "$SETTLE"

    CPU_HOST="0"
    CPU_SERVER="0"
    if kill -0 "$NOVA_PID" 2>/dev/null && kill -0 "$SERVER_PID" 2>/dev/null; then
      RSS_HOST=$(get_rss_kb "$NOVA_PID")
      RSS_SERVER=$(get_rss_kb "$SERVER_PID")
      RSS_IPC_TOTAL=$((RSS_HOST + RSS_SERVER))
      echo "  Host PID:   $NOVA_PID   RSS: $((RSS_HOST / 1024)) MB ($RSS_HOST KB)"
      echo "  Server PID: $SERVER_PID   RSS: $((RSS_SERVER / 1024)) MB ($RSS_SERVER KB)"
      echo "  IPC Total:               RSS: $((RSS_IPC_TOTAL / 1024)) MB ($RSS_IPC_TOTAL KB)"
      echo "  Measuring idle CPU (3s)..."
      CPU_HOST=$(get_cpu_pct "$NOVA_PID" 3)
      CPU_SERVER=$(get_cpu_pct "$SERVER_PID" 3)
      echo "  Host CPU (idle): ${CPU_HOST}%"
      echo "  Server CPU (idle): ${CPU_SERVER}%"
    else
      echo "  ERROR: one or both processes exited prematurely"
    fi

    [[ -n "${NOVA_PID:-}" ]] && kill "$NOVA_PID" 2>/dev/null && wait "$NOVA_PID" 2>/dev/null || true
    unset NOVA_PID
    kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null || true
    unset SERVER_PID
  else
    echo "  ERROR: server socket not created"
    kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null || true
    unset SERVER_PID
  fi
else
  echo "=== IPC Mode: SKIPPED (darktable-server not found) ==="
fi

echo ""

# --- Summary ---
echo "=== Summary ==="
printf "  %-20s %8s %8s\n" "Mode" "RSS (MB)" "CPU (%)"
printf "  %-20s %8s %8s\n" "--------------------" "--------" "--------"
[[ $RSS_DIRECT -gt 0 ]] && printf "  %-20s %8d %8s\n" "Direct" "$((RSS_DIRECT / 1024))" "${CPU_DIRECT}"
[[ $RSS_IPC_TOTAL -gt 0 ]] && printf "  %-20s %8d %8s\n" "IPC (host)" "$((RSS_HOST / 1024))" "${CPU_HOST}"
[[ $RSS_IPC_TOTAL -gt 0 ]] && printf "  %-20s %8d %8s\n" "IPC (server)" "$((RSS_SERVER / 1024))" "${CPU_SERVER}"
[[ $RSS_IPC_TOTAL -gt 0 ]] && printf "  %-20s %8d %8s\n" "IPC (total)" "$((RSS_IPC_TOTAL / 1024))" "$(echo "$CPU_HOST + $CPU_SERVER" | bc)"
if [[ $RSS_DIRECT -gt 0 && $RSS_IPC_TOTAL -gt 0 ]]; then
  OVERHEAD=$(echo "scale=1; $RSS_IPC_TOTAL * 100 / $RSS_DIRECT" | bc)
  printf "  %-20s %7s%%\n" "IPC overhead" "$OVERHEAD"
fi
echo ""

# --- JSON output ---
if [[ $JSON_OUTPUT -eq 1 ]]; then
  cat <<ENDJSON
{
  "direct_rss_kb": $RSS_DIRECT,
  "direct_cpu_pct": ${CPU_DIRECT},
  "ipc_host_rss_kb": $RSS_HOST,
  "ipc_host_cpu_pct": ${CPU_HOST:-0},
  "ipc_server_rss_kb": $RSS_SERVER,
  "ipc_server_cpu_pct": ${CPU_SERVER:-0},
  "ipc_total_rss_kb": $RSS_IPC_TOTAL
}
ENDJSON
fi

# --- Baseline comparison ---
if [[ -n "$BASELINE" && -f "$BASELINE" ]]; then
  echo "=== Regression Check (threshold: ${THRESHOLD}%) ==="
  FAILURES=0

  if command -v python3 &>/dev/null; then
    FAILURES=$(python3 -c "
import json, sys
with open('$BASELINE') as f:
    base = json.load(f)
threshold = $THRESHOLD
failures = 0
actual = {
    'direct_rss_kb': $RSS_DIRECT,
    'ipc_total_rss_kb': $RSS_IPC_TOTAL,
}
for key, val in actual.items():
    if key in base and base[key] > 0 and val > 0:
        pct = (val - base[key]) / base[key] * 100
        if pct > threshold:
            print(f'[REGRESSION] {key}: {base[key]} -> {val} (+{pct:.0f}%)', file=sys.stderr)
            failures += 1
        else:
            print(f'[OK] {key}: {val} (baseline {base[key]}, {pct:+.0f}%)')
print(failures)
")
  else
    echo "  python3 not found, skipping regression check"
  fi

  [[ $FAILURES -gt 0 ]] && exit 1
fi
