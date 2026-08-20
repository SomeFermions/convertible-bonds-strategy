#!/usr/bin/env bash
set -euo pipefail

export TZ=Asia/Shanghai
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if [ -f "${APP_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${APP_DIR}/.env"
    set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${APP_DIR}/logs"
LOCK_FILE="/tmp/cb_watchlist_realtime.lock"

mkdir -p "${LOG_DIR}"
exec >> "${LOG_DIR}/watchlist_realtime_$(date +%F).log" 2>&1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "$(date '+%F %T') watchlist realtime job is already running"
    exit 0
fi

cd "${APP_DIR}"
echo "$(date '+%F %T') starting watchlist realtime job"
exec "${PYTHON_BIN}" collect_watchlist_realtime.py \
    --rebuild-watchlist \
    --monitor \
    --skip-non-trading-day \
    --interval-seconds "${CB_REALTIME_INTERVAL_SECONDS:-10}" \
    --bar-minutes "${CB_REALTIME_BAR_MINUTES:-5}" \
    --metrics-refresh-seconds "${CB_REALTIME_METRICS_REFRESH_SECONDS:-300}" \
    --retain-live-days "${CB_REALTIME_RETAIN_LIVE_DAYS:-45}" \
    --watchlist-rebalance-days "${CB_WATCHLIST_REBALANCE_DAYS:-3}" \
    --observation-days "${CB_WATCHLIST_OBSERVATION_DAYS:-1}" \
    --stop-time "${CB_REALTIME_STOP_TIME:-15:05}" \
    --log-level "${CB_LOG_LEVEL:-INFO}"
