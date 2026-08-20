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
LOCK_FILE="/tmp/cb_daily_after_close.lock"

mkdir -p "${LOG_DIR}"
exec >> "${LOG_DIR}/daily_after_close_$(date +%F).log" 2>&1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "$(date '+%F %T') daily after-close job is already running"
    exit 0
fi

cd "${APP_DIR}"
echo "$(date '+%F %T') starting daily after-close job"
exec "${PYTHON_BIN}" collect_cb_daily.py \
    --today \
    --skip-non-trading-day \
    --sleep-seconds "${CB_DAILY_SLEEP_SECONDS:-1.5}" \
    --log-level "${CB_LOG_LEVEL:-INFO}"
