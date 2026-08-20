#!/usr/bin/env bash
set -euo pipefail

export TZ=UTC
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if [ -f "${APP_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${APP_DIR}/.env"
    set +a
fi

export TDENGINE_HOST="${TDENGINE_HOST:-localhost}"
export TDENGINE_PORT="${TDENGINE_PORT:-6030}"
export TDENGINE_USER="${TDENGINE_USER:-root}"
export TDENGINE_PASSWORD="${TDENGINE_PASSWORD:-}"
export TDENGINE_DATABASE="${TDENGINE_DATABASE:-cb_dev}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${APP_DIR}/logs"
LOG_FILE="${LOG_DIR}/intraday_readiness_$(date +%F).log"

mkdir -p "${LOG_DIR}"
exec >> "${LOG_FILE}" 2>&1

cd "${APP_DIR}"
echo "$(date '+%F %T') starting intraday readiness check"
set +e
"${PYTHON_BIN}" run_intraday_factors.py \
    --mode readiness-check \
    --date "$(TZ=Asia/Shanghai date +%F)" \
    --trading-days "${CB_HISTORY_TRADING_DAYS:-45}" \
    --log-path "${LOG_FILE}" \
    --log-level "${CB_LOG_LEVEL:-INFO}"
status=$?
set -e
echo "$(date '+%F %T') finished intraday readiness check exit_code=${status}"
exit "${status}"
