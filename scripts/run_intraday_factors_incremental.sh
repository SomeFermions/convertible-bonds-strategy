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
LOCK_FILE="/tmp/cb_intraday_factors.lock"
LOG_FILE="${LOG_DIR}/intraday_factors_$(date +%F).log"

mkdir -p "${LOG_DIR}"
export CB_JOB_LOG_PATH="${LOG_FILE}"
exec >> "${LOG_FILE}" 2>&1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "$(date '+%F %T') intraday factor job is already running"
    cd "${APP_DIR}"
    "${PYTHON_BIN}" run_intraday_factors.py \
        --mode record-locked \
        --log-path "${LOG_FILE}" \
        --lock-file "${LOCK_FILE}" \
        --log-level "${CB_LOG_LEVEL:-INFO}" || true
    exit 0
fi

cd "${APP_DIR}"
echo "$(date '+%F %T') starting intraday factor incremental job"
set +e
"${PYTHON_BIN}" run_intraday_factors.py \
    --mode incremental \
    --write \
    --log-path "${LOG_FILE}" \
    --lock-file "${LOCK_FILE}" \
    --log-level "${CB_LOG_LEVEL:-INFO}"
status=$?
set -e
echo "$(date '+%F %T') finished intraday factor incremental job exit_code=${status}"
exit "${status}"
