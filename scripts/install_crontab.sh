#!/usr/bin/env bash
set -euo pipefail

BEGIN_MARK="# BEGIN cb-system managed cron"
END_MARK="# END cb-system managed cron"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TMP_EXISTING="$(mktemp)"
TMP_NEW="$(mktemp)"

cleanup() {
    rm -f "${TMP_EXISTING}" "${TMP_NEW}"
}
trap cleanup EXIT

crontab -l > "${TMP_EXISTING}" 2>/dev/null || true

awk -v begin="${BEGIN_MARK}" -v end="${END_MARK}" '
    $0 == begin {skip = 1; next}
    $0 == end {skip = 0; next}
    !skip {print}
' "${TMP_EXISTING}" > "${TMP_NEW}"

if [ -s "${TMP_NEW}" ] && [ "$(tail -c 1 "${TMP_NEW}")" != "" ]; then
    printf "\n" >> "${TMP_NEW}"
fi

cat >> "${TMP_NEW}" <<CRON
# BEGIN cb-system managed cron
TZ=Asia/Shanghai
# Server timezone is UTC. These schedules correspond to Beijing trading workflows.
0 1 * * 1-5 ${SCRIPT_DIR}/run_watchlist_realtime.sh
10 1 * * 1-5 ${SCRIPT_DIR}/run_intraday_readiness_check.sh
*/5 1-3,5-7 * * 1-5 ${SCRIPT_DIR}/run_intraday_factors_incremental.sh
5 7 * * 1-5 ${SCRIPT_DIR}/run_daily_after_close.sh
# END cb-system managed cron
CRON

crontab "${TMP_NEW}"
crontab -l
