#!/usr/bin/env bash
set -euo pipefail

DEB_PATH="${1:-${DEB_PATH:-${HOME}/Downloads/tdengine-tsdb-oss-3.4.1.13-linux-x64.deb}}"
BACKUP_ROOT="${BACKUP_ROOT:-${HOME}/backups}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${BACKUP_ROOT}/tdengine_oss_deb_replace_${TS}"

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo "$0" "$@"
fi

if [[ ! -f "${DEB_PATH}" ]]; then
  echo "TDengine deb package not found: ${DEB_PATH}" >&2
  exit 2
fi

mkdir -p "${BACKUP_DIR}"

echo "== TDengine OSS deb replacement =="
echo "deb: ${DEB_PATH}"
echo "backup: ${BACKUP_DIR}"
echo
echo "Installer may ask for:"
echo "  - existing cluster firstEp: press Enter"
echo "  - support email: press Enter"
echo

{
  echo "=== versions before ==="
  taos -V || true
  taosd -V || true
  taosadapter -V || true
  echo "=== services before ==="
  systemctl status taosd taosadapter taoskeeper taos-explorer --no-pager -l || true
  echo "=== config before ==="
  grep -E '^(fqdn|dataDir|logDir|firstEp|serverPort|timezone)' /etc/taos/taos.cfg || true
  echo "=== disk before ==="
  du -sh /etc/taos /data/taos /var/lib/taos /usr/local/taos 2>/dev/null || true
} | tee "${BACKUP_DIR}/preflight.txt"

tar -C / -czf "${BACKUP_DIR}/etc_taos.tgz" etc/taos 2>/dev/null || true
tar -C / -czf "${BACKUP_DIR}/data_taos.tgz" data/taos 2>/dev/null || true
tar -C / -czf "${BACKUP_DIR}/usr_local_taos.tgz" usr/local/taos 2>/dev/null || true

systemctl stop taosadapter taoskeeper taos-explorer taosd 2>/dev/null || true
pkill -f '/usr/local/taos/bin/taosadapter' 2>/dev/null || true
pkill -f '/usr/local/taos/bin/taosd' 2>/dev/null || true
sleep 2

dpkg -i "${DEB_PATH}"

mkdir -p /etc/taos /data/taos /var/log/taos

ensure_taos_cfg() {
  local key="$1"
  local value="$2"
  local file="/etc/taos/taos.cfg"
  touch "${file}"
  if grep -Eq "^[#[:space:]]*${key}[[:space:]]+" "${file}"; then
    sed -i -r "s|^[#[:space:]]*(${key}[[:space:]]+).*|${key} ${value}|" "${file}"
  else
    printf '%s %s\n' "${key}" "${value}" >> "${file}"
  fi
}

ensure_taos_cfg "fqdn" "${TDENGINE_FQDN:-$(hostname -s)}"
ensure_taos_cfg "dataDir" "/data/taos"
ensure_taos_cfg "logDir" "/var/log/taos"

if [[ -d /usr/local/taos/cfg ]]; then
  ln -sf /etc/taos/taos.cfg /usr/local/taos/cfg/taos.cfg
  [[ -f /etc/taos/taosadapter.toml ]] && ln -sf /etc/taos/taosadapter.toml /usr/local/taos/cfg/taosadapter.toml || true
  [[ -f /etc/taos/taoskeeper.toml ]] && ln -sf /etc/taos/taoskeeper.toml /usr/local/taos/cfg/taoskeeper.toml || true
  [[ -f /etc/taos/explorer.toml ]] && ln -sf /etc/taos/explorer.toml /usr/local/taos/cfg/explorer.toml || true
fi

systemctl daemon-reload
systemctl reset-failed taosd taosadapter taoskeeper taos-explorer 2>/dev/null || true
systemctl enable taosd >/dev/null 2>&1 || true
systemctl restart taosd
sleep 5

systemctl restart taosadapter 2>/dev/null || true

{
  echo "=== versions after ==="
  taos -V || true
  taosd -V || true
  taosadapter -V || true
  echo "=== services after ==="
  systemctl status taosd taosadapter --no-pager -l || true
  echo "=== config after ==="
  grep -E '^(fqdn|dataDir|logDir|firstEp|serverPort|timezone)' /etc/taos/taos.cfg || true
  echo "=== ports after ==="
  ss -ltnp | grep -E ':(6030|6041|6060)\b' || true
  echo "=== databases ==="
  taos -s 'show databases;' || true
} | tee "${BACKUP_DIR}/postflight.txt"

echo
echo "Done. Backup and logs: ${BACKUP_DIR}"
