from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests


JQ_CB_FIELD_WHITELIST = {"open", "high", "low", "close", "volume", "money", "paused"}
JQ_STOCK_FIELD_WHITELIST = {"open", "high", "low", "close", "volume", "money", "paused", "live_turnover"}
JQ_OPTIONAL_FIELD_FALLBACK = {"paused", "live_turnover"}
WIND_FIELD_WHITELIST = {"live_turnover", "ytm"}
DEFAULT_JQDATA_CREDENTIALS_FILE = Path(__file__).resolve().parents[1] / "config" / "jqdata_credentials.toml"
WIND_DEFAULT_FIELD_MAP = {"live_turnover": "turn", "ytm": "ytm_b"}
WIND_FORBIDDEN_TOKENS = {
    "call",
    "redeem",
    "put",
    "reset",
    "downward",
    "clause",
    "term",
    "rating",
    "coupon",
    "trigger",
}


def account_hash(value: str | None) -> str:
    text = value or "unconfigured"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def read_secret(env_name: str, file_env_name: str | None = None) -> str | None:
    value = os.getenv(env_name)
    if value:
        return value.strip()
    if file_env_name:
        path = os.getenv(file_env_name)
        if path:
            return Path(path).expanduser().read_text(encoding="utf-8").strip()
    return None


def load_jqdata_credentials(path: str | Path | None = None) -> tuple[str | None, str | None]:
    credentials_path = Path(
        os.getenv("JQDATA_CREDENTIALS_FILE")
        or path
        or DEFAULT_JQDATA_CREDENTIALS_FILE
    ).expanduser()
    if not credentials_path.exists():
        return None, None
    with credentials_path.open("rb") as fh:
        loaded = tomllib.load(fh)
    section = loaded.get("jqdata", loaded)
    if not isinstance(section, dict):
        raise ValueError(f"JQData credentials file must contain a [jqdata] table: {credentials_path}")
    username = str(section.get("username") or "").strip() or None
    password = str(section.get("password") or "").strip() or None
    return username, password


def jq_code_from_bond(bond_code: str) -> str:
    code = str(bond_code).strip().zfill(6)
    if code.startswith(("110", "111", "113", "118", "132")):
        return f"{code}.XSHG"
    return f"{code}.XSHE"


def jq_code_from_stock(stock_code: str) -> str:
    code = str(stock_code).strip().zfill(6)
    if code.startswith(("60", "68", "90")):
        return f"{code}.XSHG"
    return f"{code}.XSHE"


def jq_intraday_bounds(start_date: str, end_date: str) -> tuple[str, str]:
    start_text = str(start_date).strip()
    end_text = str(end_date).strip()
    if len(start_text) <= 10:
        start_text = f"{pd.Timestamp(start_text).strftime('%Y-%m-%d')} 09:30:00"
    if len(end_text) <= 10:
        end_text = f"{pd.Timestamp(end_text).strftime('%Y-%m-%d')} 15:00:00"
    return start_text, end_text


def wind_code_from_bond(bond_code: str) -> str:
    code = str(bond_code).strip().zfill(6)
    suffix = "SH" if code.startswith(("110", "111", "113", "118", "132")) else "SZ"
    return f"{code}.{suffix}"


def normalize_jq_price_frame(raw: Any, vendor_code: str, fields: list[str]) -> pd.DataFrame:
    if raw is None:
        return pd.DataFrame()
    if isinstance(raw, dict):
        raw = raw.get(vendor_code)
    df = pd.DataFrame(raw).copy()
    if df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    else:
        df = df.reset_index().rename(columns={"index": "bar_start"})
    if "time" in df.columns and "bar_start" not in df.columns:
        df = df.rename(columns={"time": "bar_start"})
    if "datetime" in df.columns and "bar_start" not in df.columns:
        df = df.rename(columns={"datetime": "bar_start"})
    if "paused" not in df.columns and "is_paused" in df.columns:
        df = df.rename(columns={"is_paused": "paused"})
    for field in fields:
        if field not in df.columns:
            df[field] = None
    if "bar_start" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "bar_start"})
    df["bar_start"] = pd.to_datetime(df["bar_start"], errors="coerce")
    return df[df["bar_start"].notna()].copy()


def validate_jq_fields(asset_type: str, fields: list[str]) -> list[str]:
    allowed = JQ_CB_FIELD_WHITELIST if asset_type.upper() == "CB" else JQ_STOCK_FIELD_WHITELIST
    requested = [item.strip() for item in fields if item.strip()]
    invalid = sorted(set(requested) - allowed)
    if invalid:
        raise ValueError(f"JQData {asset_type} fields not allowed: {invalid}")
    return requested


def validate_wind_fields(fields: list[str]) -> list[str]:
    requested = [item.strip() for item in fields if item.strip()]
    invalid = sorted(set(requested) - WIND_FIELD_WHITELIST)
    forbidden = sorted(field for field in requested if any(token in field.lower() for token in WIND_FORBIDDEN_TOKENS))
    if invalid or forbidden:
        raise ValueError(f"Wind fields not allowed invalid={invalid} forbidden={forbidden}")
    return requested


def should_retry_without_optional_jq_fields(exc: Exception, requested: list[str]) -> bool:
    if not any(field in JQ_OPTIONAL_FIELD_FALLBACK for field in requested):
        return False
    message = str(exc).lower()
    field_error_tokens = ("field", "column", "unknown", "invalid", "unsupported", "not support", "不存在")
    return any(token in message for token in field_error_tokens)


@dataclass
class JQDataProvider:
    username_env: str = "JQDATA_USERNAME"
    password_env: str = "JQDATA_PASSWORD"
    username_file_env: str = "JQDATA_USERNAME_FILE"
    password_file_env: str = "JQDATA_PASSWORD_FILE"
    credentials_file: str | Path | None = None
    _get_price: Any | None = None
    _unsupported_optional_fields: set[str] = field(default_factory=set)

    def credentials(self) -> tuple[str | None, str | None]:
        file_username, file_password = load_jqdata_credentials(self.credentials_file)
        username = read_secret(self.username_env, self.username_file_env) or os.getenv("JQDATA_ACCOUNT") or file_username
        password = read_secret(self.password_env, self.password_file_env) or file_password
        return username, password

    def account_id_hash(self) -> str:
        username, _ = self.credentials()
        return account_hash(username)

    def _sdk(self):
        if self._get_price is not None:
            return self._get_price
        try:
            from jqdatasdk import auth, get_price
        except Exception as exc:  # pragma: no cover - dependency may be absent locally
            raise RuntimeError("jqdatasdk is not installed in this Python environment") from exc
        username, password = self.credentials()
        if not username or not password:
            raise RuntimeError(
                "JQData credentials are not configured. Fill config/jqdata_credentials.toml, "
                "set JQDATA_CREDENTIALS_FILE, or use the JQDATA_USERNAME/JQDATA_PASSWORD environment variables."
            )
        auth(username, password)
        self._get_price = get_price
        return get_price

    def get_price_5m(self, vendor_code: str, start_date: str, end_date: str, fields: list[str]) -> pd.DataFrame:
        get_price = self._sdk()
        requested = list(dict.fromkeys(fields))
        effective = [field for field in requested if field not in self._unsupported_optional_fields]
        request_start, request_end = jq_intraday_bounds(start_date, end_date)
        try:
            raw = get_price(
                vendor_code,
                start_date=request_start,
                end_date=request_end,
                frequency="5m",
                fields=effective,
                panel=False,
            )
            return normalize_jq_price_frame(raw, vendor_code, requested)
        except Exception as exc:
            if not should_retry_without_optional_jq_fields(exc, effective):
                raise
            message = str(exc).lower()
            rejected = {
                field
                for field in effective
                if field in JQ_OPTIONAL_FIELD_FALLBACK and field.lower() in message
            }
            if not rejected:
                rejected = set(effective) & JQ_OPTIONAL_FIELD_FALLBACK
            fallback = [field for field in effective if field not in rejected]
            if fallback == effective:
                raise
            self._unsupported_optional_fields.update(rejected)
            raw = get_price(
                vendor_code,
                start_date=request_start,
                end_date=request_end,
                frequency="5m",
                fields=fallback,
                panel=False,
            )
            return normalize_jq_price_frame(raw, vendor_code, requested)

    def get_stock_daily_unadjusted(
        self,
        vendor_codes: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        get_price = self._sdk()
        if not vendor_codes:
            return pd.DataFrame(columns=["trade_date", "vendor_code", "close"])
        raw = get_price(
            list(dict.fromkeys(vendor_codes)),
            start_date=str(start_date),
            end_date=str(end_date),
            frequency="daily",
            fields=["close"],
            skip_paused=False,
            fq=None,
            panel=False,
            fill_paused=False,
        )
        out = pd.DataFrame(raw).reset_index()
        if out.empty:
            return pd.DataFrame(columns=["trade_date", "vendor_code", "close"])
        out = out.rename(columns={"time": "trade_date", "date": "trade_date", "code": "vendor_code"})
        if "trade_date" not in out.columns:
            out = out.rename(columns={out.columns[0]: "trade_date"})
        if "vendor_code" not in out.columns and len(vendor_codes) == 1:
            out["vendor_code"] = vendor_codes[0]
        out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
        out["close"] = pd.to_numeric(out.get("close"), errors="coerce")
        return out.dropna(subset=["trade_date", "vendor_code"]).copy()

    def get_conversion_price_adjustments(self, bond_codes: list[str]) -> pd.DataFrame:
        self._sdk()
        if not bond_codes:
            return pd.DataFrame(
                columns=[
                    "bond_code",
                    "bond_name",
                    "announcement_date",
                    "effective_date",
                    "conversion_price",
                    "adjustment_reason",
                ]
            )
        from jqdatasdk import bond, query

        table = bond.CONBOND_CONVERT_PRICE_ADJUST
        raw = bond.run_query(
            query(
                table.code,
                table.name,
                table.pub_date,
                table.adjust_date,
                table.new_convert_price,
                table.adjust_reason,
            ).filter(table.code.in_([str(code).zfill(6) for code in bond_codes]))
        )
        out = pd.DataFrame(raw).rename(
            columns={
                "code": "bond_code",
                "name": "bond_name",
                "pub_date": "announcement_date",
                "adjust_date": "effective_date",
                "new_convert_price": "conversion_price",
                "adjust_reason": "adjustment_reason",
            }
        )
        if out.empty:
            return out
        out["bond_code"] = out["bond_code"].astype(str).str.zfill(6)
        out["announcement_date"] = pd.to_datetime(out["announcement_date"], errors="coerce").dt.normalize()
        out["effective_date"] = pd.to_datetime(out["effective_date"], errors="coerce").dt.normalize()
        out["conversion_price"] = pd.to_numeric(out["conversion_price"], errors="coerce")
        return out.dropna(subset=["bond_code", "effective_date", "conversion_price"]).copy()


def build_jqdata_provider(config: dict[str, Any]) -> JQDataProvider:
    path = config.get("jqdata", {}).get("credentials_file")
    if path and not Path(path).expanduser().is_absolute():
        path = Path(__file__).resolve().parents[1] / str(path)
    return JQDataProvider(credentials_file=path)


def windpy_available() -> bool:
    return (
        importlib.util.find_spec("WindPy") is not None
        or importlib.util.find_spec("remote_windpy_adapter.WindPy") is not None
    )


def _wind_error_code(response: Any) -> int:
    value = getattr(response, "ErrorCode", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def normalize_wind_wsd_response(
    response: Any,
    bond_code: str,
    requested_fields: list[str],
    field_map: dict[str, str],
) -> pd.DataFrame:
    error_code = _wind_error_code(response)
    if error_code != 0:
        raise RuntimeError(f"Wind wsd failed error_code={error_code}")
    times = list(getattr(response, "Times", []) or [])
    vendor_fields = [str(field).lower() for field in (getattr(response, "Fields", []) or [])]
    data = list(getattr(response, "Data", []) or [])
    if not times:
        return pd.DataFrame(columns=["bond_code", "trade_date", *requested_fields])
    if len(data) != len(vendor_fields):
        raise RuntimeError("Wind wsd returned an unsupported response shape")
    by_vendor_field = {field: list(values or []) for field, values in zip(vendor_fields, data)}
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(times):
        row: dict[str, Any] = {
            "bond_code": str(bond_code).zfill(6),
            "trade_date": pd.Timestamp(value).strftime("%Y-%m-%d"),
        }
        for requested in requested_fields:
            values = by_vendor_field.get(str(field_map[requested]).lower(), [])
            row[requested] = values[index] if index < len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class WindPyProvider:
    field_map: dict[str, str] | None = None
    wsd_options: str = "Days=Trading"
    close_after_request: bool = True
    _wind: Any | None = None

    def account_id_hash(self) -> str:
        return account_hash(os.getenv("WIND_ACCOUNT") or "windpy_local")

    def _client(self):
        if self._wind is not None:
            return self._wind
        try:
            from WindPy import w
        except ImportError:
            try:
                from remote_windpy_adapter.WindPy import w
            except Exception as exc:  # pragma: no cover - depends on local adapter setup
                raise RuntimeError("WindPy or remote_windpy_adapter is not available in this Python environment") from exc
        self._wind = w
        return w

    def fetch_turnover_ytm(self, bond_codes: list[str], start_date: str, end_date: str, fields: list[str]) -> pd.DataFrame:
        fields = validate_wind_fields(fields)
        field_map = dict(WIND_DEFAULT_FIELD_MAP)
        field_map.update(self.field_map or {})
        missing_map = sorted(field for field in fields if not field_map.get(field))
        if missing_map:
            raise ValueError(f"Wind field mapping missing for {missing_map}")

        wind = self._client()
        connected = bool(wind.isconnected())
        started_here = False
        if not connected:
            start_result = wind.start()
            error_code = _wind_error_code(start_result)
            if error_code != 0 or not bool(wind.isconnected()):
                raise RuntimeError(f"Wind start failed error_code={error_code}")
            started_here = True

        frames: list[pd.DataFrame] = []
        try:
            vendor_fields = ",".join(field_map[field] for field in fields)
            for bond_code in bond_codes:
                response = wind.wsd(
                    wind_code_from_bond(bond_code),
                    vendor_fields,
                    start_date,
                    end_date,
                    self.wsd_options,
                )
                frames.append(normalize_wind_wsd_response(response, bond_code, fields, field_map))
        finally:
            if started_here and self.close_after_request:
                wind.close()
        if not frames:
            return pd.DataFrame(columns=["bond_code", "trade_date", *fields])
        return pd.concat(frames, ignore_index=True)


@dataclass
class WindHttpProvider:
    """Quota-safe transport placeholder for a Mac-side Wind bridge.

    The bridge is expected to expose a narrow endpoint that accepts only
    bond_code/date/field requests for ytm and live_turnover. This class does not
    depend on WindPy and refuses to run when WIND_PROVIDER_URL is not set.
    """

    url_env: str = "WIND_PROVIDER_URL"
    token_env: str = "WIND_PROVIDER_TOKEN"

    def account_id_hash(self) -> str:
        return account_hash(os.getenv(self.url_env))

    def fetch_turnover_ytm(self, bond_codes: list[str], start_date: str, end_date: str, fields: list[str]) -> pd.DataFrame:
        fields = validate_wind_fields(fields)
        url = os.getenv(self.url_env)
        if not url:
            raise RuntimeError("WIND_PROVIDER_URL is not configured; Wind enrichment skipped without consuming quota")
        payload = {
            "bond_codes": [str(code).zfill(6) for code in bond_codes],
            "start_date": start_date,
            "end_date": end_date,
            "fields": fields,
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        token = os.getenv(self.token_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = requests.post(url.rstrip("/") + "/wind/turnover-ytm", data=json.dumps(payload), headers=headers, timeout=60)
        response.raise_for_status()
        return pd.DataFrame(response.json().get("rows", []))


def build_wind_provider(config: dict[str, Any]) -> WindPyProvider | WindHttpProvider:
    wind_config = config.get("wind", {})
    provider_name = str(os.getenv("WIND_PROVIDER") or wind_config.get("provider", "auto")).strip().lower()
    if provider_name not in {"auto", "windpy", "http"}:
        raise ValueError(f"Unknown Wind provider: {provider_name}")
    if provider_name == "windpy" or (provider_name == "auto" and windpy_available()):
        return WindPyProvider(
            field_map=dict(wind_config.get("field_map", {})),
            wsd_options=str(wind_config.get("wsd_options", "Days=Trading")),
            close_after_request=bool(wind_config.get("close_after_request", True)),
        )
    if provider_name == "http" or os.getenv("WIND_PROVIDER_URL"):
        return WindHttpProvider()
    raise RuntimeError(
        "No Wind provider is available. Install/import WindPy in this Python environment "
        "or configure WIND_PROVIDER_URL for the HTTP bridge."
    )
