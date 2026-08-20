"""Remote WindPy-compatible adapter for Ubuntu."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
import requests

REMOTE_NETWORK_ERROR = -99990001
REMOTE_AUTH_ERROR = -99990002
REMOTE_GATEWAY_ERROR = -99990003
REMOTE_BAD_RESPONSE = -99990004

@dataclass
class WindData:
    ErrorCode: int = 0
    Codes: list[Any] = field(default_factory=list)
    Fields: list[Any] = field(default_factory=list)
    Times: list[Any] = field(default_factory=list)
    Data: list[Any] = field(default_factory=list)


def _error_result(code: int, message: str) -> WindData:
    return WindData(ErrorCode=code, Fields=["OUTMESSAGE"], Data=[[message]])


class RemoteWind:
    def __init__(self) -> None:
        self.base_url = os.getenv("WIND_GATEWAY_URL", "http://127.0.0.1:18765").rstrip("/")
        self.token = os.getenv("WIND_GATEWAY_TOKEN", "").strip()
        self.connect_timeout = float(os.getenv("WIND_CONNECT_TIMEOUT", "5"))
        self.read_timeout = float(os.getenv("WIND_READ_TIMEOUT", "180"))
        self._new_session()

    def _new_session(self) -> None:
        self.session = requests.Session()
        self.session.trust_env = False

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _timeout(self) -> tuple[float, float]:
        return self.connect_timeout, self.read_timeout

    @staticmethod
    def _to_wind_data(payload: dict[str, Any]) -> WindData:
        return WindData(
            ErrorCode=int(payload.get("error_code", REMOTE_BAD_RESPONSE)),
            Codes=payload.get("codes") or [],
            Fields=payload.get("fields") or [],
            Times=payload.get("times") or [],
            Data=payload.get("data") or [],
        )

    def _request_result(self, endpoint: str, body: dict[str, Any]) -> WindData:
        if not self.token:
            return _error_result(REMOTE_AUTH_ERROR, "WIND_GATEWAY_TOKEN is not set.")
        try:
            response = self.session.post(
                f"{self.base_url}{endpoint}",
                headers=self._headers(),
                json=body,
                timeout=self._timeout(),
            )
        except requests.RequestException as exc:
            return _error_result(REMOTE_NETWORK_ERROR, f"Cannot reach gateway: {exc}")

        if response.status_code == 401:
            return _error_result(REMOTE_AUTH_ERROR, "Gateway token authentication failed.")
        try:
            payload = response.json()
        except ValueError:
            return _error_result(REMOTE_BAD_RESPONSE, f"Non-JSON response: HTTP {response.status_code}")
        if "error_code" in payload:
            return self._to_wind_data(payload)
        return _error_result(
            REMOTE_GATEWAY_ERROR,
            payload.get("error", f"Gateway error: HTTP {response.status_code}"),
        )

    def start(self, *args: Any, **kwargs: Any) -> WindData:
        if not self.token:
            return _error_result(REMOTE_AUTH_ERROR, "WIND_GATEWAY_TOKEN is not set.")
        try:
            response = self.session.get(
                f"{self.base_url}/health/ready",
                headers=self._headers(),
                timeout=self._timeout(),
            )
        except requests.RequestException as exc:
            return _error_result(REMOTE_NETWORK_ERROR, f"Cannot reach gateway: {exc}")
        if response.status_code == 401:
            return _error_result(REMOTE_AUTH_ERROR, "Gateway token authentication failed.")
        try:
            payload = response.json()
        except ValueError:
            return _error_result(REMOTE_BAD_RESPONSE, f"Non-JSON response: HTTP {response.status_code}")
        if response.ok and payload.get("wind_connected") is True:
            return WindData(ErrorCode=0)
        return _error_result(REMOTE_GATEWAY_ERROR, payload.get("error", "Remote WindPy is not ready."))

    def isconnected(self) -> bool:
        return self.start().ErrorCode == 0

    def wsd(self, codes, fields, beginTime, endTime, options="", *args, **kwargs) -> WindData:
        return self._request_result(
            "/api/v1/wsd",
            {"codes": codes, "fields": fields, "start": str(beginTime), "end": str(endTime), "options": options or ""},
        )

    def wss(self, codes, fields, options="", *args, **kwargs) -> WindData:
        return self._request_result(
            "/api/v1/wss",
            {"codes": codes, "fields": fields, "options": options or ""},
        )

    def tdays(self, beginTime, endTime, options="", *args, **kwargs) -> WindData:
        return self._request_result(
            "/api/v1/tdays",
            {"start": str(beginTime), "end": str(endTime), "options": options or ""},
        )

    def close(self) -> WindData:
        # Do not terminate the persistent Mac WindPy session.
        self.session.close()
        self._new_session()
        return WindData(ErrorCode=0)


w = RemoteWind()
