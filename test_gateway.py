#!/usr/bin/env python3

import json
import os
import sys
import traceback

import requests


print("test_gateway.py started", flush=True)

BASE_URL = os.environ.get(
    "WIND_GATEWAY_URL",
    "http://127.0.0.1:18765",
).rstrip("/")

TOKEN = os.environ.get("WIND_GATEWAY_TOKEN", "").strip()


def main():
    print(f"BASE_URL={BASE_URL}", flush=True)
    print(f"TOKEN_SET={bool(TOKEN)}", flush=True)

    if not TOKEN:
        print("ERROR: WIND_GATEWAY_TOKEN is not set", file=sys.stderr, flush=True)
        return 2

    # 禁止 requests 读取 HTTP_PROXY / HTTPS_PROXY / ~/.环境代理
    session = requests.Session()
    session.trust_env = False

    headers = {
        "Authorization": f"Bearer {TOKEN}",
    }

    print("Checking /health/live ...", flush=True)

    live = session.get(
        f"{BASE_URL}/health/live",
        timeout=(5, 10),
    )

    print(f"live status={live.status_code}", flush=True)
    print(live.text, flush=True)
    live.raise_for_status()

    print("Checking /health/ready ...", flush=True)

    ready = session.get(
        f"{BASE_URL}/health/ready",
        headers=headers,
        timeout=(5, 30),
    )

    print(f"ready status={ready.status_code}", flush=True)
    print(json.dumps(ready.json(), ensure_ascii=False, indent=2), flush=True)
    ready.raise_for_status()

    print("Requesting WSD close,turn ...", flush=True)

    response = session.post(
        f"{BASE_URL}/api/v1/wsd",
        headers={
            **headers,
            "Content-Type": "application/json",
        },
        json={
            "codes": ["127061.SZ"],
            "fields": ["close", "turn"],
            "start": "2026-06-01",
            "end": "2026-06-03",
            "options": "",
        },
        timeout=(5, 120),
    )

    print(f"wsd status={response.status_code}", flush=True)
    print(response.text, flush=True)
    response.raise_for_status()

    print("Gateway test completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
