from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd


MARKET_TZ = ZoneInfo("Asia/Shanghai")
SESSIONS = ((time(9, 30), time(11, 30)), (time(13, 0), time(15, 0)))
CLOSE_TIME = time(15, 0)


def ensure_market_naive(ts: pd.Timestamp | str | datetime) -> pd.Timestamp:
    value = pd.Timestamp(ts)
    if value.tzinfo is not None:
        value = value.tz_convert(MARKET_TZ).tz_localize(None)
    return value


def bucket_start(ts: pd.Timestamp | str | datetime, bar_minutes: int = 5) -> pd.Timestamp:
    value = ensure_market_naive(ts)
    minute = (value.minute // bar_minutes) * bar_minutes
    return value.replace(minute=minute, second=0, microsecond=0, nanosecond=0)


def completed_bar_cutoff(asof: pd.Timestamp | str | datetime, bar_minutes: int = 5) -> pd.Timestamp:
    """Return the latest bar_start that should be complete at asof."""
    return bucket_start(asof, bar_minutes) - pd.Timedelta(minutes=bar_minutes)


def slot_id(ts: pd.Timestamp | str | datetime, bar_minutes: int = 5) -> int | None:
    """Session-aware 5-minute slot id. Lunch break never creates slots."""
    value = ensure_market_naive(ts)
    current = value.time()
    slots_before = 0
    for session_idx, (start, end) in enumerate(SESSIONS):
        if start <= current < end:
            start_ts = value.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0, nanosecond=0)
            return slots_before + int((value - start_ts).total_seconds() // 60 // bar_minutes)
        if session_idx == len(SESSIONS) - 1 and current == end:
            session_minutes = (datetime.combine(value.date(), end) - datetime.combine(value.date(), start)).seconds // 60
            return slots_before + session_minutes // bar_minutes
        session_minutes = (datetime.combine(value.date(), end) - datetime.combine(value.date(), start)).seconds // 60
        slots_before += session_minutes // bar_minutes
    return None


def is_session_bar(ts: pd.Timestamp | str | datetime, bar_minutes: int = 5) -> bool:
    return slot_id(ts, bar_minutes) is not None


def session_bar_role(ts: pd.Timestamp | str | datetime, bar_minutes: int = 5) -> str | None:
    value = ensure_market_naive(ts)
    current = value.time()
    if slot_id(value, bar_minutes) is None:
        return None
    if current == CLOSE_TIME:
        return "CLOSE"
    if current == SESSIONS[0][0]:
        return "OPEN"
    if current == SESSIONS[1][0]:
        return "AFTERNOON_OPEN"
    if current < SESSIONS[0][1]:
        return "AM_CONTINUOUS"
    return "PM_CONTINUOUS"


def expected_slots_per_day(bar_minutes: int = 5) -> int:
    total = 0
    today = pd.Timestamp("2026-01-01")
    for idx, (start, end) in enumerate(SESSIONS):
        total += int((datetime.combine(today, end) - datetime.combine(today, start)).seconds // 60 // bar_minutes)
        if idx == len(SESSIONS) - 1:
            total += 1
    return total
