from __future__ import annotations

from typing import Any

import pandas as pd


ACTIVE = "ACTIVE"
OBSERVE = "OBSERVE"
SHADOW_RESEARCH = "SHADOW_RESEARCH"
DROPPED = "DROPPED"
INACTIVE = "INACTIVE"
ACTIVE_PENDING_HISTORY = "ACTIVE_PENDING_HISTORY"
ACTIVE_MANUAL = "ACTIVE_MANUAL"
OBSERVE_MANUAL = "OBSERVE_MANUAL"
BLACKLIST_MANUAL = "BLACKLIST_MANUAL"

KNOWN_POOL_STATES = {
    ACTIVE,
    OBSERVE,
    SHADOW_RESEARCH,
    DROPPED,
    INACTIVE,
    ACTIVE_PENDING_HISTORY,
    ACTIVE_MANUAL,
    OBSERVE_MANUAL,
    BLACKLIST_MANUAL,
}

DEFAULT_ACTIONABLE_POOL_STATES = {ACTIVE, ACTIVE_MANUAL}
DEFAULT_COLLECTED_POOL_STATES = {ACTIVE, OBSERVE, SHADOW_RESEARCH, ACTIVE_PENDING_HISTORY, ACTIVE_MANUAL, OBSERVE_MANUAL}
NON_ACTION_POOL_STATES = {OBSERVE, SHADOW_RESEARCH, ACTIVE_PENDING_HISTORY, OBSERVE_MANUAL, BLACKLIST_MANUAL}


def normalize_pool_state(value: Any, default: str = ACTIVE) -> str:
    if value is None or pd.isna(value):
        return default
    state = str(value).upper().strip()
    if state in KNOWN_POOL_STATES:
        return state
    return default


def parse_pool_scope(value: str | None) -> set[str]:
    if not value:
        return set(DEFAULT_ACTIONABLE_POOL_STATES)
    aliases = {
        "active": ACTIVE,
        "observe": OBSERVE,
        "shadow": SHADOW_RESEARCH,
        "shadow_research": SHADOW_RESEARCH,
        "pending": ACTIVE_PENDING_HISTORY,
        "active_pending_history": ACTIVE_PENDING_HISTORY,
        "active_manual": ACTIVE_MANUAL,
        "observe_manual": OBSERVE_MANUAL,
    }
    states = set()
    for item in str(value).split(","):
        token = item.strip().lower()
        if not token:
            continue
        states.add(aliases.get(token, token.upper()))
    return states


def is_actionable_pool_state(value: Any) -> bool:
    return normalize_pool_state(value) in DEFAULT_ACTIONABLE_POOL_STATES
