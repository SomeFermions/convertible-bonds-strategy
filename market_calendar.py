import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd


LOGGER = logging.getLogger(__name__)
MARKET_TZ = ZoneInfo("Asia/Shanghai")


def market_today() -> date:
    return datetime.now(MARKET_TZ).date()


def is_cn_trade_day(day: date | str | None = None) -> bool:
    target = market_today() if day is None else pd.to_datetime(day).date()
    try:
        calendar = ak.tool_trade_date_hist_sina()
        if calendar is None or calendar.empty or "trade_date" not in calendar.columns:
            raise RuntimeError("empty trade calendar")

        trade_dates = set(pd.to_datetime(calendar["trade_date"], errors="coerce").dropna().dt.date)
        if target in trade_dates:
            return True
        if trade_dates and min(trade_dates) <= target <= max(trade_dates):
            return False

        LOGGER.warning("Trade calendar does not cover %s; falling back to weekday check", target)
    except Exception as exc:
        LOGGER.warning("Failed to load CN trade calendar: %s; falling back to weekday check", exc)

    return target.weekday() < 5
