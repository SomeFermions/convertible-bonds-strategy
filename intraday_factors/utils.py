from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    if series.dropna().empty:
        return series
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def clip_zscore(series: pd.Series, clip: float = 3.0) -> pd.Series:
    return series.clip(-clip, clip)


def rolling_mad(values: pd.Series) -> float:
    median = values.median()
    return float((values - median).abs().median())


def rolling_mad_array(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    median = np.nanmedian(values)
    return float(np.nanmedian(np.abs(values - median)))


def past_rolling_robust_z(
    series: pd.Series,
    window: int,
    min_periods: int,
    epsilon: float = 1e-9,
    clip: float | None = 3.0,
) -> pd.Series:
    """Past-only robust z-score using t-1 and earlier observations."""
    history = series.shift(1)
    median = history.rolling(window=window, min_periods=min_periods).median()
    mad = history.rolling(window=window, min_periods=min_periods).apply(rolling_mad, raw=False)
    z = (series - median) / (1.4826 * mad + epsilon)
    return z.clip(-clip, clip) if clip is not None else z


def _grouped_past_rolling_robust_z(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    sort_cols: list[str],
    window: int,
    min_periods: int,
    epsilon: float,
    clip: float | None,
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="float64")

    work = df[[value_col, *sort_cols]].sort_values(sort_cols)
    keys = [work[col] for col in group_cols]
    values = work[value_col].astype(float)
    history = values.groupby(keys, sort=False, dropna=False).shift(1)
    grouped_history = history.groupby(keys, sort=False, dropna=False)
    median = grouped_history.rolling(window=window, min_periods=min_periods).median()
    mad = grouped_history.rolling(window=window, min_periods=min_periods).apply(rolling_mad_array, raw=True)
    if group_cols:
        median = median.reset_index(level=list(range(len(group_cols))), drop=True)
        mad = mad.reset_index(level=list(range(len(group_cols))), drop=True)
    z_sorted = (values - median) / (1.4826 * mad + epsilon)
    if clip is not None:
        z_sorted = z_sorted.clip(-clip, clip)
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    out.loc[z_sorted.index] = z_sorted
    return out


def same_slot_robust_z(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    slot_col: str,
    lookback_samples: int,
    min_samples: int,
    fallback_window: int,
    epsilon: float = 1e-9,
    clip: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """Past-only same-time-slot robust z-score with rolling fallback.

    The same-slot history is grouped by group_cols + slot_id and shifted by one
    observation, so the current day never contributes to its own scale.
    """
    same_slot_group_cols = group_cols + [slot_col]
    z = _grouped_past_rolling_robust_z(
        df,
        value_col,
        same_slot_group_cols,
        group_cols + [slot_col, "bar_start"],
        lookback_samples,
        min_samples,
        epsilon,
        clip,
    )
    fallback = pd.Series(False, index=df.index, dtype="bool")

    missing = z.isna() & df[value_col].notna()
    if missing.any():
        fallback_z = _grouped_past_rolling_robust_z(
            df,
            value_col,
            group_cols,
            group_cols + ["bar_start"],
            fallback_window,
            max(3, min_samples // 2),
            epsilon,
            clip,
        )
        use = missing & fallback_z.notna()
        z.loc[use] = fallback_z.loc[use]
        fallback.loc[use] = True
    return z, fallback


def point_in_time_asof_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    by: str,
    left_on: str,
    right_on: str,
    columns: Iterable[str],
) -> pd.DataFrame:
    """As-of join that only allows right rows with timestamp <= left timestamp."""
    if left.empty or right.empty:
        out = left.copy()
        for col in columns:
            if col not in out.columns:
                out[col] = np.nan
        return out

    pieces = []
    right_cols = [by, right_on, *columns]
    for key, left_group in left.sort_values(left_on).groupby(by, dropna=False):
        right_group = right[right[by] == key].sort_values(right_on)
        merged = pd.merge_asof(
            left_group.sort_values(left_on),
            right_group[right_cols],
            left_on=left_on,
            right_on=right_on,
            direction="backward",
        )
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True) if pieces else left.copy()


def rolling_regression(
    y: pd.Series,
    x: pd.DataFrame,
    window: int,
    min_observations: int,
    ridge_alpha: float = 0.0,
) -> pd.DataFrame:
    """Past-only rolling ridge/OLS. Parameters at row t are fit with rows < t.

    The implementation keeps rolling sufficient statistics instead of slicing
    every historical window. That matters for intraday monitoring because this
    function is called per bond for the residual model and empirical delta.
    """
    columns = ["alpha", *x.columns]
    index = y.index
    n = len(index)
    p = len(x.columns)
    out = np.full((n, len(columns) + 2), np.nan, dtype=float)
    if n == 0:
        return pd.DataFrame(out, index=index, columns=columns + ["nobs", "r2"])

    y_values = y.to_numpy(dtype=float, copy=False)
    x_values = x.to_numpy(dtype=float, copy=False)
    if p == 0:
        valid = np.isfinite(y_values)
    else:
        valid = np.isfinite(y_values) & np.isfinite(x_values).all(axis=1)

    count_values = valid.astype(float)
    y_clean = np.where(valid, y_values, 0.0)
    x_clean = np.where(valid[:, None], x_values, 0.0) if p else np.empty((n, 0), dtype=float)

    count_prefix = np.concatenate([[0.0], np.cumsum(count_values)])
    y_prefix = np.concatenate([[0.0], np.cumsum(y_clean)])
    y2_prefix = np.concatenate([[0.0], np.cumsum(y_clean * y_clean)])
    x_prefix = np.vstack([np.zeros((1, p)), np.cumsum(x_clean, axis=0)])
    xy_prefix = np.vstack([np.zeros((1, p)), np.cumsum(x_clean * y_clean[:, None], axis=0)])
    xx_prefix: dict[tuple[int, int], np.ndarray] = {}
    for i in range(p):
        for j in range(i, p):
            xx_prefix[(i, j)] = np.concatenate([[0.0], np.cumsum(x_clean[:, i] * x_clean[:, j])])

    eye = np.eye(p + 1) * ridge_alpha
    eye[0, 0] = 0.0
    min_obs = int(min_observations)
    win = int(window)

    for pos in range(n):
        start = max(0, pos - win)
        nobs = int(count_prefix[pos] - count_prefix[start])
        out[pos, len(columns)] = nobs
        if nobs < min_observations:
            continue
        sum_y = y_prefix[pos] - y_prefix[start]
        sum_y2 = y2_prefix[pos] - y2_prefix[start]
        sum_x = x_prefix[pos] - x_prefix[start]
        sum_xy = xy_prefix[pos] - xy_prefix[start]
        xtx = np.empty((p + 1, p + 1), dtype=float)
        xtx[0, 0] = nobs
        xtx[0, 1:] = sum_x
        xtx[1:, 0] = sum_x
        for i in range(p):
            for j in range(i, p):
                value = xx_prefix[(i, j)][pos] - xx_prefix[(i, j)][start]
                xtx[i + 1, j + 1] = value
                xtx[j + 1, i + 1] = value
        xty = np.concatenate([[sum_y], sum_xy])
        try:
            beta = np.linalg.solve(xtx + eye, xty)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(xtx + eye) @ xty
        out[pos, : len(columns)] = beta
        ss_tot = float(sum_y2 - (sum_y * sum_y / nobs))
        ss_res = float(sum_y2 - 2.0 * beta @ xty + beta @ xtx @ beta)
        out[pos, len(columns) + 1] = np.nan if math.isclose(ss_tot, 0.0) else 1.0 - ss_res / ss_tot
    return pd.DataFrame(out, index=index, columns=columns + ["nobs", "r2"])


def safe_log_return(values: pd.Series) -> pd.Series:
    prev = values.shift(1)
    out = np.log(values / prev)
    out[(values <= 0) | (prev <= 0)] = np.nan
    return out


def rolling_sum(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    return series.rolling(window=window, min_periods=min_periods or window).sum()


def rolling_rv(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    return np.sqrt((series.pow(2)).rolling(window=window, min_periods=min_periods or window).mean())
