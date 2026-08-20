from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.linear_model import HuberRegressor


EPS = 1e-12


@dataclass(frozen=True)
class RegressionResult:
    intercept: float
    coefficients: np.ndarray
    r2: float
    sample_count: int
    source: str


def bar_returns(
    frame: pd.DataFrame,
    close_column: str,
    open_column: str,
) -> pd.Series:
    close = pd.to_numeric(frame[close_column], errors="coerce")
    open_price = pd.to_numeric(frame[open_column], errors="coerce")
    previous = frame.groupby(["bond_code", "trade_date"], sort=False)[close_column].shift(1)
    first = frame["bar_slot"].eq(0)
    result = np.where(first, close / open_price - 1.0, close / previous - 1.0)
    valid = close.gt(0) & np.where(first, open_price.gt(0), previous.gt(0))
    return pd.Series(np.where(valid, result, np.nan), index=frame.index, dtype=float)


def _past_mean_std(values: np.ndarray, window: int, minimum: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    clean = np.where(valid, values, 0.0)
    count = np.concatenate([[0], np.cumsum(valid.astype(np.int64))])
    total = np.concatenate([[0.0], np.cumsum(clean)])
    square = np.concatenate([[0.0], np.cumsum(clean * clean)])
    index = np.arange(len(values))
    start = np.maximum(0, index - window)
    n = count[index] - count[start]
    sums = total[index] - total[start]
    sums2 = square[index] - square[start]
    mean = np.divide(sums, n, out=np.full(len(values), np.nan), where=n > 0)
    variance = np.divide(sums2, n, out=np.full(len(values), np.nan), where=n > 0) - mean**2
    std = np.sqrt(np.maximum(variance, 0.0))
    mean[n < minimum] = np.nan
    std[(n < minimum) | (std < EPS)] = np.nan
    return mean, std


def past_only_same_slot_z(
    frame: pd.DataFrame,
    value_column: str,
    lookback_days: int,
    minimum_days: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, positions in frame.groupby("bar_slot", sort=False).groups.items():
        idx = np.asarray(list(positions), dtype=int)
        values = pd.to_numeric(frame.loc[idx, value_column], errors="coerce").to_numpy(float)
        mean, std = _past_mean_std(values, lookback_days, minimum_days)
        result.loc[idx] = (values - mean) / std
    return result


def past_only_same_slot_ratio(
    frame: pd.DataFrame,
    value_column: str,
    lookback_days: int,
    minimum_days: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, positions in frame.groupby("bar_slot", sort=False).groups.items():
        idx = np.asarray(list(positions), dtype=int)
        values = pd.to_numeric(frame.loc[idx, value_column], errors="coerce").to_numpy(float)
        medians = np.full(len(values), np.nan)
        for i in range(len(values)):
            history = values[max(0, i - lookback_days) : i]
            history = history[np.isfinite(history) & (history > 0)]
            if len(history) >= minimum_days:
                medians[i] = float(np.median(history))
        result.loc[idx] = values / medians
    return result


def fit_regression(
    x: np.ndarray,
    y: np.ndarray,
    model: str = "ridge",
    ridge_alpha: float = 1.0,
    minimum_samples: int = 30,
) -> RegressionResult | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x = x[valid]
    y = y[valid]
    if len(y) < minimum_samples:
        return None
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0, ddof=0)
    usable = x_std > EPS
    if not usable.any():
        return None
    x_std = np.where(usable, x_std, 1.0)
    xs = (x - x_mean) / x_std
    y_mean = float(y.mean())
    yc = y - y_mean
    model_name = str(model).lower()
    if model_name == "huber":
        estimator = HuberRegressor(alpha=max(float(ridge_alpha), 0.0), fit_intercept=True)
        try:
            estimator.fit(xs, y)
        except (ValueError, np.linalg.LinAlgError):
            return None
        standardized_coef = estimator.coef_
        coefficients = standardized_coef / x_std
        intercept = float(estimator.intercept_ - np.dot(coefficients, x_mean))
        prediction = intercept + x @ coefficients
        source = "rolling_huber_prior_days"
    else:
        gram = xs.T @ xs
        alpha = float(ridge_alpha) if model_name == "ridge" else 0.0
        penalty = np.eye(gram.shape[0]) * alpha
        try:
            standardized_coef = np.linalg.solve(gram + penalty, xs.T @ yc)
        except np.linalg.LinAlgError:
            standardized_coef = np.linalg.pinv(gram + penalty) @ (xs.T @ yc)
        coefficients = standardized_coef / x_std
        coefficients = np.where(usable, coefficients, 0.0)
        intercept = float(y_mean - np.dot(coefficients, x_mean))
        prediction = intercept + x @ coefficients
        source = "rolling_ridge_prior_days" if model_name == "ridge" else "rolling_ols_prior_days"
    residual = y - prediction
    denominator = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > EPS else np.nan
    return RegressionResult(intercept, coefficients, r2, int(len(y)), source)


def choose_market_index_for_history(
    history: pd.DataFrame,
    benchmark_codes: Iterable[str],
    minimum_samples: int,
) -> tuple[str | None, RegressionResult | None]:
    y = pd.to_numeric(history["stock_return"], errors="coerce").to_numpy(float)
    best_code = None
    best_result = None
    best_score = -np.inf
    for code in benchmark_codes:
        column = f"index_{code}_return"
        if column not in history.columns:
            continue
        x = pd.to_numeric(history[column], errors="coerce").to_numpy(float)
        result = fit_regression(x[:, None], y, model="ols", minimum_samples=minimum_samples)
        if result is None:
            continue
        score = result.r2 if np.isfinite(result.r2) else -np.inf
        if score > best_score:
            best_code = str(code)
            best_result = result
            best_score = score
    return best_code, best_result


def _date_slices(frame: pd.DataFrame) -> tuple[list[pd.Timestamp], dict[pd.Timestamp, tuple[int, int]]]:
    dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize().to_numpy()
    unique_dates = list(pd.unique(dates))
    slices: dict[pd.Timestamp, tuple[int, int]] = {}
    for date in unique_dates:
        positions = np.flatnonzero(dates == date)
        if len(positions):
            slices[pd.Timestamp(date)] = (int(positions[0]), int(positions[-1]) + 1)
    return [pd.Timestamp(date) for date in unique_dates], slices


def add_market_adjustment_for_bond(
    group: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    out = group.sort_values(["trade_date", "bar_start"]).copy().reset_index(drop=True)
    dates, slices = _date_slices(out)
    window_days = int(config["linkage"]["default_window_days"])
    minimum = int(config["linkage"]["minimum_model_samples"])
    codes = list(config["linkage"]["benchmark_codes"])
    out["chosen_market_index"] = None
    out["stock_index_beta"] = np.nan
    out["stock_index_alpha"] = np.nan
    out["stock_index_model_r2"] = np.nan
    out["stock_index_model_samples"] = 0
    out["stock_market_component"] = np.nan
    out["stock_idiosyncratic_return"] = np.nan
    for day_index, date in enumerate(dates):
        start, end = slices[date]
        first_history_day = max(0, day_index - window_days)
        if day_index == 0:
            continue
        history_start = slices[dates[first_history_day]][0]
        history_end = start
        code, result = choose_market_index_for_history(
            out.iloc[history_start:history_end], codes, minimum
        )
        if code is None or result is None:
            continue
        market_return = pd.to_numeric(
            out.loc[start : end - 1, f"index_{code}_return"], errors="coerce"
        )
        component = result.intercept + result.coefficients[0] * market_return
        out.loc[start : end - 1, "chosen_market_index"] = code
        out.loc[start : end - 1, "stock_index_beta"] = result.coefficients[0]
        out.loc[start : end - 1, "stock_index_alpha"] = result.intercept
        out.loc[start : end - 1, "stock_index_model_r2"] = result.r2
        out.loc[start : end - 1, "stock_index_model_samples"] = result.sample_count
        out.loc[start : end - 1, "stock_market_component"] = component.to_numpy()
        out.loc[start : end - 1, "stock_idiosyncratic_return"] = (
            pd.to_numeric(out.loc[start : end - 1, "stock_return"], errors="coerce").to_numpy()
            - component.to_numpy()
        )
    lookback = int(config["stock_shock"]["same_slot_lookback_days"])
    min_days = int(config["stock_shock"]["same_slot_min_days"])
    out["stock_shock_z"] = past_only_same_slot_z(
        out, "stock_idiosyncratic_return", lookback, min_days
    )
    out["stock_return_z_unadjusted"] = past_only_same_slot_z(
        out, "stock_return", lookback, min_days
    )
    out["stock_shock_amount_z"] = past_only_same_slot_z(
        out, "stock_log_amount", lookback, min_days
    )
    out["cb_amount_z_5m"] = past_only_same_slot_z(out, "cb_log_amount", lookback, min_days)
    out["relative_amount"] = past_only_same_slot_ratio(
        out, "cb_amount", lookback, min_days
    )
    out["high_low_range_proxy_z"] = past_only_same_slot_z(
        out, "high_low_range_proxy", lookback, min_days
    )
    out["premium_expansion_proxy_z"] = past_only_same_slot_z(
        out, "premium_expansion_proxy", lookback, min_days
    )
    return out


def _lag_within_day(frame: pd.DataFrame, column: str, lag: int) -> np.ndarray:
    return (
        frame.groupby("trade_date", sort=False)[column]
        .shift(lag)
        .to_numpy(dtype=float)
    )


def _fit_daily_rolling_model(
    frame: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    window_days: int,
    minimum_samples: int,
    model: str,
    ridge_alpha: float,
) -> dict[str, np.ndarray]:
    dates, slices = _date_slices(frame)
    n_rows = len(frame)
    n_features = x.shape[1]
    coefficients = np.full((n_rows, n_features), np.nan)
    intercept = np.full(n_rows, np.nan)
    r2 = np.full(n_rows, np.nan)
    samples = np.zeros(n_rows, dtype=np.int32)
    for day_index, date in enumerate(dates):
        if day_index == 0:
            continue
        start, end = slices[date]
        history_start = slices[dates[max(0, day_index - window_days)]][0]
        result = fit_regression(
            x[history_start:start],
            y[history_start:start],
            model=model,
            ridge_alpha=ridge_alpha,
            minimum_samples=minimum_samples,
        )
        if result is None:
            continue
        coefficients[start:end, :] = result.coefficients
        intercept[start:end] = result.intercept
        r2[start:end] = result.r2
        samples[start:end] = result.sample_count
    return {
        "coefficients": coefficients,
        "intercept": intercept,
        "r2": r2,
        "samples": samples,
    }


def response_half_life(beta_lag1: float, beta_lag2: float, beta_lag3: float) -> float:
    values = np.abs(np.asarray([beta_lag1, beta_lag2, beta_lag3], dtype=float))
    if not np.isfinite(values).all() or values.sum() <= EPS:
        return np.nan
    cumulative = np.cumsum(values) / values.sum()
    return float(np.searchsorted(cumulative, 0.5) + 1)


def add_distributed_lag_for_bond(
    group: pd.DataFrame,
    config: dict[str, Any],
    model: str | None = None,
) -> pd.DataFrame:
    out = group.sort_values(["trade_date", "bar_start"]).copy().reset_index(drop=True)
    stock_idio = pd.to_numeric(out["stock_idiosyncratic_return"], errors="coerce").to_numpy(float)
    stock_lags = [stock_idio]
    for lag in range(1, 4):
        stock_lags.append(_lag_within_day(out, "stock_idiosyncratic_return", lag))
    cb_market = pd.to_numeric(out["cb_market_return"], errors="coerce").to_numpy(float)
    x = np.column_stack([*stock_lags, cb_market])
    y = pd.to_numeric(out["cb_return"], errors="coerce").to_numpy(float)
    minimum = int(config["linkage"]["minimum_model_samples"])
    selected_model = str(model or config["linkage"]["default_model"])
    ridge_alpha = float(config["linkage"]["ridge_alpha"])
    default_window = int(config["linkage"]["default_window_days"])
    default_result = None
    for window in config["linkage"]["rolling_windows_days"]:
        result = _fit_daily_rolling_model(
            out,
            x,
            y,
            int(window),
            minimum,
            selected_model,
            ridge_alpha,
        )
        suffix = f"_{int(window)}d"
        out[f"stock_to_cb_beta_lag0{suffix}"] = result["coefficients"][:, 0]
        out[f"stock_to_cb_beta_lag1{suffix}"] = result["coefficients"][:, 1]
        out[f"stock_to_cb_beta_lag2{suffix}"] = result["coefficients"][:, 2]
        out[f"stock_to_cb_beta_lag3{suffix}"] = result["coefficients"][:, 3]
        out[f"cb_market_gamma{suffix}"] = result["coefficients"][:, 4]
        out[f"linkage_regression_intercept{suffix}"] = result["intercept"]
        out[f"linkage_regression_r2{suffix}"] = result["r2"]
        out[f"linkage_sample_count{suffix}"] = result["samples"]
        if int(window) == default_window:
            default_result = result
    if default_result is None:
        raise ValueError("default linkage window is missing from rolling_windows_days")
    aliases = {
        "stock_to_cb_beta_lag0": 0,
        "stock_to_cb_beta_lag1": 1,
        "stock_to_cb_beta_lag2": 2,
        "stock_to_cb_beta_lag3": 3,
        "cb_market_gamma": 4,
    }
    for column, position in aliases.items():
        out[column] = default_result["coefficients"][:, position]
    out["linkage_regression_intercept"] = default_result["intercept"]
    out["linkage_regression_r2"] = default_result["r2"]
    out["linkage_sample_count"] = default_result["samples"]
    out["linkage_model_source"] = np.where(
        out["linkage_sample_count"].ge(minimum),
        f"rolling_{selected_model}_{default_window}d_prior_days",
        "unavailable_insufficient_history",
    )
    out["stock_to_cb_beta_actionable_sum"] = out[
        ["stock_to_cb_beta_lag1", "stock_to_cb_beta_lag2", "stock_to_cb_beta_lag3"]
    ].sum(axis=1, min_count=3)
    gamma = pd.to_numeric(out["cb_market_gamma"], errors="coerce")
    out["cb_idiosyncratic_return"] = out["cb_return"] - gamma * out["cb_market_return"]
    reverse_x = np.column_stack(
        [
            _lag_within_day(out, "cb_idiosyncratic_return", lag)
            for lag in range(1, 4)
        ]
    )
    reverse_y = stock_idio
    reverse = _fit_daily_rolling_model(
        out,
        reverse_x,
        reverse_y,
        default_window,
        minimum,
        selected_model,
        ridge_alpha,
    )
    for position, lag in enumerate(range(1, 4)):
        out[f"cb_to_stock_beta_lag{lag}"] = reverse["coefficients"][:, position]
    out["cb_to_stock_beta_actionable_sum"] = out[
        ["cb_to_stock_beta_lag1", "cb_to_stock_beta_lag2", "cb_to_stock_beta_lag3"]
    ].sum(axis=1, min_count=3)
    forward_strength = out["stock_to_cb_beta_actionable_sum"].abs()
    reverse_strength = out["cb_to_stock_beta_actionable_sum"].abs()
    out["dominant_lead_direction"] = np.select(
        [
            forward_strength > reverse_strength * 1.20,
            reverse_strength > forward_strength * 1.20,
            forward_strength.notna() & reverse_strength.notna(),
        ],
        ["STOCK_LEADS_CB", "CB_LEADS_STOCK", "BIDIRECTIONAL_OR_WEAK"],
        default="UNAVAILABLE",
    )
    lag_strength = np.abs(
        out[
            ["stock_to_cb_beta_lag1", "stock_to_cb_beta_lag2", "stock_to_cb_beta_lag3"]
        ].to_numpy(dtype=float)
    )
    lag_total = np.nansum(lag_strength, axis=1)
    cumulative = np.cumsum(np.nan_to_num(lag_strength, nan=0.0), axis=1)
    out["response_half_life_bars"] = np.where(
        np.isfinite(lag_strength).all(axis=1) & (lag_total > EPS),
        np.argmax(cumulative >= (lag_total[:, None] * 0.5), axis=1) + 1,
        np.nan,
    )
    full_samples = float(config["linkage"]["model_confidence_full_samples"])
    sample_component = (out["linkage_sample_count"] / full_samples).clip(0, 1)
    r2_component = pd.to_numeric(out["linkage_regression_r2"], errors="coerce").clip(0, 1).fillna(0)
    market_component = pd.to_numeric(out["stock_index_model_r2"], errors="coerce").clip(0, 1).fillna(0)
    out["linkage_model_confidence"] = (
        0.15 + 0.45 * sample_component + 0.25 * r2_component + 0.15 * market_component
    ).where(out["linkage_sample_count"].ge(minimum), 0.0)
    out["lead_lag_confidence"] = out["linkage_model_confidence"]
    raw_stock = pd.to_numeric(out["stock_return"], errors="coerce").to_numpy(float)
    raw_lags = [raw_stock]
    for lag in range(1, 4):
        raw_lags.append(_lag_within_day(out, "stock_return", lag))
    raw_x = np.column_stack([*raw_lags, cb_market])
    raw_result = _fit_daily_rolling_model(
        out,
        raw_x,
        y,
        default_window,
        minimum,
        selected_model,
        ridge_alpha,
    )
    for position, lag in enumerate(range(4)):
        out[f"unadjusted_stock_to_cb_beta_lag{lag}"] = raw_result["coefficients"][:, position]
    out["unadjusted_cb_market_gamma"] = raw_result["coefficients"][:, 4]
    out["unadjusted_linkage_r2"] = raw_result["r2"]
    out["unadjusted_linkage_sample_count"] = raw_result["samples"]
    return out


def _parallel_jobs(config: dict[str, Any]) -> int:
    configured = int(config["strategy"].get("n_jobs", -1))
    if configured == -1:
        return max(1, os.cpu_count() or 1)
    return max(1, configured)


def prepare_intraday_base_features(
    bars: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    out = bars.sort_values(["bond_code", "trade_date", "bar_start"]).copy().reset_index(drop=True)
    out["cb_return"] = bar_returns(out, "cb_close", "cb_open")
    out["stock_return"] = bar_returns(out, "stock_close", "stock_open")
    out["cb_market_return"] = out.groupby("bar_start", sort=False)["cb_return"].transform("median")
    out["cb_log_amount"] = np.log1p(pd.to_numeric(out["cb_amount"], errors="coerce").clip(lower=0))
    out["stock_log_amount"] = np.log1p(
        pd.to_numeric(out["stock_amount"], errors="coerce").clip(lower=0)
    )
    close = pd.to_numeric(out["cb_close"], errors="coerce")
    high = pd.to_numeric(out["cb_high"], errors="coerce")
    low = pd.to_numeric(out["cb_low"], errors="coerce")
    out["high_low_range_proxy"] = (high - low) / close.replace(0, np.nan)
    out["close_location"] = (close - low) / (high - low).replace(0, np.nan)
    by_day = out.groupby(["bond_code", "trade_date"], sort=False)
    out["cb_cum_return"] = close / by_day["cb_open"].transform("first") - 1.0
    out["stock_cum_return"] = (
        pd.to_numeric(out["stock_close"], errors="coerce")
        / by_day["stock_open"].transform("first")
        - 1.0
    )
    volume = pd.to_numeric(out["cb_volume"], errors="coerce").clip(lower=0)
    typical = (high + low + close) / 3.0
    cumulative_value = (typical * volume).groupby([out["bond_code"], out["trade_date"]]).cumsum()
    cumulative_volume = volume.groupby([out["bond_code"], out["trade_date"]]).cumsum()
    out["cb_intraday_vwap"] = cumulative_value / cumulative_volume.replace(0, np.nan)
    out["cb_vwap_dev"] = close / out["cb_intraday_vwap"] - 1.0
    stock_close = pd.to_numeric(out["stock_close"], errors="coerce")
    stock_high = pd.to_numeric(out["stock_high"], errors="coerce")
    stock_low = pd.to_numeric(out["stock_low"], errors="coerce")
    stock_volume = pd.to_numeric(out["stock_volume"], errors="coerce").clip(lower=0)
    stock_typical = (stock_high + stock_low + stock_close) / 3.0
    stock_value = (stock_typical * stock_volume).groupby(
        [out["bond_code"], out["trade_date"]]
    ).cumsum()
    stock_cum_volume = stock_volume.groupby([out["bond_code"], out["trade_date"]]).cumsum()
    out["stock_intraday_vwap"] = stock_value / stock_cum_volume.replace(0, np.nan)
    out["stock_vwap_dev"] = stock_close / out["stock_intraday_vwap"] - 1.0
    out["vwap_gap"] = out["stock_vwap_dev"] - out["cb_vwap_dev"]
    delta = pd.to_numeric(out.get("empirical_delta"), errors="coerce").clip(0, 1).fillna(0.35)
    out["premium_expansion_proxy"] = out["cb_cum_return"] - delta * out["stock_cum_return"]
    groups = [group for _, group in out.groupby("bond_code", sort=False)]
    jobs = _parallel_jobs(config)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        adjusted = Parallel(n_jobs=jobs, batch_size=1)(
            delayed(add_market_adjustment_for_bond)(group, config) for group in groups
        )
    return pd.concat(adjusted, ignore_index=True).sort_values(
        ["bond_code", "trade_date", "bar_start"]
    ).reset_index(drop=True)


def add_linkage_features(
    bars: pd.DataFrame,
    config: dict[str, Any],
    model: str | None = None,
) -> pd.DataFrame:
    base = prepare_intraday_base_features(bars, config)
    groups = [group for _, group in base.groupby("bond_code", sort=False)]
    jobs = _parallel_jobs(config)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        linked = Parallel(n_jobs=jobs, batch_size=1)(
            delayed(add_distributed_lag_for_bond)(group, config, model) for group in groups
        )
    out = pd.concat(linked, ignore_index=True).sort_values(
        ["bond_code", "trade_date", "bar_start"]
    ).reset_index(drop=True)
    shock = pd.to_numeric(out["stock_idiosyncratic_return"], errors="coerce")
    lag0_expected = pd.to_numeric(out["stock_to_cb_beta_lag0"], errors="coerce") * shock
    out["realized_cb_response_since_shock"] = out["cb_idiosyncratic_return"]
    realized_beyond_lag0 = out["realized_cb_response_since_shock"] - lag0_expected
    for horizon in range(1, 4):
        beta_columns = [f"stock_to_cb_beta_lag{lag}" for lag in range(1, horizon + 1)]
        actionable = out[beta_columns].sum(axis=1, min_count=horizon) * shock
        out[f"expected_cb_response_{horizon}bar"] = actionable
        out[f"linkage_gap_{horizon}bar"] = actionable - realized_beyond_lag0
    out["linkage_repair_speed"] = realized_beyond_lag0
    out["linkage_repair_started"] = (
        out["linkage_repair_speed"]
        > float(config["linkage_gap"]["repair_positive_cb_idio_min_bps"]) / 10000.0
    )
    out["linkage_residual"] = out["cb_return"] - (
        out["linkage_regression_intercept"]
        + out["stock_to_cb_beta_lag0"] * out["stock_idiosyncratic_return"]
        + out["cb_market_gamma"] * out["cb_market_return"]
    )
    raw_shock = pd.to_numeric(out["stock_return"], errors="coerce")
    raw_lag0 = out["unadjusted_stock_to_cb_beta_lag0"] * raw_shock
    raw_realized = out["cb_return"] - out["unadjusted_cb_market_gamma"] * out["cb_market_return"]
    raw_actionable = out[
        [
            "unadjusted_stock_to_cb_beta_lag1",
            "unadjusted_stock_to_cb_beta_lag2",
            "unadjusted_stock_to_cb_beta_lag3",
        ]
    ].sum(axis=1, min_count=3) * raw_shock
    out["unadjusted_linkage_gap_3bar"] = raw_actionable - (raw_realized - raw_lag0)
    out["cb_shock_z"] = np.nan
    out["linkage_gap_z"] = np.nan
    out["linkage_residual_z"] = np.nan
    lookback = int(config["stock_shock"]["same_slot_lookback_days"])
    minimum = int(config["stock_shock"]["same_slot_min_days"])
    for _, positions in out.groupby("bond_code", sort=False).groups.items():
        idx = np.asarray(list(positions), dtype=int)
        group = out.loc[idx].copy().reset_index(drop=True)
        out.loc[idx, "cb_shock_z"] = past_only_same_slot_z(
            group, "cb_idiosyncratic_return", lookback, minimum
        ).to_numpy()
        out.loc[idx, "linkage_gap_z"] = past_only_same_slot_z(
            group, "linkage_gap_3bar", lookback, minimum
        ).to_numpy()
        out.loc[idx, "linkage_residual_z"] = past_only_same_slot_z(
            group, "linkage_residual", lookback, minimum
        ).to_numpy()
    return add_liquidity_and_events(out, config)


def add_liquidity_and_events(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    out = frame.copy()
    liquidity = config["liquidity"]
    amount_z = pd.to_numeric(out["cb_amount_z_5m"], errors="coerce")
    range_z = pd.to_numeric(out["high_low_range_proxy_z"], errors="coerce")
    premium_z = pd.to_numeric(out["premium_expansion_proxy_z"], errors="coerce")
    close_location = pd.to_numeric(out["close_location"], errors="coerce").fillna(0.5)
    out["volume_climax_score"] = (
        0.45 * ((amount_z - float(liquidity["hot_amount_z"])) / 2.0).clip(0, 1).fillna(0)
        + 0.35 * (range_z / 3.0).clip(0, 1).fillna(0)
        + 0.20 * (1.0 - close_location).clip(0, 1)
    ).clip(0, 1)
    out["hot_money_risk_score_5m"] = (
        0.35 * (amount_z / 3.0).clip(0, 1).fillna(0)
        + 0.30 * (premium_z / 2.0).clip(0, 1).fillna(0)
        + 0.20 * (pd.to_numeric(out["cb_vwap_dev"], errors="coerce") / 0.02).clip(0, 1).fillna(0)
        + 0.15 * (1.0 - close_location).clip(0, 1)
    ).clip(0, 1)
    relative = pd.to_numeric(out["relative_amount"], errors="coerce")
    adv20 = pd.to_numeric(out.get("adv20_amount"), errors="coerce")
    active = pd.to_numeric(out.get("active_bar_ratio"), errors="coerce")
    illiquid = (
        adv20.lt(float(liquidity["illiquid_adv20_amount"]))
        | active.lt(float(liquidity["illiquid_active_bar_ratio"]))
        | relative.lt(0.20)
    )
    climax = (
        amount_z.ge(float(liquidity["climax_amount_z"]))
        & range_z.ge(float(liquidity["climax_range_z"]))
    )
    hot = (
        amount_z.ge(float(liquidity["hot_amount_z"]))
        & premium_z.ge(float(liquidity["premium_expansion_hot_z"]))
    ) | out["hot_money_risk_score_5m"].ge(float(liquidity["hot_money_reject_score"]))
    sweet = relative.between(
        float(liquidity["relative_amount_sweet_min"]),
        float(liquidity["relative_amount_sweet_max"]),
    ) & ~illiquid & ~hot & ~climax
    out["liquidity_regime_5m"] = np.select(
        [illiquid, climax, hot, sweet],
        ["ILLIQUID", "VOLUME_CLIMAX", "HOT", "LIQUIDITY_SWEET_SPOT"],
        default="NORMAL",
    )
    out["liquidity_sweet_spot_score_5m"] = (
        1.0 - (relative.fillna(0) - 1.0).abs().clip(0, 1)
    ) * (~illiquid).astype(float) * (1.0 - out["hot_money_risk_score_5m"])
    shock = (
        pd.to_numeric(out["stock_shock_z"], errors="coerce")
        >= float(config["stock_shock"]["default_return_z_min"])
    ) & (
        pd.to_numeric(out["stock_shock_amount_z"], errors="coerce")
        >= float(config["stock_shock"]["default_amount_z_min"])
    )
    out["stock_shock_flag"] = shock
    minimum = int(config["linkage"]["minimum_model_samples"])
    model_available = (
        pd.to_numeric(out["linkage_sample_count"], errors="coerce").ge(minimum)
        & pd.to_numeric(out["linkage_regression_r2"], errors="coerce").notna()
    )
    gap_positive = pd.to_numeric(out["linkage_gap_3bar"], errors="coerce").ge(
        float(config["linkage_gap"]["default_minimum_gap_bps"]) / 10000.0
    )
    liquidity_ok = out["liquidity_regime_5m"].isin(config["liquidity"]["accepted_regimes"])
    hot_reject = out["liquidity_regime_5m"].isin(["HOT", "VOLUME_CLIMAX"])
    repair = out["linkage_repair_started"].fillna(False)
    formal_source = out.get(
        "pit_universe_eligible", pd.Series(False, index=out.index, dtype=bool)
    )
    formal = formal_source.fillna(False).astype(bool)
    residual_baseline = (
        shock
        & pd.to_numeric(out["linkage_residual_z"], errors="coerce").le(-0.8)
        & ~illiquid
    )
    out["baseline_residual_lag_repair"] = residual_baseline
    out["strategy_linkage_gap"] = shock & model_available & gap_positive
    raw_shock_flag = (
        pd.to_numeric(out["stock_return_z_unadjusted"], errors="coerce")
        >= float(config["stock_shock"]["default_return_z_min"])
    ) & (
        pd.to_numeric(out["stock_shock_amount_z"], errors="coerce")
        >= float(config["stock_shock"]["default_amount_z_min"])
    )
    out["strategy_linkage_gap_unadjusted"] = (
        raw_shock_flag
        & pd.to_numeric(out["unadjusted_linkage_sample_count"], errors="coerce").ge(minimum)
        & pd.to_numeric(out["unadjusted_linkage_gap_3bar"], errors="coerce").ge(
            float(config["linkage_gap"]["default_minimum_gap_bps"]) / 10000.0
        )
    )
    out["strategy_linkage_repair"] = out["strategy_linkage_gap"] & repair
    out["strategy_linkage_liquidity"] = out["strategy_linkage_repair"] & liquidity_ok
    broad_liquidity = ~illiquid
    out["strategy_linkage_liquidity_no_hot_reject"] = (
        out["strategy_linkage_repair"] & broad_liquidity
    )
    out["strategy_linkage_hot_reject"] = out["strategy_linkage_liquidity"] & ~hot_reject
    no_response = out["strategy_linkage_gap"] & ~repair & (
        relative.lt(0.8) | amount_z.lt(0)
    )
    overshoot = shock & (
        pd.to_numeric(out["linkage_gap_3bar"], errors="coerce").le(0)
        | hot_reject
    ) & (premium_z.gt(0.5) | out["cb_vwap_dev"].gt(0.01))
    cb_leads = (
        pd.to_numeric(out["cb_shock_z"], errors="coerce").ge(
            float(config["events"]["cb_leads_shock_z_min"])
        )
        & out["cb_to_stock_beta_actionable_sum"].gt(0)
        & out["dominant_lead_direction"].eq("CB_LEADS_STOCK")
        & ~hot_reject
    )
    uncertain = (shock | cb_leads) & ~model_available
    out["event_type"] = np.select(
        [
            uncertain,
            overshoot,
            out["strategy_linkage_hot_reject"],
            no_response,
            cb_leads,
        ],
        [
            "LINKAGE_UNCERTAIN",
            "CB_OVERSHOOT_HOT_MONEY",
            "STOCK_LEADS_CB_REPAIR",
            "STOCK_LEADS_CB_NO_RESPONSE",
            "CB_LEADS_STOCK_INFORMATION",
        ],
        default="NONE",
    )
    out["formal_event_eligible"] = formal & out["event_type"].ne("LINKAGE_UNCERTAIN")
    out["event_action_state"] = np.select(
        [
            out["event_type"].eq("STOCK_LEADS_CB_REPAIR") & formal,
            out["event_type"].eq("CB_OVERSHOOT_HOT_MONEY"),
            out["event_type"].eq("STOCK_LEADS_CB_NO_RESPONSE"),
        ],
        ["RESEARCH_ACTION", "DO_NOT_CHASE", "REJECT_LAG"],
        default="WATCH_ONLY",
    )
    out["qiv_mc_formal_weight"] = 0.0
    out["lsmc_entry_enabled"] = False
    out["momentum_standalone_trade_signal"] = False
    return out


def deterministic_episode_id(
    trade_date: Any,
    bond_code: str,
    event_type: str,
    sequence: int,
) -> str:
    payload = f"{pd.Timestamp(trade_date).date()}|{bond_code}|{event_type}|{sequence}"
    return "lnk_" + hashlib.sha1(payload.encode("ascii")).hexdigest()[:16]


def deduplicate_event_episodes(
    features: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    events = features[features["event_type"].ne("NONE")].copy()
    if events.empty:
        return events
    events = events.sort_values(["trade_date", "bond_code", "event_type", "bar_slot"])
    merge_bars = int(config["linkage"]["episode_merge_bars"])
    anchors = []
    for (trade_date, bond_code, event_type), group in events.groupby(
        ["trade_date", "bond_code", "event_type"], sort=False
    ):
        group = group.sort_values("bar_slot")
        sequence = (group["bar_slot"].diff().fillna(merge_bars + 1) > merge_bars).cumsum()
        for episode_sequence, episode in group.groupby(sequence, sort=False):
            anchor = episode.iloc[0].copy()
            anchor["episode_id"] = deterministic_episode_id(
                trade_date, str(bond_code), str(event_type), int(episode_sequence)
            )
            anchor["episode_start_time"] = episode["bar_start"].min()
            anchor["episode_last_signal_time"] = episode["bar_start"].max()
            anchor["episode_signal_bars"] = int(len(episode))
            anchor["signal_time"] = anchor["bar_end"]
            anchor["first_seen_time"] = anchor["bar_end"]
            anchor["data_mode"] = "scheduled_backfill"
            anchors.append(anchor)
    return pd.DataFrame(anchors).sort_values(
        ["trade_date", "signal_time", "bond_code"]
    ).reset_index(drop=True)
