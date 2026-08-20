from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "hybrid_strategy_v07.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_hybrid_config(path: str | None = None) -> dict[str, Any]:
    import yaml

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"hybrid config must be a mapping: {config_path}")
    return config


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def robust_z_past(series: pd.Series, window: int = 60, min_periods: int = 20) -> pd.Series:
    shifted = pd.to_numeric(series, errors="coerce").shift(1)
    median = shifted.rolling(window, min_periods=min_periods).median()
    mad = (shifted - median).abs().rolling(window, min_periods=min_periods).median()
    return ((pd.to_numeric(series, errors="coerce") - median) / (1.4826 * mad + 1e-12)).clip(-8, 8)


def add_daily_features(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel.copy()
    df = panel.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df = df.sort_values(["bond_code", "trade_date"]).reset_index(drop=True)
    output: list[pd.DataFrame] = []
    for _, group in df.groupby("bond_code", sort=False):
        g = group.copy()
        gap = g["trade_date"].diff().dt.days.fillna(1)
        g["_segment"] = (gap > 15).cumsum()
        pieces: list[pd.DataFrame] = []
        for _, piece in g.groupby("_segment", sort=False):
            p = piece.copy()
            p["bond_return_1d"] = p["bond_close"].pct_change(fill_method=None)
            p["stock_return_1d"] = p["stock_close"].pct_change(fill_method=None)
            p["bond_return_5d"] = p["bond_close"].pct_change(5, fill_method=None)
            p["stock_return_5d"] = p["stock_close"].pct_change(5, fill_method=None)
            p["bond_return_20d"] = p["bond_close"].pct_change(20, fill_method=None)
            p["stock_return_20d"] = p["stock_close"].pct_change(20, fill_method=None)
            p["bond_rv_20d"] = p["bond_return_1d"].rolling(20, min_periods=10).std() * math.sqrt(252)
            p["stock_rv_20d"] = p["stock_return_1d"].rolling(20, min_periods=10).std() * math.sqrt(252)
            p["bond_ma20d"] = p["bond_close"].rolling(20, min_periods=10).mean()
            p["stock_ma20d"] = p["stock_close"].rolling(20, min_periods=10).mean()
            p["cb_ma20d_dev"] = p["bond_close"] / p["bond_ma20d"] - 1
            p["stock_ma20d_dev"] = p["stock_close"] / p["stock_ma20d"] - 1
            p["cb_ma90min_dev"] = (
                p["bond_close"] / pd.to_numeric(p.get("cb_ma90min"), errors="coerce") - 1
                if "cb_ma90min" in p.columns
                else np.nan
            )
            p["stock_ma90min_dev"] = (
                p["stock_close"] / pd.to_numeric(p.get("stock_ma90min"), errors="coerce") - 1
                if "stock_ma90min" in p.columns
                else np.nan
            )
            p["adv20_amount"] = p["bond_amount"].rolling(20, min_periods=10).mean()
            p["amount_z"] = robust_z_past(p["bond_amount"], 60, 20)
            p["turnover_z"] = robust_z_past(p.get("daily_turnover", pd.Series(np.nan, index=p.index)), 60, 20)
            p["high_low_range_proxy"] = p["bond_high"] / p["bond_low"].replace(0, np.nan) - 1
            p["high_low_range_proxy_z"] = robust_z_past(p["high_low_range_proxy"], 60, 20)
            p["close_location"] = (
                (p["bond_close"] - p["bond_low"])
                / (p["bond_high"] - p["bond_low"]).replace(0, np.nan)
            ).clip(0, 1)
            p["amihud"] = p["bond_return_1d"].abs() / p["bond_amount"].replace(0, np.nan)
            p["amihud_z"] = robust_z_past(np.log1p(p["amihud"] * 1e9), 60, 20)
            cov = p["bond_return_1d"].shift(1).rolling(60, min_periods=30).cov(p["stock_return_1d"].shift(1))
            var = p["stock_return_1d"].shift(1).rolling(60, min_periods=30).var()
            p["return_beta_60d"] = (cov / var.replace(0, np.nan)).clip(-2, 5)
            p["empirical_delta"] = p["return_beta_60d"] * p["bond_close"] / p["stock_close"].replace(0, np.nan)
            p["elasticity"] = p["return_beta_60d"]
            p["residual_daily"] = p["bond_return_1d"] - p["return_beta_60d"] * p["stock_return_1d"]
            p["residual_z_daily"] = robust_z_past(p["residual_daily"], 60, 20)
            p["residual_std_20d"] = p["residual_daily"].shift(1).rolling(20, min_periods=10).std()
            p["residual_change"] = p["residual_z_daily"].diff()
            p["premium_change"] = pd.to_numeric(p.get("premium", np.nan), errors="coerce").diff()
            p["premium_change_z"] = robust_z_past(p["premium_change"], 60, 20)
            p["active_bar_ratio"] = pd.to_numeric(p["bond_valid_bars"], errors="coerce") / 48.0
            p["zero_bar_ratio"] = pd.to_numeric(p["bond_zero_bars"], errors="coerce") / 48.0
            p["stock_market_return"] = p["stock_return_1d"]
            p["market_regime"] = np.select(
                [
                    p["stock_return_20d"] > 0.05,
                    p["stock_return_20d"] < -0.05,
                    p["adv20_amount"] < 2e7,
                ],
                ["RISK_ON", "RISK_OFF", "THIN_MARKET"],
                default="NEUTRAL",
            )
            p["ma_alignment_state"] = np.select(
                [
                    (p["cb_ma20d_dev"] > 0) & (p["stock_ma20d_dev"] > 0),
                    (p["cb_ma20d_dev"] < 0) & (p["stock_ma20d_dev"] < 0),
                ],
                ["BOTH_ABOVE_MA20", "BOTH_BELOW_MA20"],
                default="MIXED",
            )
            p["ma_monitor_score"] = (
                p[
                    [
                        "cb_ma20d_dev",
                        "stock_ma20d_dev",
                        "cb_ma90min_dev",
                        "stock_ma90min_dev",
                    ]
                ]
                .clip(-0.2, 0.2)
                .mean(axis=1)
            )
            debt = pd.to_numeric(p.get("debt_ratio", np.nan), errors="coerce")
            ocf = pd.to_numeric(p.get("operating_cash_flow", np.nan), errors="coerce")
            fcf = pd.to_numeric(p.get("free_cash_flow", np.nan), errors="coerce")
            p["fundamental_quality_bucket"] = np.select(
                [
                    (debt < 50) & (ocf > 0) & (fcf > 0),
                    (debt < 70) & ((ocf > 0) | (fcf > 0)),
                    debt.notna() | ocf.notna() | fcf.notna(),
                ],
                ["HIGH", "MEDIUM", "LOW"],
                default="UNKNOWN",
            )
            pieces.append(p)
        output.append(pd.concat(pieces, ignore_index=False))
    result = pd.concat(output, ignore_index=True).drop(columns=["_segment"], errors="ignore")
    result["forward_return_1d"] = result.groupby("bond_code")["bond_close"].shift(-1) / result["bond_close"] - 1
    for horizon in [2, 3, 5, 10]:
        result[f"forward_return_{horizon}d"] = (
            result.groupby("bond_code")["bond_close"].shift(-horizon) / result["bond_close"] - 1
        )
    result["next_open"] = result.groupby("bond_code")["bond_open"].shift(-1)
    result["next_close"] = result.groupby("bond_code")["bond_close"].shift(-1)
    result["next_trade_date"] = result.groupby("bond_code")["trade_date"].shift(-1)
    result["next_stock_return_1d"] = result.groupby("bond_code")["stock_return_1d"].shift(-1)
    result["next_open_to_close_return"] = result["next_close"] / result["next_open"] - 1
    result["signal_execution_lag"] = "next_trade_day_open"
    result["same_close_execution_forbidden"] = True
    return add_forward_path_labels(result)


def add_forward_path_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["max_return_5d"] = np.nan
    out["min_return_5d"] = np.nan
    out["mfe_5d"] = np.nan
    out["mae_5d"] = np.nan
    for _, idx in out.groupby("bond_code", sort=False).groups.items():
        positions = np.asarray(list(idx), dtype=int)
        prices = pd.to_numeric(out.loc[positions, "bond_close"], errors="coerce").to_numpy()
        for local_i, row_i in enumerate(positions):
            future = prices[local_i + 1 : local_i + 6]
            if len(future) == 0 or not np.isfinite(prices[local_i]):
                continue
            returns = future / prices[local_i] - 1
            out.at[row_i, "max_return_5d"] = np.nanmax(returns)
            out.at[row_i, "min_return_5d"] = np.nanmin(returns)
            out.at[row_i, "mfe_5d"] = np.nanmax(returns)
            out.at[row_i, "mae_5d"] = np.nanmin(returns)
    market_mean = out.groupby("trade_date")["forward_return_1d"].transform("mean")
    out["excess_return_vs_cb_market"] = out["forward_return_1d"] - market_mean
    out["beta_hedged_return"] = (
        out["forward_return_1d"] - out["return_beta_60d"] * out["next_stock_return_1d"]
    )
    return out


def classify_bonds(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = config["bond_type"]
    out = df.copy()
    adv = pd.to_numeric(out["adv20_amount"], errors="coerce")
    active = pd.to_numeric(out["active_bar_ratio"], errors="coerce")
    price = pd.to_numeric(out["bond_close"], errors="coerce")
    elasticity = pd.to_numeric(out["elasticity"], errors="coerce")
    premium = pd.to_numeric(out.get("premium", np.nan), errors="coerce")
    amount_z = pd.to_numeric(out["amount_z"], errors="coerce")
    range_z = pd.to_numeric(out["high_low_range_proxy_z"], errors="coerce")

    illiquid = (adv < float(cfg["illiquid_adv20_amount"])) | (
        active < float(cfg["illiquid_active_bar_ratio"])
    )
    rolling_floor = (
        out.assign(_price=price)
        .groupby("bond_code", sort=False)["_price"]
        .transform(lambda s: s.rolling(20, min_periods=10).min())
    )
    distress = (
        (rolling_floor <= float(cfg["distress_price_floor"]))
        & (pd.to_numeric(out["bond_return_20d"], errors="coerce") >= float(cfg["distress_reversal_20d_min"]))
    )
    clause = out.get("clause_driven_flag", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    speculative = (
        (premium >= float(cfg["speculative_premium_min"]))
        | (
            (price >= float(cfg["speculative_price_min"]))
            & ((amount_z >= 2.0) | (range_z >= 2.0))
        )
    )
    equity_like = (price >= float(cfg["equity_like_price_min"])) | (
        elasticity >= float(cfg["equity_like_elasticity_min"])
    )
    bond_like = (price <= float(cfg["bond_like_price_max"])) | (
        (elasticity <= float(cfg["bond_like_elasticity_max"])) & (price < 130)
    )
    out["bond_type"] = np.select(
        [illiquid, clause, distress, speculative, equity_like, bond_like],
        [
            "ILLIQUID",
            "CLAUSE_DRIVEN",
            "DISTRESS_REVERSAL",
            "SPECULATIVE_HIGH_PREMIUM",
            "EQUITY_LIKE",
            "BOND_LIKE",
        ],
        default="BALANCED_CORE",
    )
    out["clause_driven_flag"] = clause
    out["clause_data_available"] = out.get(
        "clause_data_available", pd.Series(False, index=out.index)
    ).fillna(False).astype(bool)
    out["distress_flag"] = distress
    available = pd.DataFrame(
        {
            "price": price.notna(),
            "elasticity": elasticity.notna(),
            "adv": adv.notna(),
            "active": active.notna(),
            "premium": premium.notna(),
            "fundamental": pd.to_numeric(out.get("debt_ratio", np.nan), errors="coerce").notna(),
        }
    ).sum(axis=1)
    out["bond_type_confidence"] = (0.35 + available * 0.10).clip(upper=0.95)
    out.loc[out["bond_type"].isin(["CLAUSE_DRIVEN", "DISTRESS_REVERSAL"]), "bond_type_confidence"] = (
        out.loc[out["bond_type"].isin(["CLAUSE_DRIVEN", "DISTRESS_REVERSAL"]), "bond_type_confidence"]
        .clip(upper=0.75)
    )
    out["classification_reason"] = (
        "price="
        + price.round(2).astype("string")
        + ";elasticity="
        + elasticity.round(3).astype("string")
        + ";adv20="
        + adv.round(0).astype("string")
        + ";premium_pti="
        + premium.round(2).astype("string")
    )
    out["bond_type_version"] = str(cfg["version"])
    out["intraday_eligible"] = ~out["bond_type"].isin(["ILLIQUID", "CLAUSE_DRIVEN", "BOND_LIKE"])
    out["daily_holding_eligible"] = ~out["bond_type"].isin(
        ["ILLIQUID", "CLAUSE_DRIVEN", "SPECULATIVE_HIGH_PREMIUM"]
    )
    out["lsmc_eligible"] = out["bond_type"].isin(["BALANCED_CORE", "EQUITY_LIKE", "BOND_LIKE"])
    if str(cfg.get("distress_lsmc_mode", "experimental")) == "experimental":
        out.loc[out["bond_type"] == "DISTRESS_REVERSAL", "lsmc_eligible"] = True
    out["lsmc_exclusion_reason"] = np.select(
        [
            out["bond_type"] == "CLAUSE_DRIVEN",
            out["bond_type"] == "ILLIQUID",
            out["bond_type"] == "SPECULATIVE_HIGH_PREMIUM",
            out["bond_type"] == "DISTRESS_REVERSAL",
        ],
        [
            "CLAUSE_DRIVEN_EXCLUDED",
            "ILLIQUID_EXCLUDED",
            "EXTREME_SPECULATIVE_EXCLUDED",
            "DISTRESS_EXPERIMENTAL_ONLY",
        ],
        default="",
    )
    out["classification_point_in_time"] = True
    return out


def compute_liquidity_and_friction(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = config["friction"]
    out = df.copy()
    adv = pd.to_numeric(out["adv20_amount"], errors="coerce")
    amount = pd.to_numeric(out["bond_amount"], errors="coerce")
    amount_z = pd.to_numeric(out["amount_z"], errors="coerce").fillna(0)
    range_z = pd.to_numeric(out["high_low_range_proxy_z"], errors="coerce").fillna(0)
    active = pd.to_numeric(out["active_bar_ratio"], errors="coerce").fillna(0)
    zero = pd.to_numeric(out["zero_bar_ratio"], errors="coerce").fillna(1)
    close_loc = pd.to_numeric(out["close_location"], errors="coerce").fillna(0.5)
    daily_range_bps = (
        pd.to_numeric(out["high_low_range_proxy"], errors="coerce").fillna(0).clip(0, 0.20) * 10000
    )

    illiquid_score = (
        (float(cfg["sweet_spot_adv20_min"]) / adv.replace(0, np.nan)).clip(0, 3).fillna(3)
        + (1 - active).clip(0, 1) * 2
        + zero.clip(0, 1)
    ) / 6
    climax_score = (
        amount_z.clip(lower=0) / max(float(cfg["overheated_amount_z"]), 1e-6)
        + range_z.clip(lower=0) / max(float(cfg["climax_range_z"]), 1e-6)
        + (0.5 - close_loc).clip(lower=0) * 2
    ) / 3
    sweet = (1 - illiquid_score.clip(0, 1)) * (1 - climax_score.clip(0, 1))
    out["liquidity_sweet_spot_score"] = sweet.clip(0, 1)
    out["volume_climax_score"] = climax_score.clip(0, 2)
    out["hot_money_risk_score"] = (
        0.45 * out["volume_climax_score"]
        + 0.35 * amount_z.clip(lower=0, upper=4) / 4
        + 0.20 * (pd.to_numeric(out["bond_return_1d"], errors="coerce") - pd.to_numeric(out["stock_return_1d"], errors="coerce")).clip(lower=0, upper=0.10) / 0.10
    ).clip(0, 1)
    out["retail_chase_risk_score"] = (
        0.50 * out["hot_money_risk_score"] + 0.30 * (1 - close_loc) + 0.20 * range_z.clip(0, 4) / 4
    ).clip(0, 1)
    out["liquidity_regime"] = np.select(
        [
            (adv < float(cfg["sweet_spot_adv20_min"])) | (active < 0.8),
            out["volume_climax_score"] >= 0.8,
            out["liquidity_sweet_spot_score"] >= 0.45,
        ],
        ["ILLIQUID", "OVERHEATED_CLIMAX", "LIQUIDITY_SWEET_SPOT"],
        default="NORMAL",
    )
    out["commission_bps"] = float(cfg["actual_fee_bps_proxy"]) * float(cfg["stress_multiplier"])
    out["spread_proxy_bps"] = (daily_range_bps * float(cfg["spread_range_fraction"])).clip(1, 30)
    out["slippage_bps"] = (
        daily_range_bps * float(cfg["slippage_range_fraction"]) * (0.5 + (close_loc - 0.5).abs())
    ).clip(1, 35)
    order = float(cfg["order_notional"])
    participation_bar = order / amount.replace(0, np.nan)
    participation_adv = order / adv.replace(0, np.nan)
    out["market_impact_bps"] = (
        float(cfg["impact_coefficient_bps"])
        * (np.sqrt(participation_bar.clip(lower=0).fillna(1)) + np.sqrt(participation_adv.clip(lower=0).fillna(1)))
    ).clip(0, 60)
    out["liquidity_risk_bps"] = (
        illiquid_score * float(cfg["liquidity_risk_max_bps"]) + climax_score.clip(0, 1) * 10
    ).clip(0, float(cfg["liquidity_risk_max_bps"]))
    out["total_friction_bps"] = out[
        ["commission_bps", "spread_proxy_bps", "slippage_bps", "market_impact_bps", "liquidity_risk_bps"]
    ].sum(axis=1)
    out["friction_confidence"] = (
        0.35
        + 0.20 * adv.notna().astype(float)
        + 0.15 * amount.notna().astype(float)
        + 0.15 * pd.to_numeric(out.get("daily_turnover", np.nan), errors="coerce").notna().astype(float)
        + 0.15 * pd.to_numeric(out.get("amihud", np.nan), errors="coerce").notna().astype(float)
    ).clip(0, 1)
    out["friction_source"] = "local_amount_turnover_amihud_range_proxy_stress5x"
    return out


def credit_spread_proxy(row: pd.Series, config: dict[str, Any]) -> tuple[float, str]:
    cfg = config["credit_spread_proxy"]
    bond_type = str(row.get("bond_type", "BALANCED_CORE"))
    spread = float(cfg["base_bps"].get(bond_type, 350))
    debt = safe_float(row.get("debt_ratio"))
    ocf = safe_float(row.get("operating_cash_flow"))
    fcf = safe_float(row.get("free_cash_flow"))
    if np.isfinite(debt) and debt > 70:
        spread += float(cfg["debt_ratio_penalty_bps"])
    if (np.isfinite(ocf) and ocf < 0) or (np.isfinite(fcf) and fcf < 0):
        spread += float(cfg["negative_cash_flow_penalty_bps"])
    if str(row.get("liquidity_regime", "")) == "ILLIQUID":
        spread += float(cfg["illiquidity_penalty_bps"])
    if bool(row.get("distress_flag", False)):
        spread += float(cfg["distress_penalty_bps"])
    source = str(cfg["source"]) + ";ytm_unavailable=true"
    return spread / 10000.0, source


def _normal_cdf(x: np.ndarray | float) -> np.ndarray | float:
    from scipy.special import ndtr

    return ndtr(x)


def proxy_convertible_price(
    stock_price: np.ndarray | float,
    conversion_price: np.ndarray | float,
    years: np.ndarray | float,
    risk_free_rate: float,
    credit_spread: np.ndarray | float,
    sigma: np.ndarray | float,
    par_value: float = 100.0,
) -> np.ndarray | float:
    s = np.asarray(stock_price, dtype=float)
    k_conv = np.asarray(conversion_price, dtype=float)
    t = np.maximum(np.asarray(years, dtype=float), 1 / 365)
    vol = np.maximum(np.asarray(sigma, dtype=float), 1e-4)
    spread = np.asarray(credit_spread, dtype=float)
    ratio = par_value / k_conv
    strike_value = par_value
    conversion_asset = ratio * s
    d1 = (np.log(np.maximum(conversion_asset, 1e-12) / strike_value) + (risk_free_rate + 0.5 * vol**2) * t) / (
        vol * np.sqrt(t)
    )
    d2 = d1 - vol * np.sqrt(t)
    option = conversion_asset * _normal_cdf(d1) - strike_value * np.exp(-risk_free_rate * t) * _normal_cdf(d2)
    floor = par_value * np.exp(-(risk_free_rate + spread) * t)
    return floor + option


def solve_qiv(row: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["qiv"]
    price = safe_float(row.get("bond_close"))
    has_explicit_pricing_stock = "qiv_stock_close" in row.index
    stock = safe_float(row.get("qiv_stock_close"))
    stock_price_source = str(row.get("qiv_stock_close_source") or "jqdata_fq_none_daily")
    if not has_explicit_pricing_stock:
        stock = safe_float(row.get("stock_close"))
        stock_price_source = "panel_stock_close"
    conversion = safe_float(row.get("conversion_price"))
    years = safe_float(row.get("days_to_maturity")) / 365.0
    if not all(np.isfinite(v) and v > 0 for v in [price, stock, conversion, years]):
        return {
            "qiv_raw": np.nan,
            "qiv_solve_success": False,
            "qiv_solve_error": "POINT_IN_TIME_INPUT_UNAVAILABLE",
            "qiv_iterations": 0,
        }
    spread, spread_source = credit_spread_proxy(row, config)
    risk_free = float(config["lsmc"]["risk_free_rate"])
    low = float(cfg["min_sigma"])
    high = float(cfg["max_sigma"])
    price_low = float(proxy_convertible_price(stock, conversion, years, risk_free, spread, low))
    price_high = float(proxy_convertible_price(stock, conversion, years, risk_free, spread, high))
    if not price_low <= price <= price_high:
        return {
            "qiv_raw": np.nan,
            "qiv_solve_success": False,
            "qiv_solve_error": f"MARKET_PRICE_OUTSIDE_PROXY_RANGE[{price_low:.3f},{price_high:.3f}]",
            "qiv_iterations": 0,
        }
    tolerance = float(cfg["solve_tolerance"])
    iterations = 0
    for iterations in range(1, int(cfg["max_iterations"]) + 1):
        mid = (low + high) / 2
        model_price = float(proxy_convertible_price(stock, conversion, years, risk_free, spread, mid))
        if abs(model_price - price) <= tolerance:
            break
        if model_price < price:
            low = mid
        else:
            high = mid
    qiv = (low + high) / 2
    return {
        "qiv_raw": qiv,
        "qiv_solve_success": True,
        "qiv_solve_error": "",
        "qiv_iterations": iterations,
        "qiv_model_assumptions": json.dumps(
            {
                "model": "discounted_par_plus_european_conversion_option_proxy",
                "credit_spread_source": spread_source,
                "coupon": "unavailable_zero_proxy",
                "clauses": "not_modelled",
                "ytm": "unavailable_not_used",
                "stock_price_source": stock_price_source,
            },
            ensure_ascii=True,
        ),
    }


def monte_carlo_price(row: pd.Series, config: dict[str, Any], paths: int | None = None) -> dict[str, Any]:
    cfg = config["monte_carlo"]
    path_count = int(paths or cfg["default_paths"])
    has_explicit_pricing_stock = "qiv_stock_close" in row.index
    stock = safe_float(row.get("qiv_stock_close"))
    if not has_explicit_pricing_stock:
        stock = safe_float(row.get("stock_close"))
    conversion = safe_float(row.get("conversion_price"))
    years = safe_float(row.get("days_to_maturity")) / 365.0
    sigma = safe_float(row.get("stock_rv_20d"))
    if not np.isfinite(sigma):
        sigma = {
            "EQUITY_LIKE": 0.45,
            "BALANCED_CORE": 0.35,
            "BOND_LIKE": 0.25,
            "DISTRESS_REVERSAL": 0.60,
        }.get(str(row.get("bond_type")), 0.40)
    if not all(np.isfinite(v) and v > 0 for v in [stock, conversion, years, sigma]):
        return {"mc_model_price": np.nan, "mc_path_count": 0, "mc_runtime_ms": 0.0}
    spread, source = credit_spread_proxy(row, config)
    risk_free = float(config["lsmc"]["risk_free_rate"])
    seed = int(cfg["random_seed"]) + int(str(row.get("bond_code", "0"))[-4:] or 0) + int(
        pd.Timestamp(row.get("trade_date")).strftime("%Y%m%d")
    )
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(path_count)
    par = float(cfg["redemption_value_proxy"])
    ratio = float(cfg["par_value"]) / conversion
    start = time.perf_counter()

    def prices(s0: float, vol: float, maturity: float) -> np.ndarray:
        terminal = s0 * np.exp((risk_free - 0.5 * vol**2) * maturity + vol * math.sqrt(maturity) * z)
        payoff = np.maximum(par, ratio * terminal)
        return np.exp(-(risk_free + spread) * maturity) * payoff

    discounted = prices(stock, sigma, years)
    price = float(discounted.mean())
    bump = float(cfg["finite_difference_stock_pct"])
    up = float(prices(stock * (1 + bump), sigma, years).mean())
    down = float(prices(stock * (1 - bump), sigma, years).mean())
    delta = (up - down) / (2 * stock * bump)
    gamma = (up - 2 * price + down) / (stock * bump) ** 2
    vol_bump = float(cfg["finite_difference_vol"])
    vega = (
        float(prices(stock, sigma + vol_bump, years).mean())
        - float(prices(stock, max(sigma - vol_bump, 1e-4), years).mean())
    ) / (2 * vol_bump)
    theta_years = max(years - 1 / 365, 1 / 365)
    theta = float(prices(stock, sigma, theta_years).mean()) - price
    sensitivities: dict[str, float] = {}
    for shift in config["lsmc"]["credit_spread_sensitivity_bps"]:
        shifted = spread + float(shift) / 10000
        sensitivities[str(int(shift))] = float(np.mean(np.maximum(par, ratio * (
            stock * np.exp((risk_free - 0.5 * sigma**2) * years + sigma * math.sqrt(years) * z)
        ))) * np.exp(-(risk_free + shifted) * years))
    runtime_ms = (time.perf_counter() - start) * 1000
    return {
        "mc_model_price": price,
        "mc_price_std": float(discounted.std(ddof=1)),
        "mc_price_p05": float(np.quantile(discounted, 0.05)),
        "mc_price_p50": float(np.quantile(discounted, 0.50)),
        "mc_price_p95": float(np.quantile(discounted, 0.95)),
        "mc_delta": delta,
        "mc_gamma": gamma,
        "mc_vega": vega,
        "mc_theta_1d": theta,
        "mc_runtime_ms": runtime_ms,
        "mc_path_count": path_count,
        "mc_seed": seed,
        "mc_model_version": str(cfg["model_version"]),
        "model_assumption_flags": (
            "terminal_conversion_or_par_proxy;no_coupon;no_call_put_reset;"
            f"credit_proxy={source};ytm_unavailable"
        ),
        "credit_spread_proxy": spread,
        "credit_spread_source": source,
        "mc_credit_spread_sensitivity": json.dumps(sensitivities, sort_keys=True),
    }


def add_qiv_and_mc(df: pd.DataFrame, config: dict[str, Any], paths: int | None = None) -> pd.DataFrame:
    out = df.copy()
    def column(name: str, default: Any = np.nan) -> pd.Series:
        if name in out.columns:
            return out[name]
        return pd.Series(default, index=out.index)

    qiv_rows = [solve_qiv(row, config) for _, row in out.iterrows()]
    qiv = pd.DataFrame(qiv_rows, index=out.index)
    out = pd.concat([out, qiv], axis=1)
    bond_vwap = pd.to_numeric(column("bond_amount"), errors="coerce") / pd.to_numeric(
        column("bond_volume"), errors="coerce"
    ).replace(0, np.nan)
    qiv_vwap_values: list[float] = []
    for idx, row in out.iterrows():
        vwap = safe_float(bond_vwap.loc[idx])
        if not np.isfinite(vwap) or vwap <= 0:
            qiv_vwap_values.append(np.nan)
            continue
        vwap_row = row.copy()
        vwap_row["bond_close"] = vwap
        qiv_vwap_values.append(safe_float(solve_qiv(vwap_row, config).get("qiv_raw")))
    out["qiv_vwap"] = qiv_vwap_values
    out["qiv_eod"] = out["qiv_raw"]
    out["qiv_smooth"] = out.groupby("bond_code")["qiv_raw"].transform(
        lambda s: s.ewm(span=int(config["qiv"]["smooth_span"]), adjust=False, min_periods=1).mean()
    )
    relative_z = pd.to_numeric(column("amount_z"), errors="coerce")
    premium_z = pd.to_numeric(column("premium_change_z"), errors="coerce")
    range_z = pd.to_numeric(column("high_low_range_proxy_z"), errors="coerce")
    close_location = pd.to_numeric(column("close_location"), errors="coerce")
    detached = (
        pd.to_numeric(column("bond_return_1d"), errors="coerce")
        - pd.to_numeric(column("stock_return_1d"), errors="coerce")
    ).abs() > 0.05
    out["qiv_contamination_assessable"] = (
        relative_z.notna()
        & premium_z.notna()
        & range_z.notna()
        & close_location.notna()
    )
    out["qiv_microstructure_contaminated"] = (
        (relative_z >= float(config["qiv"]["contamination_relative_amount_z"]))
        | (premium_z >= float(config["qiv"]["contamination_premium_change_z"]))
        | (range_z >= float(config["qiv"]["contamination_range_z"]))
        | ((close_location <= float(config["qiv"]["contamination_close_location_max"])) & detached)
    )
    out["qiv_contamination_status"] = np.select(
        [
            ~out["qiv_contamination_assessable"],
            out["qiv_microstructure_contaminated"],
        ],
        ["UNAVAILABLE", "CONTAMINATED"],
        default="CLEAN",
    )
    out["qiv_confidence"] = np.where(
        out["qiv_solve_success"],
        np.where(
            ~out["qiv_contamination_assessable"],
            0.35,
            np.where(out["qiv_microstructure_contaminated"], 0.25, 0.70),
        ),
        0.0,
    )
    mc_rows = [
        monte_carlo_price(row, config, paths=paths) if bool(row.get("qiv_solve_success", False)) else {
            "mc_model_price": np.nan,
            "mc_path_count": 0,
            "mc_runtime_ms": 0.0,
        }
        for _, row in out.iterrows()
    ]
    mc = pd.DataFrame(mc_rows, index=out.index)
    out = pd.concat([out, mc], axis=1)
    out["mc_price_gap"] = out["mc_model_price"] - out["bond_close"]
    out["mc_price_gap_pct"] = out["mc_price_gap"] / out["bond_close"]
    clean_qiv = out["qiv_smooth"].mask(out["qiv_microstructure_contaminated"])
    out["qiv_rank_cross_section"] = clean_qiv.groupby(out["trade_date"]).rank(pct=True)
    out["qiv_rank_within_bond_type"] = clean_qiv.groupby([out["trade_date"], out["bond_type"]]).rank(pct=True)
    parity_bucket = pd.cut(
        pd.to_numeric(column("parity"), errors="coerce"),
        [-np.inf, 90, 100, 115, 130, np.inf],
        labels=["LT90", "90_100", "100_115", "115_130", "GE130"],
    )
    price_bucket = pd.cut(
        out["bond_close"],
        [-np.inf, 110, 120, 130, np.inf],
        labels=["LT110", "110_120", "120_130", "GE130"],
    )
    adv_rank = pd.to_numeric(column("adv20_amount"), errors="coerce").rank(method="first")
    liquidity_bucket = (
        pd.qcut(adv_rank, 3, labels=["LOW", "MID", "HIGH"])
        if adv_rank.notna().sum() >= 3
        else pd.Series(pd.Categorical([np.nan] * len(out)), index=out.index)
    )
    out["qiv_rank_within_parity_bucket"] = clean_qiv.groupby([out["trade_date"], parity_bucket], observed=True).rank(pct=True)
    out["qiv_rank_within_price_bucket"] = clean_qiv.groupby([out["trade_date"], price_bucket], observed=True).rank(pct=True)
    out["qiv_rank_within_liquidity_bucket"] = clean_qiv.groupby([out["trade_date"], liquidity_bucket], observed=True).rank(pct=True)
    out["qiv_minus_stock_rv_20d"] = out["qiv_smooth"] - out["stock_rv_20d"]
    out["qiv_minus_stock_rv_5d"] = out["qiv_smooth"] - (
        out.groupby("bond_code")["stock_return_1d"].transform(lambda s: s.rolling(5, min_periods=3).std() * math.sqrt(252))
    )
    out["qiv_relative_value_score"] = (
        (0.5 - out["qiv_rank_within_bond_type"]).fillna(0)
        + out["mc_price_gap_pct"].clip(-0.5, 0.5).fillna(0)
    )
    qiv_rv_usable = (
        out["qiv_solve_success"].fillna(False)
        & out["qiv_contamination_assessable"].fillna(False)
        & ~out["qiv_microstructure_contaminated"].fillna(False)
        & out["mc_model_price"].notna()
    )
    out.loc[~qiv_rv_usable, "qiv_relative_value_score"] = np.nan
    return out


def _ridge_predict(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    if valid.sum() < x.shape[1] + 5:
        mean = float(np.nanmean(y)) if np.isfinite(y).any() else 0.0
        return np.full(len(y), mean), np.nan
    xv = x[valid]
    yv = y[valid]
    xtx = xv.T @ xv
    beta = np.linalg.solve(xtx + alpha * np.eye(xtx.shape[0]), xv.T @ yv)
    pred = x @ beta
    denom = np.sum((yv - yv.mean()) ** 2)
    r2 = 1 - np.sum((yv - xv @ beta) ** 2) / denom if denom > 0 else np.nan
    return pred, float(r2)


def _continuation_predict(
    x: np.ndarray,
    y: np.ndarray,
    model: str,
    alpha: float,
) -> tuple[np.ndarray, float]:
    if model != "huber":
        return _ridge_predict(x, y, 0.0 if model == "ols" else alpha)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    if valid.sum() < x.shape[1] + 5:
        return _ridge_predict(x, y, alpha)
    from sklearn.linear_model import HuberRegressor

    estimator = HuberRegressor(alpha=alpha, max_iter=100)
    try:
        estimator.fit(x[valid], y[valid])
        pred = estimator.predict(x)
        score = estimator.score(x[valid], y[valid])
        return pred, float(score)
    except Exception:
        return _ridge_predict(x, y, alpha)


def _lsmc_path_basis(
    bond_path: np.ndarray,
    stock_path: np.ndarray,
    initial_bond: float,
    initial_stock: float,
    residual_shock: np.ndarray,
) -> np.ndarray:
    bond_ratio = bond_path / initial_bond - 1
    stock_ratio = stock_path / initial_stock - 1
    return np.column_stack(
        [
            np.ones_like(bond_ratio),
            bond_ratio,
            bond_ratio**2,
            stock_ratio,
            stock_ratio**2,
            residual_shock,
            bond_ratio * stock_ratio,
        ]
    )


@dataclass
class LSMCResult:
    immediate_liquidation_value: float
    continuation_value: float
    hold_edge: float
    hold_probability: float
    expected_holding_days: float
    recommended_action: str
    regression_r2: float
    confidence: float
    path_count: int
    exclusion_reason: str
    path_model: str
    structural_mapping_terminal_mean: float = np.nan
    empirical_mapping_terminal_mean: float = np.nan
    structural_mapping_available: bool = False
    empirical_mapping_available: bool = False


def run_lsmc_row(
    row: pd.Series,
    history: pd.DataFrame,
    config: dict[str, Any],
    path_models: Iterable[str] | None = None,
    regression_model: str | None = None,
    paths: int | None = None,
) -> LSMCResult:
    cfg = config["lsmc"]
    friction = safe_float(row.get("total_friction_bps"), 50.0) / 10000
    current_bond = safe_float(row.get("bond_close"))
    current_stock = safe_float(row.get("stock_close"))
    immediate = current_bond * (1 - friction)
    raw_exclusion = row.get("lsmc_exclusion_reason", "")
    exclusion = "" if pd.isna(raw_exclusion) else str(raw_exclusion or "")
    if not bool(row.get("lsmc_eligible", False)):
        return LSMCResult(immediate, np.nan, np.nan, 0.0, 0.0, "WATCH_ONLY", 0.0, 0.0, 0, exclusion or "NOT_ELIGIBLE", "")
    if str(row.get("bond_type")) == "DISTRESS_REVERSAL":
        exclusion = "DISTRESS_EXPERIMENTAL_ONLY"
    valid_history = history[
        (history["trade_date"] < row["trade_date"])
        & (history["bond_type"] == row["bond_type"])
    ].copy()
    residual_pool = pd.to_numeric(valid_history.get("residual_daily", np.nan), errors="coerce").dropna().to_numpy()
    stock_pool = pd.to_numeric(valid_history.get("stock_return_1d", np.nan), errors="coerce").dropna().to_numpy()
    min_history = int(cfg["minimum_history_days"])
    if len(residual_pool) < min_history or len(stock_pool) < min_history:
        reason = ";".join(part for part in [exclusion, "INSUFFICIENT_PAST_HISTORY"] if part)
        return LSMCResult(immediate, np.nan, np.nan, 0.0, 0.0, "WATCH_ONLY", 0.0, 0.0, 0, reason, "")
    horizon = int(cfg["max_horizon_days"])
    path_count = int(paths or cfg["simulation_paths"])
    seed = int(cfg["random_seed"]) + int(str(row.get("bond_code", "0"))[-4:] or 0) + int(
        pd.Timestamp(row["trade_date"]).strftime("%Y%m%d")
    )
    rng = np.random.default_rng(seed)
    models = list(path_models or cfg["path_models"])
    beta = safe_float(row.get("return_beta_60d"), 0.5)
    sigma_stock = safe_float(row.get("stock_rv_20d"), np.nan)
    if not np.isfinite(sigma_stock):
        sigma_stock = float(np.nanstd(stock_pool) * math.sqrt(252))
    sigma_stock = float(np.clip(sigma_stock, 0.08, 1.20))
    risk_free = float(cfg["risk_free_rate"])
    dt = 1 / 252
    combined_bond: list[np.ndarray] = []
    combined_stock: list[np.ndarray] = []
    combined_residual: list[np.ndarray] = []
    structural_mapping_terminal_mean = np.nan
    empirical_terminal_values: list[float] = []
    structural_mapping_available = False

    if "gbm" in models:
        z = rng.standard_normal((path_count, horizon))
        stock_returns = (risk_free - 0.5 * sigma_stock**2) * dt + sigma_stock * math.sqrt(dt) * z
        residual = rng.choice(residual_pool, size=(path_count, horizon), replace=True)
        stock_path = current_stock * np.exp(np.cumsum(stock_returns, axis=1))
        empirical_bond = current_bond * np.exp(np.cumsum(beta * stock_returns + residual, axis=1))
        empirical_terminal_values.append(float(np.mean(empirical_bond[:, -1])))
        structural_available = all(
            np.isfinite(safe_float(row.get(name))) and safe_float(row.get(name)) > 0
            for name in ["conversion_price", "days_to_maturity"]
        )
        if structural_available:
            conversion = safe_float(row["conversion_price"])
            years0 = safe_float(row["days_to_maturity"]) / 365
            spread, _ = credit_spread_proxy(row, config)
            qiv = safe_float(row.get("qiv_smooth"), sigma_stock)
            remaining = np.maximum(years0 - np.arange(1, horizon + 1) / 252, 1 / 365)
            structural = proxy_convertible_price(
                stock_path,
                conversion,
                remaining[np.newaxis, :],
                risk_free,
                spread,
                qiv,
            )
            structural_mapping_terminal_mean = float(np.mean(structural[:, -1]))
            structural_mapping_available = True
            weight = float(cfg["structural_weight"])
            bond_path = weight * structural + (1 - weight) * empirical_bond
        else:
            bond_path = empirical_bond
        combined_bond.append(bond_path)
        combined_stock.append(stock_path)
        combined_residual.append(residual)

    if "empirical_bootstrap" in models:
        sample_idx = rng.integers(0, min(len(residual_pool), len(stock_pool)), size=(path_count, horizon))
        stock_returns = stock_pool[sample_idx]
        residual = residual_pool[sample_idx]
        stock_path = current_stock * np.exp(np.cumsum(np.log1p(np.clip(stock_returns, -0.5, 0.5)), axis=1))
        bond_path = current_bond * np.exp(
            np.cumsum(np.log1p(np.clip(beta * stock_returns + residual, -0.5, 0.5)), axis=1)
        )
        empirical_terminal_values.append(float(np.mean(bond_path[:, -1])))
        combined_bond.append(bond_path)
        combined_stock.append(stock_path)
        combined_residual.append(residual)

    bond_paths = np.concatenate(combined_bond, axis=0)
    stock_paths = np.concatenate(combined_stock, axis=0)
    residual_paths = np.concatenate(combined_residual, axis=0)
    n = len(bond_paths)
    discount = math.exp(-risk_free / 252)
    liquidation = bond_paths * (1 - friction)
    values = liquidation[:, -1].copy()
    exit_day = np.full(n, horizon, dtype=float)
    r2_values: list[float] = []
    effective_regression = regression_model or cfg["default_regression_model"]
    alpha = float(cfg["ridge_alpha"])
    for t in range(horizon - 2, -1, -1):
        future_value = values * discount + current_bond * float(cfg["carry_proxy_annual"]) / 252
        x = _lsmc_path_basis(
            bond_paths[:, t],
            stock_paths[:, t],
            current_bond,
            current_stock,
            residual_paths[:, t],
        )
        continuation_hat, r2 = _continuation_predict(
            x, future_value, effective_regression, alpha
        )
        exercise = liquidation[:, t] >= continuation_hat
        values = np.where(exercise, liquidation[:, t], future_value)
        exit_day = np.where(exercise, t + 1, exit_day)
        if np.isfinite(r2):
            r2_values.append(r2)
    continuation = float(np.mean(values) * discount)
    hold_edge = continuation - immediate
    hold_probability = float(np.mean(exit_day > 1))
    expected_days = float(np.mean(exit_day))
    required_buffer = current_bond * (
        float(cfg["daily_rebalance_friction_bps"])
        + float(cfg["overnight_risk_buffer_bps"])
        + float(cfg["model_uncertainty_buffer_bps"])
    ) / 10000
    if hold_edge > required_buffer:
        action = "HOLD"
    elif hold_edge > 0:
        action = "REDUCE"
    else:
        action = "SELL"
    has_regression_diagnostic = bool(r2_values)
    confidence = min(
        0.9,
        0.30
        + 0.20 * (len(residual_pool) >= 100)
        + 0.20 * (n >= 1000)
        + 0.20 * has_regression_diagnostic,
    )
    if exclusion:
        confidence = min(confidence, 0.45)
    return LSMCResult(
        immediate,
        continuation,
        hold_edge,
        hold_probability,
        expected_days,
        action,
        float(np.nanmean(r2_values)) if r2_values else np.nan,
        confidence,
        n,
        exclusion,
        ",".join(models),
        structural_mapping_terminal_mean,
        float(np.mean(empirical_terminal_values)) if empirical_terminal_values else np.nan,
        structural_mapping_available,
        bool(empirical_terminal_values),
    )


def select_trade_horizon(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    intraday_cfg = config["intraday"]
    residual_edge = (
        -pd.to_numeric(out["residual_z_daily"], errors="coerce").clip(upper=0)
        * pd.to_numeric(out["residual_std_20d"], errors="coerce").fillna(0)
        * float(intraday_cfg["repair_fraction"])
        * 10000
    )
    rv_edge = (
        pd.to_numeric(out.get("mc_price_gap_pct", np.nan), errors="coerce")
        .fillna(0)
        .clip(lower=0)
        * 10000
    )
    out["expected_intraday_edge_bps"] = (
        float(intraday_cfg["residual_weight"]) * residual_edge
        + float(intraday_cfg["relative_value_weight"]) * rv_edge
        + float(intraday_cfg["liquidity_weight"]) * out["liquidity_sweet_spot_score"] * 25
        - float(intraday_cfg["premium_risk_weight"]) * out["hot_money_risk_score"] * 50
        + float(intraday_cfg["momentum_weight"])
        * pd.to_numeric(out["stock_return_1d"], errors="coerce").clip(-0.05, 0.05)
        * 10000
    )
    out["intraday_net_edge_bps"] = out["expected_intraday_edge_bps"] - out["total_friction_bps"]
    intraday_ok = (
        out["intraday_eligible"].fillna(False)
        & (out["liquidity_regime"] == "LIQUIDITY_SWEET_SPOT")
        & (out["hot_money_risk_score"] < 0.65)
        & (out["residual_z_daily"] <= float(intraday_cfg["residual_entry_z"]))
        & (out["intraday_net_edge_bps"] >= float(intraday_cfg["min_net_edge_bps"]))
        & (
            out["intraday_net_edge_bps"]
            >= out["total_friction_bps"] * float(intraday_cfg["friction_safety_multiplier"])
        )
    )
    hold_edge_bps = (
        pd.to_numeric(out.get("lsmc_hold_edge", np.nan), errors="coerce")
        / pd.to_numeric(out["bond_close"], errors="coerce")
        * 10000
    )
    out["daily_hold_edge_bps"] = hold_edge_bps
    buffers = (
        float(config["lsmc"]["daily_rebalance_friction_bps"])
        + float(config["lsmc"]["overnight_risk_buffer_bps"])
        + float(config["lsmc"]["model_uncertainty_buffer_bps"])
    )
    daily_ok = (
        out["daily_holding_eligible"].fillna(False)
        & out["lsmc_eligible"].fillna(False)
        & (hold_edge_bps > buffers)
        & (out.get("qiv_microstructure_contaminated", False) != True)
        & (out["liquidity_regime"] != "ILLIQUID")
        & ~out["bond_type"].isin(["CLAUSE_DRIVEN", "SPECULATIVE_HIGH_PREMIUM"])
    )
    out["selected_horizon"] = np.select(
        [
            out["bond_type"].isin(["CLAUSE_DRIVEN", "ILLIQUID"]),
            intraday_ok,
            daily_ok,
            out["intraday_eligible"] | out["daily_holding_eligible"],
        ],
        ["EXCLUDE", "INTRADAY_T0", "OVERNIGHT_DAILY", "WATCH_ONLY"],
        default="EXCLUDE",
    )
    out["recommended_action"] = np.select(
        [
            out["selected_horizon"] == "INTRADAY_T0",
            out["selected_horizon"] == "OVERNIGHT_DAILY",
            out["selected_horizon"] == "EXCLUDE",
        ],
        ["INTRADAY_T0", "OPEN_DAILY", "EXCLUDE"],
        default="WATCH_ONLY",
    )
    out["loss_to_overnight_prohibition"] = bool(intraday_cfg["loss_to_overnight_prohibition"])
    failed_intraday = pd.to_numeric(out.get("current_intraday_pnl", np.nan), errors="coerce") < 0
    out["carry_overnight_action"] = np.select(
        [
            failed_intraday & ~daily_ok,
            daily_ok & (out["lsmc_recommended_action"] == "HOLD"),
            daily_ok & (out["lsmc_recommended_action"] == "REDUCE"),
        ],
        ["FLATTEN_BEFORE_CLOSE", "CARRY_OVERNIGHT", "REDUCE_AND_CARRY"],
        default="FLATTEN_BEFORE_CLOSE",
    )
    out["reason_text"] = (
        "type="
        + out["bond_type"].astype(str)
        + ";intraday_net_edge_bps="
        + out["intraday_net_edge_bps"].round(1).astype(str)
        + ";daily_hold_edge_bps="
        + out["daily_hold_edge_bps"].round(1).astype(str)
        + ";friction_bps="
        + out["total_friction_bps"].round(1).astype(str)
    )
    out["risk_text"] = (
        "liquidity="
        + out["liquidity_regime"].astype(str)
        + ";hot_money="
        + out["hot_money_risk_score"].round(2).astype(str)
        + ";qiv_contaminated="
        + out.get("qiv_microstructure_contaminated", False).astype(str)
    )
    out["research_only"] = True
    out["version"] = str(config["strategy"]["version"])
    return out


def portfolio_metrics(returns: pd.Series, turnover: pd.Series | None = None) -> dict[str, Any]:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return {
            "days": 0,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
            "turnover": np.nan,
        }
    curve = (1 + values).cumprod()
    drawdown = curve / curve.cummax() - 1
    annualized = curve.iloc[-1] ** (252 / len(values)) - 1 if curve.iloc[-1] > 0 else -1
    std = values.std(ddof=1)
    return {
        "days": int(len(values)),
        "total_return": float(curve.iloc[-1] - 1),
        "annualized_return": float(annualized),
        "sharpe": float(values.mean() / std * math.sqrt(252)) if std > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((values > 0).mean()),
        "worst_day": float(values.min()),
        "mean_daily_return": float(values.mean()),
        "turnover": float(pd.to_numeric(turnover, errors="coerce").mean()) if turnover is not None else np.nan,
    }
