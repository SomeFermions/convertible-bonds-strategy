import logging
import math
import re
from typing import Any

import akshare as ak
import pandas as pd


LOGGER = logging.getLogger(__name__)

EXPECTED_COLUMNS = ["bond_code", "bond_name", "ak_symbol", "market"]

CODE_COLUMNS = ["代码", "债券代码", "证券代码", "symbol", "bond_code", "raw_code"]
NAME_COLUMNS = ["名称", "债券简称", "证券简称", "name", "bond_name"]


def normalize_bond_code(value: Any) -> str | None:
    """Normalize AKShare/Sina bond identifiers to a plain 6-digit bond code."""
    if value is None or pd.isna(value):
        return None

    if isinstance(value, float):
        if math.isnan(value) or not value.is_integer():
            return None
        value = int(value)

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    text = text.lower()

    prefixed = re.match(r"^(sh|sz)(\d{6})$", text)
    if prefixed:
        return prefixed.group(2)

    suffixed = re.match(r"^(\d{6})\.(sh|sz)$", text)
    if suffixed:
        return suffixed.group(1)

    numeric_float = re.match(r"^(\d{6})\.0+$", text)
    if numeric_float:
        return numeric_float.group(1)

    digits = re.sub(r"\D", "", text)
    if len(digits) == 6:
        return digits

    return None


def infer_market(bond_code: str | None) -> str | None:
    if not bond_code:
        return None
    if bond_code.startswith(("110", "111", "113", "118")):
        return "SH"
    if bond_code.startswith(("123", "127", "128")):
        return "SZ"
    return None


def to_ak_bond_symbol(bond_code: str | None) -> str | None:
    market = infer_market(bond_code)
    if market == "SH":
        return f"sh{bond_code}"
    if market == "SZ":
        return f"sz{bond_code}"
    return None


def _first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def get_cb_universe() -> pd.DataFrame:
    """
    返回字段:
    bond_code, bond_name, ak_symbol, market
    """
    df = ak.bond_zh_hs_cov_spot()
    raw_count = len(df)

    if df is None or df.empty:
        LOGGER.warning("AKShare returned an empty convertible bond universe")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    code_col = _first_existing_column(list(df.columns), CODE_COLUMNS)
    name_col = _first_existing_column(list(df.columns), NAME_COLUMNS)

    if code_col is None:
        raise ValueError(f"Cannot find bond code column. columns={list(df.columns)}")

    out = pd.DataFrame()
    out["bond_code"] = df[code_col].apply(normalize_bond_code)
    out["bond_name"] = df[name_col].astype(str).str.strip() if name_col else ""
    out["market"] = out["bond_code"].apply(infer_market)
    out["ak_symbol"] = out["bond_code"].apply(to_ak_bond_symbol)

    valid_code_count = out["bond_code"].notna().sum()
    valid_symbol_count = out["ak_symbol"].notna().sum()
    before_dedup_count = len(out)

    out = out.dropna(subset=["bond_code", "ak_symbol"])
    out = out.drop_duplicates(subset=["bond_code"])

    LOGGER.info(
        "Universe rows: raw=%s valid_codes=%s valid_symbols=%s duplicates_removed=%s final=%s",
        raw_count,
        valid_code_count,
        valid_symbol_count,
        before_dedup_count - len(out),
        len(out),
    )

    return out[EXPECTED_COLUMNS].reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    universe = get_cb_universe()
    print(universe.head(20))
    print(universe["market"].value_counts(dropna=False))
