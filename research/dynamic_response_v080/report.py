from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import output_dir


LOGGER = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_plots(
    surface: pd.DataFrame,
    forward: pd.DataFrame,
    states: pd.DataFrame,
    gap_deciles: pd.DataFrame,
    state_outcomes: pd.DataFrame,
    model_metrics: pd.DataFrame,
    quantile: pd.DataFrame,
    surfaces: pd.DataFrame,
    ale_profiles: pd.DataFrame,
    transition_payload: dict[str, Any],
    config: dict[str, Any],
) -> list[Path]:
    plot_dir = output_dir(config) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(config["evaluation"]["plot_dpi"])
    paths: list[Path] = []

    if {"surface_target_cb_return", "prediction_lightgbm_q50"}.issubset(surface.columns):
        sample = surface[["surface_target_cb_return", "prediction_lightgbm_q50"]].dropna()
        if len(sample) > 100000:
            sample = sample.sample(100000, random_state=int(config["strategy"]["random_seed"]))
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.hexbin(
            sample["prediction_lightgbm_q50"],
            sample["surface_target_cb_return"],
            gridsize=70,
            bins="log",
            mincnt=1,
            cmap="viridis",
        )
        ax.set(xlabel="OOF predicted CB return", ylabel="Actual CB return", title="OOF actual vs predicted")
        path = plot_dir / "oof_actual_vs_predicted.png"
        _save_figure(fig, path, dpi)
        paths.append(path)

        residual = sample["surface_target_cb_return"] - sample["prediction_lightgbm_q50"]
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(residual.clip(residual.quantile(0.005), residual.quantile(0.995)), bins=80, ax=ax)
        ax.set(title="OOF response-gap distribution", xlabel="Actual - predicted")
        path = plot_dir / "response_gap_distribution.png"
        _save_figure(fig, path, dpi)
        paths.append(path)

    if not gap_deciles.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        subset = gap_deciles[gap_deciles["horizon"].eq("60m")]
        sns.lineplot(data=subset, x="response_gap_decile", y="forward_return", marker="o", ax=ax, label="CB return")
        sns.lineplot(data=subset, x="response_gap_decile", y="residual_return", marker="s", ax=ax, label="Residual return")
        ax.set(title="Response-gap decile outcomes (60m)")
        path = plot_dir / "response_gap_decile_outcome.png"
        _save_figure(fig, path, dpi)
        paths.append(path)

    if not quantile.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        melted = quantile.melt(
            id_vars="fold_id",
            value_vars=["q10_empirical_coverage", "q50_empirical_coverage", "q90_empirical_coverage"],
            var_name="quantile",
            value_name="empirical_coverage",
        )
        sns.barplot(data=melted, x="fold_id", y="empirical_coverage", hue="quantile", ax=ax)
        ax.set(title="OOF quantile calibration", ylim=(0, 1))
        path = plot_dir / "quantile_coverage.png"
        _save_figure(fig, path, dpi)
        paths.append(path)

    if not surfaces.empty:
        pairs = surfaces[["x_feature", "y_feature"]].drop_duplicates().head(6)
        fig, axes = plt.subplots(2, 3, figsize=(17, 10))
        for ax, pair in zip(axes.flat, pairs.itertuples(index=False), strict=False):
            subset = surfaces[
                surfaces["x_feature"].eq(pair.x_feature)
                & surfaces["y_feature"].eq(pair.y_feature)
            ]
            pivot = subset.pivot(index="y_bin", columns="x_bin", values="predicted_cb_return")
            sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax)
            ax.set_title(f"{pair.x_feature} x {pair.y_feature}", fontsize=9)
        path = plot_dir / "heldout_response_surfaces.png"
        _save_figure(fig, path, dpi)
        paths.append(path)

    if not ale_profiles.empty:
        features = list(ale_profiles["feature"].drop_duplicates().head(6))
        columns = 2
        rows = int(np.ceil(len(features) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(13, 4.5 * rows), squeeze=False)
        for ax, feature in zip(axes.flat, features, strict=False):
            subset = ale_profiles[ale_profiles["feature"].eq(feature)].sort_values(
                "feature_mean"
            )
            sns.lineplot(
                data=subset,
                x="feature_mean",
                y="ale_value",
                marker="o",
                ax=ax,
            )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title(f"Held-out ALE: {feature}", fontsize=10)
        for ax in axes.flat[len(features):]:
            ax.set_visible(False)
        path = plot_dir / "heldout_ale_profiles.png"
        _save_figure(fig, path, dpi)
        paths.append(path)

    validation_states = states[states.get("state_data_role", "validation").eq("validation")] if not states.empty else states
    if not validation_states.empty:
        occupancy = validation_states.groupby(["fold_id", "canonical_state_id"]).size().rename("count").reset_index()
        occupancy["occupancy"] = occupancy["count"] / occupancy.groupby("fold_id")["count"].transform("sum")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=occupancy, x="fold_id", y="occupancy", hue="canonical_state_id", ax=ax)
        ax.set(title="Causal filtered-state occupancy")
        path = plot_dir / "state_occupancy.png"
        _save_figure(fig, path, dpi)
        paths.append(path)

        duration = validation_states.groupby(["canonical_state_id", "state_semantic_label"], as_index=False)["state_expected_duration"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=duration, x="canonical_state_id", y="state_expected_duration", hue="state_semantic_label", ax=ax)
        ax.set(title="Expected state duration (bars)")
        path = plot_dir / "state_duration.png"
        _save_figure(fig, path, dpi)
        paths.append(path)

    matrices = transition_payload.get("matrices", {})
    if matrices:
        first_name, first_matrix = next(iter(matrices.items()))
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(np.asarray(first_matrix), annot=True, fmt=".2f", vmin=0, vmax=1, cmap="Blues", ax=ax)
        ax.set(title=f"Filtered HMM transition matrix: {first_name}", xlabel="To", ylabel="From")
        path = plot_dir / "state_transition_heatmap.png"
        _save_figure(fig, path, dpi)
        paths.append(path)

    if not state_outcomes.empty:
        subset = state_outcomes[state_outcomes["horizon"].eq("60m")]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.barplot(data=subset, x="canonical_state_id", y="forward_return", hue="state_semantic_label", ax=axes[0])
        sns.scatterplot(data=subset, x="mae", y="mfe", hue="state_semantic_label", size="sample_count", ax=axes[1])
        axes[0].set_title("State-conditioned 60m return")
        axes[1].set_title("State-conditioned 60m MFE / MAE")
        path = plot_dir / "state_conditioned_outcomes.png"
        _save_figure(fig, path, dpi)
        paths.append(path)

    primary = model_metrics[
        model_metrics.get("target", pd.Series(index=model_metrics.index, dtype=str)).astype(str).str.startswith("forward_cb_return")
    ] if not model_metrics.empty else pd.DataFrame()
    if not primary.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.barplot(data=primary, x="horizon", y="rmse", hue="model", ax=axes[0])
        sns.barplot(data=primary, x="fold_id", y="spearman", hue="model", ax=axes[1])
        axes[0].set_title("Model A/B/C RMSE")
        axes[1].set_title("Fold-level Spearman dispersion")
        path = plot_dir / "model_abc_comparison.png"
        _save_figure(fig, path, dpi)
        paths.append(path)

    if not forward.empty and "forward_cb_return_60m" in forward:
        bond = forward.groupby("bond_code", as_index=False)["forward_cb_return_60m"].agg(["mean", "count"]).reset_index()
        bond = bond[bond["count"] >= 20]
        if not bond.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(bond["mean"], bins=40, ax=ax)
            ax.set(title="Bond-level 60m outcome dispersion", xlabel="Mean forward return")
            path = plot_dir / "bond_level_performance_dispersion.png"
            _save_figure(fig, path, dpi)
            paths.append(path)
    return paths


def _percentage(value: float | int | None) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{100.0 * float(value):.3f}%"


def determine_stage_gate(
    surface_metrics: pd.DataFrame,
    outcome_metrics: pd.DataFrame,
    state_summary: pd.DataFrame,
    state_stability: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    official = surface_metrics[surface_metrics["official"].fillna(False)].copy()
    pivot = official.pivot_table(index="fold_id", columns="model", values="rmse")
    linear_columns = [column for column in ["ridge", "v071_fair_linear"] if column in pivot]
    if linear_columns and "lightgbm_q50" in pivot:
        strongest_linear = pivot[linear_columns].min(axis=1)
        comparable = pivot["lightgbm_q50"].notna() & strongest_linear.notna()
        surface_wins = int((pivot.loc[comparable, "lightgbm_q50"] < strongest_linear[comparable]).sum())
        surface_folds = int(comparable.sum())
    else:
        surface_wins = 0
        surface_folds = 0
    minimum_folds = 3
    surface_pass = surface_folds >= minimum_folds and surface_wins > surface_folds / 2

    primary = outcome_metrics[
        outcome_metrics["target"].str.startswith("forward_cb_return", na=False)
        & outcome_metrics["horizon"].isin(["30m", "60m"])
    ]
    outcome_pivot = primary.pivot_table(index=["fold_id", "horizon"], columns="model", values="rmse")
    if {"B_SURFACE", "C_SURFACE_STATE"}.issubset(outcome_pivot.columns):
        state_wins = int((outcome_pivot["C_SURFACE_STATE"] < outcome_pivot["B_SURFACE"]).sum())
        state_comparisons = int(outcome_pivot[["B_SURFACE", "C_SURFACE_STATE"]].dropna().shape[0])
    else:
        state_wins = 0
        state_comparisons = 0
    occupancy_ok = True
    if not state_summary.empty:
        fold_max = state_summary.groupby("fold_id")["occupancy"].max()
        fold_min = state_summary.groupby("fold_id")["occupancy"].min()
        occupancy_ok = bool((fold_max < 0.90).all() and (fold_min > 0.005).all())
    ari = state_stability.get("adjusted_rand_vs_first_seed", pd.Series(dtype=float)).dropna()
    fold_seed_rows = state_stability[state_stability.get("fold_id", pd.Series(index=state_stability.index)).notna()].copy()
    valid_seed_counts = (
        fold_seed_rows.groupby("fold_id")["seed_valid"].sum()
        if "seed_valid" in fold_seed_rows
        else pd.Series(dtype=float)
    )
    minimum_valid_seed_count = int(valid_seed_counts.min()) if not valid_seed_counts.empty else 0
    seed_stable = bool(
        not ari.empty
        and ari.median() >= 0.50
        and minimum_valid_seed_count >= 2
    )
    semantic_counts = (
        state_summary.groupby("canonical_state_id")["semantic_label"].nunique()
        if not state_summary.empty
        else pd.Series(dtype=float)
    )
    semantic_stable_fraction = float(semantic_counts.eq(1).mean()) if not semantic_counts.empty else 0.0
    semantic_stable = semantic_stable_fraction >= 0.80
    state_pass = (
        state_comparisons >= minimum_folds * 2
        and state_wins > state_comparisons / 2
        and occupancy_ok
        and seed_stable
        and semantic_stable
    )
    if surface_pass and state_pass:
        conclusion = "A_CAN_ENTER_V09"
    elif surface_pass:
        conclusion = "B_RESPONSE_SURFACE_ONLY"
    else:
        conclusion = "C_NO_STABLE_INCREMENT"
    details = {
        "surface_winning_folds": surface_wins,
        "surface_compared_folds": surface_folds,
        "surface_pass": surface_pass,
        "state_winning_comparisons": state_wins,
        "state_compared_comparisons": state_comparisons,
        "state_occupancy_pass": occupancy_ok,
        "state_seed_stability_pass": seed_stable,
        "minimum_valid_seed_count_per_fold": minimum_valid_seed_count,
        "state_semantic_stable_fraction": semantic_stable_fraction,
        "state_semantic_stability_pass": semantic_stable,
        "state_pass": state_pass,
    }
    return conclusion, details


def generate_report(config: dict[str, Any]) -> Path:
    out = output_dir(config)
    dataset_manifest = json.loads((out / "dataset_manifest.json").read_text())
    environment_manifest = json.loads((out / "environment_manifest.json").read_text())
    surface_metrics = _safe_read_csv(out / "surface_fold_metrics.csv")
    outcome_metrics = _safe_read_csv(out / "outcome_fold_metrics.csv")
    model_comparison = _safe_read_csv(out / "model_comparison.csv")
    state_summary = _safe_read_csv(out / "state_summary.csv")
    state_stability = _safe_read_csv(out / "state_stability.csv")
    quantile = _safe_read_csv(out / "quantile_calibration.csv")
    cost = _safe_read_csv(out / "cost_sensitivity.csv")
    gap = _safe_read_csv(out / "response_gap_decile_outcomes.csv")
    state_outcomes = _safe_read_csv(out / "state_conditioned_outcomes.csv")
    ale_profiles = _safe_read_csv(out / "ale_profiles.csv")
    stratified = _safe_read_csv(out / "stratified_outcomes.csv")
    transition = json.loads((out / "state_transition_matrices.json").read_text())
    conclusion, gate = determine_stage_gate(
        surface_metrics, outcome_metrics, state_summary, state_stability
    )
    official_surface = surface_metrics[surface_metrics.get("official", False).astype(bool)]
    surface_summary = official_surface.groupby("model", as_index=False).agg(
        folds=("fold_id", "nunique"), mae=("mae", "mean"), rmse=("rmse", "mean"), spearman=("spearman", "mean")
    ) if not official_surface.empty else pd.DataFrame()
    state_count = int(transition.get("selected_n_states", 0))
    dominant_occupancy = float(state_summary["occupancy"].max()) if not state_summary.empty else np.nan
    median_ari = float(state_stability["adjusted_rand_vs_first_seed"].dropna().median()) if "adjusted_rand_vs_first_seed" in state_stability and state_stability["adjusted_rand_vs_first_seed"].notna().any() else np.nan
    quantile_coverage = float(quantile["q10_q90_interval_coverage"].mean()) if not quantile.empty else np.nan

    lines = [
        "# Dynamic Response V0.8.0 Research Report",
        "",
        "## Research contract",
        "",
        f"- Strategy version: `{config['strategy']['version']}`",
        f"- Factor version: `{config['strategy']['factor_version']}`",
        f"- Config version: `{config['strategy']['config_version']}`",
        "- Research only; no production alert or order integration.",
        "- Same-bar CB return is a target, never a response-surface feature.",
        "- All response gaps are chronological OOF predictions.",
        "- HMM probabilities use causal forward filtering, never future smoothing.",
        "- Intraday rolling windows and labels do not cross lunch or overnight.",
        "",
        "## Data coverage",
        "",
        f"- Range: {dataset_manifest['start_date']} to {dataset_manifest['end_date']}",
        f"- Trading dates: {dataset_manifest['trade_dates']}",
        f"- Rows: {dataset_manifest['rows']:,}",
        f"- Bonds: {dataset_manifest['bond_count']}",
        f"- Feature count: {dataset_manifest['feature_count']}",
        f"- Universe bias: `{dataset_manifest['universe_bias_flag']}`",
        f"- Historical fundamental PIT status: `{dataset_manifest['fundamental_pit_status']}`",
        "",
        "This historical sample remains a future-selected research universe. Results are therefore diagnostic and cannot be described as an unbiased historical production-universe backtest.",
        "",
        "## Response surface",
        "",
    ]
    if not surface_summary.empty:
        lines.append(surface_summary.to_markdown(index=False, floatfmt=".8f"))
    else:
        lines.append("No surface metrics were available.")
    lines.extend(
        [
            "",
            f"Mean q10-q90 OOF interval coverage: {_percentage(quantile_coverage)}.",
            "Quantile crossing is explicitly detected and monotonically reordered; the raw crossing rate remains in `quantile_calibration.csv`.",
            "Held-out first-order ALE is computed without refitting the model and is stored in `ale_profiles.csv`.",
            "",
            "## Hidden state",
            "",
            f"- Selected states: {state_count}",
            f"- Covariance: `{transition.get('selected_covariance_type', 'NA')}`",
            f"- Maximum fold/state occupancy: {_percentage(dominant_occupancy)}",
            f"- Median cross-seed ARI: {median_ari:.4f}" if np.isfinite(median_ari) else "- Median cross-seed ARI: NA",
            f"- Daily reset: {transition.get('daily_reset')}; lunch reset: {transition.get('lunch_reset')}",
            "",
            "Semantic labels are assigned only where emission signatures are sufficiently distinctive; ambiguous states remain `STATE_K`. Future outcomes are not HMM inputs.",
            "",
            "## Model A/B/C",
            "",
        ]
    )
    if not model_comparison.empty:
        primary = model_comparison[
            model_comparison["target"].str.startswith("forward_cb_return", na=False)
        ]
        lines.append(primary.to_markdown(index=False, floatfmt=".8f"))
    else:
        lines.append("No forward outcome metrics were available.")
    lines.extend(
        [
            "",
            "Model A is the V0.7.1-style linear context, Model B adds the nonlinear OOF response surface and response gap, and Model C adds causal filtered state probabilities. All use identical official validation rows within each horizon.",
            "",
            "## Stratified diagnostics",
            "",
            f"Descriptive OOF strata: {', '.join(sorted(stratified['dimension'].dropna().astype(str).unique())) if not stratified.empty else 'unavailable'}.",
            "Buckets are evaluation-only and are not recycled into training or parameter selection. Detailed fold, bond, date, shock, premium, volatility, liquidity, time-slot, state, source, and pool results are in `stratified_outcomes.csv`.",
            "",
            "## Gap closure attribution",
            "",
            "For gap `g_t`, required closure direction is `d=-sign(g_t)`. Lagger catch-up is `d * future_cb_return`; leader reversal is `-d * beta_t * future_stock_return`. This prevents a stock reversal from being mislabeled as CB catch-up merely because absolute gap falls.",
            "",
            "## Cost proxy",
            "",
        ]
    )
    if not cost.empty:
        lines.append(cost.to_markdown(index=False, floatfmt=".8f"))
    else:
        lines.append("No cost-sensitive candidate evaluation was available.")
    lines.extend(
        [
            "",
            "Candidate returns use the next valid bar open as executable entry and the configured horizon close as exit. These are conservative proxy costs, not observed bid-ask execution. Gross and net estimates are kept separate.",
            "",
            "## Stage gate",
            "",
            f"**Decision: `{conclusion}`**",
            "",
            f"- Surface wins: {gate['surface_winning_folds']} / {gate['surface_compared_folds']} official folds.",
            f"- State increment wins: {gate['state_winning_comparisons']} / {gate['state_compared_comparisons']} 30m/60m comparisons.",
            f"- State occupancy pass: {gate['state_occupancy_pass']}.",
            f"- State seed stability pass: {gate['state_seed_stability_pass']}.",
            f"- Minimum valid HMM seeds in any official fold: {gate['minimum_valid_seed_count_per_fold']}.",
            f"- Canonical-state semantic stability: {gate['state_semantic_stable_fraction']:.1%} (pass={gate['state_semantic_stability_pass']}).",
            "",
        ]
    )
    if conclusion == "A_CAN_ENTER_V09":
        lines.extend(
            [
                "The response surface and causal state representation both pass this version's diagnostic gate. V0.9 may use these OOF artifacts as inputs while adding market latent factors and stronger overfitting diagnostics. This is not authorization to replace production alerts.",
            ]
        )
    elif conclusion == "B_RESPONSE_SURFACE_ONLY":
        lines.extend(
            [
                "The nonlinear response surface shows a majority-fold increment, but the HMM does not add stable value. Keep the response surface; treat Markov-switching regression, mixture-of-experts, or change-point methods only as later hypotheses.",
            ]
        )
    else:
        lines.extend(
            [
                "Neither layer demonstrates stable majority-fold incremental value under the present contract. The negative result should redirect work toward data frequency, label quality, alignment, and observable microstructure rather than additional model complexity.",
            ]
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The universe is future-selected and cannot support an unbiased historical deployment claim.",
            "- Five-minute bars cannot identify responses completed inside the same bar.",
            "- No true bid-ask, L2 flow, YTM, company fundamentals, credit model, or subjective clause labels enter V0.8.",
            "- Current bar CB range/close-derived premium is excluded from Stage A to avoid target leakage; only lagged CB microstructure is used.",
            "- HMM state semantics are descriptive, not causal economic proof.",
            "- No model score is connected to production alerts or orders.",
            "",
            "## Reproducibility",
            "",
            f"- Python: `{environment_manifest['python'].splitlines()[0]}`",
            f"- Seed: `{config['strategy']['random_seed']}`",
            "- Exact package versions: `environment_manifest.json`",
            "- Fold dates: `fold_definition.csv`",
            "- Feature contract: `feature_schema.json`",
        ]
    )
    path = out / "dynamic_response_v080_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "stage_gate.json").write_text(
        json.dumps({"decision": conclusion, **gate}, indent=2), encoding="utf-8"
    )
    return path
