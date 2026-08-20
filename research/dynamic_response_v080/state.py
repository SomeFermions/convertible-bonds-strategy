from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from .config import output_dir
from .data import WalkForwardFold
from .models import deterministic_row_sample


LOGGER = logging.getLogger(__name__)
EPS = 1e-12

HMM_FEATURES = [
    "response_gap",
    "response_gap_z",
    "response_gap_change",
    "stock_idio_return_current",
    "surface_target_cb_return",
    "prediction_lightgbm_q50",
    "prediction_interval_width",
    "stock_realized_vol_6bar",
    "relative_amount_lag1",
    "premium_slot_residual_lag1",
    "high_low_range_z_lag1",
    "bar_slot_numeric",
]


@dataclass
class CausalHMMModel:
    model: GaussianHMM
    imputer: SimpleImputer
    scaler: StandardScaler
    feature_columns: list[str]
    canonical_map: dict[int, int]
    semantic_labels: dict[int, str]
    fold_id: str
    train_end: pd.Timestamp
    model_version: str


def _sequence_keys(config: dict[str, Any]) -> list[str]:
    keys = ["bond_code", "trade_date"]
    if config["hmm"].get("default_lunch_reset", True):
        keys.append("session_part")
    return keys


def _ordered_sequences(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[int]]:
    keys = _sequence_keys(config)
    ordered = frame.sort_values(keys + ["bar_start"]).copy()
    lengths = ordered.groupby(keys, sort=False, observed=True).size().astype(int).tolist()
    return ordered, lengths


def _sample_complete_sequences(
    frame: pd.DataFrame,
    cap: int,
    seed: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame
    keys = _sequence_keys(config)
    sequence_table = frame[keys].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sequence_table))
    selected: list[pd.DataFrame] = []
    count = 0
    indexed = frame.set_index(keys, drop=False)
    for position in order:
        key_values = tuple(sequence_table.iloc[position][key] for key in keys)
        key = key_values if len(keys) > 1 else key_values[0]
        sequence = indexed.loc[key]
        if isinstance(sequence, pd.Series):
            sequence = sequence.to_frame().T
        if count + len(sequence) > cap and selected:
            continue
        selected.append(sequence)
        count += len(sequence)
        if count >= cap:
            break
    if not selected:
        return deterministic_row_sample(frame, cap, seed)
    return pd.concat(selected, ignore_index=True)


def _fit_transformer(
    frame: pd.DataFrame, feature_columns: list[str]
) -> tuple[SimpleImputer, StandardScaler, np.ndarray]:
    imputer = SimpleImputer(strategy="median", add_indicator=False)
    imputed = imputer.fit_transform(frame[feature_columns])
    scaler = StandardScaler()
    matrix = scaler.fit_transform(imputed)
    return imputer, scaler, np.asarray(matrix, dtype=float)


def transform_hmm_features(model: CausalHMMModel, frame: pd.DataFrame) -> np.ndarray:
    imputed = model.imputer.transform(frame[model.feature_columns])
    return np.asarray(model.scaler.transform(imputed), dtype=float)


def _emission_log_probability(model: GaussianHMM, matrix: np.ndarray) -> np.ndarray:
    """Evaluate Gaussian emissions without invoking forward-backward smoothing."""
    means = np.asarray(model.means_, dtype=float)
    covariance_type = model.covariance_type
    n_samples = len(matrix)
    n_states, n_features = means.shape
    result = np.empty((n_samples, n_states), dtype=float)
    if covariance_type == "diag":
        # hmmlearn expands the public covars_ property to full matrices. The
        # fitted private representation retains the diagonal variances.
        variances = np.maximum(np.asarray(model._covars_, dtype=float), 1e-8)
        for state in range(n_states):
            diff = matrix - means[state]
            result[:, state] = -0.5 * (
                n_features * np.log(2.0 * np.pi)
                + np.log(variances[state]).sum()
                + (np.square(diff) / variances[state]).sum(axis=1)
            )
        return result
    covariances = np.asarray(model.covars_, dtype=float)
    for state in range(n_states):
        covariance = covariances[state] + np.eye(n_features) * 1e-8
        sign, log_det = np.linalg.slogdet(covariance)
        if sign <= 0:
            covariance = covariance + np.eye(n_features) * 1e-5
            _, log_det = np.linalg.slogdet(covariance)
        inverse = np.linalg.pinv(covariance)
        diff = matrix - means[state]
        mahalanobis = np.einsum("ij,jk,ik->i", diff, inverse, diff)
        result[:, state] = -0.5 * (
            n_features * np.log(2.0 * np.pi) + log_det + mahalanobis
        )
    return result


def causal_forward_filter(
    model: GaussianHMM,
    matrix: np.ndarray,
    lengths: list[int],
) -> np.ndarray:
    """Return P(S_t | X_1,...,X_t) with explicit sequence resets.

    No backward recursion is performed. Consequently, changing X after time t
    cannot change the returned probability at time t.
    """
    if sum(lengths) != len(matrix):
        raise ValueError("HMM sequence lengths do not sum to matrix rows")
    emission = _emission_log_probability(model, matrix)
    log_start = np.log(np.maximum(np.asarray(model.startprob_, dtype=float), EPS))
    log_transition = np.log(np.maximum(np.asarray(model.transmat_, dtype=float), EPS))
    probabilities = np.empty_like(emission)
    offset = 0
    for length in lengths:
        if length <= 0:
            continue
        first = log_start + emission[offset]
        first -= logsumexp(first)
        probabilities[offset] = np.exp(first)
        previous = first
        for row in range(offset + 1, offset + length):
            current = emission[row] + logsumexp(previous[:, None] + log_transition, axis=0)
            current -= logsumexp(current)
            probabilities[row] = np.exp(current)
            previous = current
        offset += length
    return probabilities


def _parameter_count(states: int, features: int, covariance_type: str) -> int:
    start = states - 1
    transitions = states * (states - 1)
    means = states * features
    covariance = states * features if covariance_type == "diag" else states * features * (features + 1) // 2
    return start + transitions + means + covariance


def _candidate_diagnostics(
    model: GaussianHMM,
    matrix: np.ndarray,
    lengths: list[int],
    seed: int,
) -> dict[str, Any]:
    log_likelihood = float(model.score(matrix, lengths))
    probabilities = causal_forward_filter(model, matrix, lengths)
    occupancy = probabilities.mean(axis=0)
    states = int(model.n_components)
    parameters = _parameter_count(states, matrix.shape[1], model.covariance_type)
    bic = -2.0 * log_likelihood + parameters * np.log(max(len(matrix), 2))
    history = np.asarray(list(model.monitor_.history), dtype=float)
    last_delta = float(history[-1] - history[-2]) if len(history) >= 2 else np.nan
    negative_tolerance = max(float(model.tol), 1e-6)
    strict_convergence = bool(
        model.monitor_.converged
        and np.isfinite(last_delta)
        and last_delta >= -negative_tolerance
    )
    return {
        "n_states": states,
        "covariance_type": model.covariance_type,
        "seed": seed,
        "log_likelihood": log_likelihood,
        "bic": float(bic),
        "minimum_occupancy": float(occupancy.min()),
        "maximum_occupancy": float(occupancy.max()),
        "mean_persistence": float(np.diag(model.transmat_).mean()),
        "library_converged": bool(model.monitor_.converged),
        "converged": strict_convergence,
        "last_likelihood_delta": last_delta,
        "iterations": int(model.monitor_.iter),
        "occupancy": occupancy.tolist(),
    }


def select_hmm_specification(
    training: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[int, str, int, pd.DataFrame]:
    """Select HMM structure on the earliest training period only."""
    features = [column for column in HMM_FEATURES if column in training.columns]
    seed0 = int(config["strategy"]["random_seed"])
    sampled = _sample_complete_sequences(
        training,
        int(config["hmm"]["candidate_row_cap"]),
        seed0,
        config,
    )
    ordered, lengths = _ordered_sequences(sampled, config)
    _, _, matrix = _fit_transformer(ordered, features)
    rows: list[dict[str, Any]] = []
    fitted: dict[tuple[int, str, int], GaussianHMM] = {}
    # Structure selection uses one fixed seed. Cross-seed stability is measured
    # after selecting a specification, on every official fold.
    selection_seed = int(config["hmm"]["candidate_seeds"][0])
    for states in config["hmm"]["candidate_states"]:
        for covariance_type in config["hmm"]["covariance_types"]:
            for seed in [selection_seed]:
                model = GaussianHMM(
                    n_components=int(states),
                    covariance_type=str(covariance_type),
                    n_iter=int(config["hmm"]["n_iter"]),
                    tol=float(config["hmm"]["tolerance"]),
                    random_state=int(seed),
                    min_covar=1e-5,
                )
                try:
                    model.fit(matrix, lengths)
                    diagnostics = _candidate_diagnostics(model, matrix, lengths, int(seed))
                    rows.append(diagnostics)
                    fitted[(int(states), str(covariance_type), int(seed))] = model
                    LOGGER.info(
                        "HMM candidate states=%s covariance=%s converged=%s bic=%.2f occupancy=[%.3f, %.3f]",
                        states,
                        covariance_type,
                        diagnostics["converged"],
                        diagnostics["bic"],
                        diagnostics["minimum_occupancy"],
                        diagnostics["maximum_occupancy"],
                    )
                except Exception as error:  # HMM candidates may be singular.
                    rows.append(
                        {
                            "n_states": int(states),
                            "covariance_type": str(covariance_type),
                            "seed": int(seed),
                            "log_likelihood": np.nan,
                            "bic": np.inf,
                            "minimum_occupancy": 0.0,
                            "maximum_occupancy": 1.0,
                            "mean_persistence": np.nan,
                            "converged": False,
                            "iterations": 0,
                            "error": str(error),
                        }
                    )
    diagnostics_frame = pd.DataFrame(rows)
    eligible = diagnostics_frame[
        diagnostics_frame["converged"].fillna(False)
        & diagnostics_frame["minimum_occupancy"].ge(float(config["hmm"]["minimum_state_occupancy"]))
        & diagnostics_frame["maximum_occupancy"].le(float(config["hmm"]["maximum_dominant_occupancy"]))
    ].copy()
    if eligible.empty:
        raise RuntimeError(
            "Every Gaussian HMM candidate failed strict convergence or occupancy checks"
        )
    # Median BIC across seeds rewards structures that are not one-seed accidents.
    grouped = eligible.groupby(["n_states", "covariance_type"], as_index=False).agg(
        median_bic=("bic", "median"),
        median_minimum_occupancy=("minimum_occupancy", "median"),
        median_maximum_occupancy=("maximum_occupancy", "median"),
        converged_seeds=("converged", "sum"),
    )
    best_spec = grouped.sort_values(["median_bic", "n_states"]).iloc[0]
    states = int(best_spec["n_states"])
    covariance_type = str(best_spec["covariance_type"])
    seed_rows = eligible[
        eligible["n_states"].eq(states) & eligible["covariance_type"].eq(covariance_type)
    ]
    best_seed = int(seed_rows.sort_values("bic").iloc[0]["seed"])
    diagnostics_frame["selected_specification"] = (
        diagnostics_frame["n_states"].eq(states)
        & diagnostics_frame["covariance_type"].eq(covariance_type)
    )
    diagnostics_frame["selected_seed"] = diagnostics_frame["seed"].eq(best_seed) & diagnostics_frame["selected_specification"]
    return states, covariance_type, best_seed, diagnostics_frame


def _fit_hmm(
    training: pd.DataFrame,
    states: int,
    covariance_type: str,
    seed: int,
    fold: WalkForwardFold,
    config: dict[str, Any],
) -> CausalHMMModel:
    features = [column for column in HMM_FEATURES if column in training.columns]
    sampled = _sample_complete_sequences(
        training,
        int(config["hmm"]["training_row_cap"]),
        seed,
        config,
    )
    ordered, lengths = _ordered_sequences(sampled, config)
    imputer, scaler, matrix = _fit_transformer(ordered, features)
    model = GaussianHMM(
        n_components=states,
        covariance_type=covariance_type,
        n_iter=int(config["hmm"]["n_iter"]),
        tol=float(config["hmm"]["tolerance"]),
        random_state=seed,
        min_covar=1e-5,
    )
    model.fit(matrix, lengths)
    return CausalHMMModel(
        model=model,
        imputer=imputer,
        scaler=scaler,
        feature_columns=features,
        canonical_map={state: state for state in range(states)},
        semantic_labels={state: f"STATE_{state}" for state in range(states)},
        fold_id=fold.fold_id,
        train_end=fold.train_end,
        model_version=f"hmm_{states}_{covariance_type}_{config['strategy']['config_version']}",
    )


def align_states(
    reference_centers: np.ndarray,
    candidate_centers: np.ndarray,
) -> dict[int, int]:
    """Map candidate state IDs to canonical reference IDs by center distance."""
    if reference_centers.shape != candidate_centers.shape:
        raise ValueError("State centers must have identical shape for alignment")
    distances = np.linalg.norm(
        candidate_centers[:, None, :] - reference_centers[None, :, :], axis=2
    )
    candidate_ids, reference_ids = linear_sum_assignment(distances)
    return {int(candidate): int(reference) for candidate, reference in zip(candidate_ids, reference_ids, strict=True)}


def _canonical_probabilities(
    probabilities: np.ndarray, mapping: dict[int, int]
) -> np.ndarray:
    result = np.zeros_like(probabilities)
    for source, target in mapping.items():
        result[:, target] = probabilities[:, source]
    return result


def _canonical_transition(model: GaussianHMM, mapping: dict[int, int]) -> np.ndarray:
    states = model.n_components
    result = np.zeros((states, states), dtype=float)
    for source_i, target_i in mapping.items():
        for source_j, target_j in mapping.items():
            result[target_i, target_j] = model.transmat_[source_i, source_j]
    return result


def _semantic_labels(
    model: CausalHMMModel,
    training: pd.DataFrame,
    probabilities: np.ndarray,
) -> dict[int, str]:
    """Assign conservative post-hoc labels; ambiguous states retain STATE_K."""
    states = probabilities.argmax(axis=1)
    diagnostic = training.copy()
    diagnostic["_state"] = states
    summary = diagnostic.groupby("_state", observed=True).agg(
        response_gap_z=("response_gap_z", "mean"),
        gap_change=("response_gap_change", "mean"),
        stock_shock=("stock_idio_return_current", "mean"),
        cb_return=("surface_target_cb_return", "mean"),
        interval_width=("prediction_interval_width", "mean"),
        noise=("high_low_range_z_lag1", "mean"),
    )
    labels: dict[int, str] = {state: f"STATE_{state}" for state in range(model.model.n_components)}
    used: set[str] = set()
    for state, row in summary.iterrows():
        candidate = None
        if row["noise"] > 0.7 or row["interval_width"] > summary["interval_width"].quantile(0.75):
            candidate = "LIQUIDITY_NOISE"
        elif row["stock_shock"] > 0 and row["response_gap_z"] < -0.35:
            candidate = "UNDERREACTION"
        elif row["response_gap_z"] < 0 and row["gap_change"] > 0 and row["cb_return"] > 0:
            candidate = "CATCHUP"
        elif abs(row["response_gap_z"]) < 0.25 and row["interval_width"] <= summary["interval_width"].median():
            candidate = "CO_MOVE"
        elif abs(row["response_gap_z"]) > 0.8 and np.sign(row["gap_change"]) == np.sign(row["response_gap_z"]):
            candidate = "BREAKDOWN"
        if candidate and candidate not in used:
            labels[int(state)] = candidate
            used.add(candidate)
    return labels


def filter_state_frame(
    fitted: CausalHMMModel,
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    ordered, lengths = _ordered_sequences(frame, config)
    matrix = transform_hmm_features(fitted, ordered)
    raw_probabilities = causal_forward_filter(fitted.model, matrix, lengths)
    probabilities = _canonical_probabilities(raw_probabilities, fitted.canonical_map)
    transition = _canonical_transition(fitted.model, fitted.canonical_map)
    state = probabilities.argmax(axis=1)
    entropy = -(probabilities * np.log(np.maximum(probabilities, EPS))).sum(axis=1)
    expected_duration = 1.0 / np.maximum(1.0 - np.diag(transition), 1e-6)
    result = ordered[
        ["trade_date", "bar_start", "bar_slot", "session_part", "bond_code", "stock_code", "fold_id"]
    ].copy()
    for index in range(probabilities.shape[1]):
        result[f"state_probability_{index}"] = probabilities[:, index].astype("float32")
    result["canonical_state_id"] = state.astype("int16")
    result["state_semantic_label"] = [fitted.semantic_labels.get(int(value), f"STATE_{value}") for value in state]
    result["state_argmax_probability"] = probabilities.max(axis=1).astype("float32")
    result["state_entropy"] = entropy.astype("float32")
    result["state_expected_duration"] = expected_duration[state].astype("float32")
    keys = _sequence_keys(config)
    result["previous_state"] = result.groupby(keys, sort=False)["canonical_state_id"].shift(1).astype("Int16")
    result["most_likely_transition"] = [
        f"{int(value)}->{int(np.argmax(transition[int(value)]))}" for value in state
    ]
    result["hmm_model_version"] = fitted.model_version
    result["hmm_train_end_date"] = fitted.train_end
    return result.sort_index(), transition


def fit_predict_state_folds(
    oof_frame: pd.DataFrame,
    folds: list[WalkForwardFold],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit HMMs on earlier OOF gaps and causally filter each official fold."""
    official_folds = [fold for fold in folds if fold.official]
    if not official_folds:
        raise ValueError("HMM requires at least one official walk-forward fold")
    valid_oof = oof_frame[oof_frame["response_gap"].notna()].copy()
    first_fold = official_folds[0]
    first_training = valid_oof[valid_oof["trade_date"] <= first_fold.train_end].copy()
    if first_training.empty:
        raise ValueError("No earlier causal OOF gaps are available to train the HMM")
    states, covariance, selected_seed, candidate_diagnostics = select_hmm_specification(
        first_training, config
    )
    all_probabilities: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    transitions: dict[str, Any] = {}
    selected_fold_seeds: dict[str, int] = {}
    reference_centers: np.ndarray | None = None
    artifact_dir = output_dir(config) / "models" / "hmm"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for fold_index, fold in enumerate(official_folds):
        training = valid_oof[valid_oof["trade_date"] <= fold.train_end].copy()
        validation = valid_oof[
            valid_oof["trade_date"].between(fold.validation_start, fold.validation_end)
        ].copy()
        if training.empty or validation.empty:
            continue
        seed_models: list[CausalHMMModel] = []
        seed_assignments: list[np.ndarray] = []
        seed_diagnostics: list[dict[str, Any]] = []
        successful_seeds: list[int] = []
        for seed in config["hmm"]["candidate_seeds"]:
            try:
                fitted_seed = _fit_hmm(training, states, covariance, int(seed), fold, config)
                ordered_training, lengths = _ordered_sequences(training, config)
                matrix = transform_hmm_features(fitted_seed, ordered_training)
                diagnostics = _candidate_diagnostics(
                    fitted_seed.model, matrix, lengths, int(seed)
                )
                raw_probabilities = causal_forward_filter(
                    fitted_seed.model, matrix, lengths
                )
                seed_models.append(fitted_seed)
                seed_assignments.append(raw_probabilities.argmax(axis=1))
                seed_diagnostics.append(diagnostics)
                successful_seeds.append(int(seed))
            except Exception as error:
                stability_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "seed": int(seed),
                        "adjusted_rand_vs_first_seed": np.nan,
                        "n_states": states,
                        "covariance_type": covariance,
                        "seed_valid": False,
                        "error": str(error),
                    }
                )
                LOGGER.warning(
                    "HMM seed rejected fold=%s seed=%s error=%s",
                    fold.fold_id,
                    seed,
                    error,
                )
        if not seed_models:
            raise RuntimeError(f"Every HMM seed failed for fold {fold.fold_id}")
        reference_assignment = seed_assignments[0]
        for seed_index, assignment in enumerate(seed_assignments):
            stability_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "seed": successful_seeds[seed_index],
                    "adjusted_rand_vs_first_seed": float(adjusted_rand_score(reference_assignment, assignment)),
                    "n_states": states,
                    "covariance_type": covariance,
                    "seed_valid": True,
                }
            )
        eligible_seed_indexes = [
            index
            for index, diagnostics in enumerate(seed_diagnostics)
            if diagnostics["converged"]
            and diagnostics["minimum_occupancy"] >= float(config["hmm"]["minimum_state_occupancy"])
            and diagnostics["maximum_occupancy"] <= float(config["hmm"]["maximum_dominant_occupancy"])
        ]
        if not eligible_seed_indexes:
            raise RuntimeError(
                f"No stable {states}-state {covariance} HMM seed for fold {fold.fold_id}"
            )
        selected_index = min(
            eligible_seed_indexes, key=lambda index: seed_diagnostics[index]["bic"]
        )
        fitted = seed_models[selected_index]
        selected_fold_seeds[fold.fold_id] = successful_seeds[selected_index]
        centers = fitted.model.means_.copy()
        if reference_centers is None:
            reference_centers = centers
            fitted.canonical_map = {state: state for state in range(states)}
        else:
            fitted.canonical_map = align_states(reference_centers, centers)
        ordered_training, train_lengths = _ordered_sequences(training, config)
        train_probabilities_raw = causal_forward_filter(
            fitted.model, transform_hmm_features(fitted, ordered_training), train_lengths
        )
        train_probabilities = _canonical_probabilities(train_probabilities_raw, fitted.canonical_map)
        fitted.semantic_labels = _semantic_labels(fitted, ordered_training, train_probabilities)
        training_for_filter = training.copy()
        training_for_filter["fold_id"] = fold.fold_id
        training_probability_frame, _ = filter_state_frame(
            fitted, training_for_filter, config
        )
        training_probability_frame["state_data_role"] = "train"
        validation = validation.copy()
        validation["fold_id"] = fold.fold_id
        probability_frame, transition = filter_state_frame(fitted, validation, config)
        probability_frame["state_data_role"] = "validation"
        all_probabilities.extend([training_probability_frame, probability_frame])
        transitions[fold.fold_id] = transition.tolist()
        occupancy = probability_frame["canonical_state_id"].value_counts(normalize=True)
        for state in range(states):
            subset = probability_frame[probability_frame["canonical_state_id"].eq(state)]
            summary_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "canonical_state_id": state,
                    "semantic_label": fitted.semantic_labels.get(state, f"STATE_{state}"),
                    "occupancy": float(occupancy.get(state, 0.0)),
                    "sample_count": int(len(subset)),
                    "expected_duration": float(1.0 / max(1.0 - transition[state, state], 1e-6)),
                    "transition_persistence": float(transition[state, state]),
                    "train_end": str(fold.train_end.date()),
                }
            )
        joblib.dump(fitted, artifact_dir / f"{fold.fold_id}.joblib", compress=3)
        LOGGER.info(
            "HMM fold %s complete states=%d covariance=%s validation_rows=%d",
            fold.fold_id,
            states,
            covariance,
            len(validation),
        )
    if not all_probabilities:
        raise RuntimeError("No HMM state probabilities were generated")
    payload = {
        "selected_n_states": states,
        "selected_covariance_type": covariance,
        "selected_seed": selected_seed,
        "selected_seed_by_fold": selected_fold_seeds,
        "selection_period_end": str(first_fold.train_end.date()),
        "daily_reset": bool(config["hmm"]["daily_reset"]),
        "lunch_reset": bool(config["hmm"]["default_lunch_reset"]),
        "filtered_not_smoothed": True,
        "matrices": transitions,
    }
    return (
        pd.concat(all_probabilities, ignore_index=True),
        pd.DataFrame(summary_rows),
        pd.concat([candidate_diagnostics, pd.DataFrame(stability_rows)], ignore_index=True, sort=False),
        payload,
    )


def write_transition_payload(payload: dict[str, Any], config: dict[str, Any]) -> Path:
    path = output_dir(config) / "state_transition_matrices.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
