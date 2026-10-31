"""One auditable run: train-only fitting, validation policies, final test reporting."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from .evaluation import (approval_summary, band_table, calibration_table, cutoff_table, discrimination,
                         paired_bootstrap, score_band_edges, score_psi)
from .model import (CARD_FEATURES, FACTOR, FEATURES, INCOME, LATE, OFFSET, TARGET, UTIL,
                    Scorecard, TrainPreprocessor, validate_features)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(path):
    frame = pd.read_csv(path, compression="infer")
    raw_n = len(frame)
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are not supported")
    if TARGET not in frame:
        raise ValueError(f"Labeled data must contain {TARGET}")
    target = pd.to_numeric(frame[TARGET], errors="raise")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("Target must contain nonmissing binary labels 0 and 1")
    X = validate_features(frame)
    age_missing, underage = X.age.isna(), X.age < 18
    eligible = ~age_missing & ~underage
    X, y = X.loc[eligible].reset_index(drop=True), target.loc[eligible].reset_index(drop=True).astype(int)
    if y.value_counts().reindex([0, 1], fill_value=0).min() < 25:
        raise ValueError("Need at least 25 eligible observations of each class for this workflow")
    audit = {"source_file": Path(path).name, "source_sha256": file_hash(path), "rows_read": raw_n,
             "excluded_missing_age": int(age_missing.sum()), "excluded_under_18": int(underage.sum()),
             "eligible_rows": len(X), "eligible_bads": int(y.sum()), "eligible_bad_rate": float(y.mean()),
             "missing_fraction_before_imputation": {c: float(v) for c, v in X.isna().mean().items()},
             "late_count_90_or_above_replaced_with_missing": {c: int((X[c] >= 90).sum()) for c in LATE},
             "predictor_summary": X.describe().to_dict(),
             "extra_columns_ignored": [str(c) for c in frame.columns if c not in FEATURES+[TARGET]],
             "upstream_lineage": "Only this input file is verified. Earlier sampling or preprocessing cannot be inferred from its filename."}
    # describe() can have NaN for an entirely missing predictor; fail before writing JSON.
    if X.notna().sum().eq(0).any():
        raise ValueError("An input predictor has no observed values")
    return X, y, audit


def split_groups(X, y, seed):
    # Equal feature vectors stay together, even when their source row identifiers differ.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    for label in (0, 1):
        if len(np.unique(groups[np.asarray(y) == label])) < 5:
            raise ValueError("Need at least five distinct feature groups per class")
    folds = np.full(len(X), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, indices) in enumerate(splitter.split(X, y, groups)):
        folds[indices] = fold
    partitions = {"train": np.flatnonzero(folds >= 2), "validation": np.flatnonzero(folds == 0),
                  "test": np.flatnonzero(folds == 1)}
    for idx in partitions.values():
        if len(np.unique(np.asarray(y)[idx])) != 2:
            raise ValueError("A partition lacks one outcome class; cannot evaluate this split")
    return partitions, groups


def training_metadata(output, config, audit, partitions, groups, y):
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib")}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                            text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", None
    write_json(output / "run.json", {"started_utc": datetime.now(timezone.utc).isoformat(), "config": config,
               "python": platform.python_version(), "platform": platform.platform(), "versions": versions,
               "git_commit": revision, "git_dirty": dirty})
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata.get("Name"))
    (output / "environment.txt").write_text("\n".join(packages)+"\n")
    audit["split_method"] = "Five stratified feature-group folds: three train, one validation, one test (approximately 60/20/20)."
    audit["partitions"] = {name: {"rows": len(idx), "bads": int(np.asarray(y)[idx].sum()),
                                "bad_rate": float(np.asarray(y)[idx].mean()), "distinct_feature_groups": len(np.unique(groups[idx]))}
                           for name, idx in partitions.items()}
    audit["duplicate_feature_rows"] = len(groups) - len(np.unique(groups))
    write_json(output / "data_audit.json", audit)
    split_rows = [{"eligible_row_position": int(i), "partition": name} for name, indices in partitions.items() for i in indices]
    pd.DataFrame(split_rows).sort_values("eligible_row_position").to_csv(output / "split_manifest.csv", index=False)


def run_experiment(data, output, *, seed=42, trees=300, bootstrap=500, reject_share=.3, synthetic=False):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a fresh output directory; previous runs are never overwritten")
    if trees < 1 or bootstrap < 20 or not 0 <= reject_share <= .5:
        raise ValueError("Require positive trees, at least 20 bootstrap draws and reject_share in [0, .5]")
    X, y, audit = load_data(data)
    parts, groups = split_groups(X, y, seed)
    output.mkdir(parents=True)
    config = {"seed": seed, "random_forest_trees": trees, "bootstrap_repetitions": bootstrap,
              "illustrative_validation_reject_share": reject_share, "synthetic": synthetic,
              "logistic_C": 1.0, "random_forest_min_samples_leaf": 50,
              "smoothing": .5, "scorecard_features": CARD_FEATURES, "forest_features": FEATURES+["income_missing"]}
    training_metadata(output, config, audit, parts, groups, y)
    train, validation, test = (parts[n] for n in ("train", "validation", "test"))
    card = Scorecard().fit(X.iloc[train], y.iloc[train])
    preprocessor = TrainPreprocessor().fit(X.iloc[train])
    forest = RandomForestClassifier(n_estimators=trees, min_samples_leaf=50, n_jobs=-1, random_state=seed)
    forest.fit(preprocessor.transform(X.iloc[train]), y.iloc[train])
    scores = {name: card.score_points(X.iloc[idx]) for name, idx in parts.items()}
    edges = score_band_edges(scores["train"])
    # Freeze policies from validation scores before reading test outcomes for reporting.
    cutoff = -np.inf if reject_share == 0 else float(np.quantile(scores["validation"], reject_share))
    policy = {"selection_split": "validation", "selection_rule": "predeclared score quantile; not optimized for profit or bad rate",
              "requested_validation_reject_share": reject_share, "cutoff": None if np.isneginf(cutoff) else cutoff,
              "null_cutoff_means": "accept all", "tie_policy": "accept score >= cutoff",
              "validation": approval_summary(scores["validation"], y.iloc[validation], cutoff),
              "test": approval_summary(scores["test"], y.iloc[test], cutoff)}
    metrics = {"status": "computed_synthetic_demonstration" if synthetic else "computed_from_supplied_data",
               "data_sha256": audit["source_sha256"], "policy": policy, "splits": {}, "scorecard_checks": {}}
    for name in ("validation", "test"):
        idx = parts[name]
        p = card.predict_proba(X.iloc[idx])[:, 1]
        p_rf = forest.predict_proba(preprocessor.transform(X.iloc[idx]))[:, 1]
        baseline = np.full(len(idx), float(y.iloc[train].mean()))
        metrics["splits"][name] = {"scorecard": discrimination(y.iloc[idx], p),
                                   "random_forest": discrimination(y.iloc[idx], p_rf),
                                   "constant_training_prevalence": discrimination(y.iloc[idx], baseline),
                                   "score_psi_from_training": score_psi(scores["train"], scores[name], edges),
                                   "neutral_woe_bin_usage": card.binner_.unsupported_counts(card.preprocessor_.transform(X.iloc[idx]))}
        for model_name, pred in (("scorecard", p), ("random_forest", p_rf)):
            calibration_table(y.iloc[idx], pred).to_csv(output / f"{name}_{model_name}_calibration.csv", index=False)
        band_table(scores[name], y.iloc[idx], edges).to_csv(output / f"{name}_score_bands.csv", index=False)
        if name == "test":
            metrics["test_auc_uncertainty"] = paired_bootstrap(y.iloc[idx], p, p_rf, groups[idx], bootstrap, seed)
            metrics["splits"][name]["forest_minus_scorecard_auc"] = float(roc_auc(p_rf, y.iloc[idx])-roc_auc(p, y.iloc[idx]))
        exact = np.max(np.abs(card.sum_bin_points(X.iloc[idx])-scores[name]))
        rounded = np.max(np.abs(card.sum_bin_points(X.iloc[idx], rounded=True)-scores[name]))
        if exact > 1e-8 or rounded > len(CARD_FEATURES)*.5 + 1e-8:
            raise AssertionError("Scorecard points do not reconcile to model scores")
        metrics["scorecard_checks"][name] = {"max_exact_points_error": float(exact), "max_rounded_points_error": float(rounded),
                                                "rounded_error_bound": len(CARD_FEATURES)*.5}
    metrics["limitations"] = ["Random grouped split, not out-of-time validation.",
        "Feature equality is a duplicate safeguard, not verified borrower identity.",
        "The final test is not for repeated tuning; changes after inspection require a new evaluation design.",
        "No profitability optimum, production lending suitability, reject inference or subgroup fairness validation is established.",
        "Calibration is assessed; no additional calibration model is fitted.",
        "Historical results belong to a different split and manually reviewed bins; do not merge them with these results."]
    card.points_table().to_csv(output / "scorecard_points.csv", index=False)
    export_bin_audit(card, X, y, parts, output)
    cutoff_table(scores["validation"], y.iloc[validation], scores["test"], y.iloc[test]).to_csv(output / "cutoff_table.csv", index=False)
    write_json(output / "metrics.json", metrics)
    write_json(output / "score_scaling.json", {"base_score": 600, "base_good_bad_odds": 50, "pdo": 20,
               "factor": float(FACTOR), "offset": float(OFFSET), "intercept": float(card.logistic_.intercept_[0]),
               "coefficients": dict(zip(CARD_FEATURES, map(float, card.logistic_.coef_[0]))),
               "score_band_edges": [None if not np.isfinite(e) else float(e) for e in edges],
               "outer_null_edges_mean": "minus infinity and plus infinity",
               "training_medians": {k:float(v) for k,v in card.preprocessor_.medians_.items()},
               "training_caps": {k:float(v) for k,v in card.preprocessor_.caps_.items()}})
    write_json(output / "worked_example.json", worked_example(card))
    model_path = output / "model.joblib"
    joblib.dump({"scorecard": card, "forest": forest, "forest_preprocessor": preprocessor,
                 "policy_cutoff": cutoff, "config": config}, model_path)
    write_json(output / "model_manifest.json", {"filename": "model.joblib", "sha256": file_hash(model_path),
               "warning": "Only load trusted serialized model files; dependency versions are recorded in run.json."})
    from .reporting import render_report
    render_report(output)
    return output


def roc_auc(probabilities, labels):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, probabilities)


def export_bin_audit(card, X, y, parts, output):
    audit = card.binner_.audit_.copy()
    for name in ("validation", "test"):
        idx = parts[name]
        B = card.binner_.bin(card.preprocessor_.transform(X.iloc[idx]))
        records = []
        for col in CARD_FEATURES:
            table = pd.DataFrame({"bin": B[col].to_numpy(), "bad": np.asarray(y.iloc[idx])}).groupby("bin").bad.agg(["size", "sum", "mean"])
            for key, row in table.iterrows():
                records.append({"variable": col, "bin": key, f"{name}_count": int(row["size"]),
                                f"{name}_bads": int(row["sum"]), f"{name}_bad_rate": float(row["mean"])})
        audit = audit.merge(pd.DataFrame(records), on=["variable", "bin"], how="left")
    audit.to_csv(output / "bin_audit.csv", index=False)
    iv_rows = []
    for col in CARD_FEATURES:
        t = audit[audit.variable == col]
        regular = t.loc[(t.bin != "missing") & t.training_supported]
        diffs = np.diff(regular.woe)
        iv_rows.append({"variable": col, "training_iv": float(t.iv_contribution.sum()),
                        "iv_above_0_5_review_flag": bool(t.iv_contribution.sum() > .5),
                        "regular_bin_woe_monotonic": bool((diffs >= -1e-12).all() or (diffs <= 1e-12).all()),
                        "sparse_bins": int(t.sparse_bin.sum()),
                        "note": "IV flag is a diagnostic, not proof of leakage. Missing-income bin is excluded from ordered monotonicity check."})
    pd.DataFrame(iv_rows).to_csv(output / "iv_diagnostics.csv", index=False)


def worked_example(card):
    values = {UTIL: .35, "age": 40, INCOME: 5000, "DebtRatio": .3,
              "NumberOfOpenCreditLinesAndLoans": 5, "NumberRealEstateLoansOrLines": 1,
              "NumberOfDependents": 1, **{c:0 for c in LATE}}
    frame = pd.DataFrame([values])
    bins = card.binner_.bin(card.preprocessor_.transform(frame)).iloc[0]
    points = card.points_table().set_index(["variable", "bin"])
    contributions = [{"variable": c, "value": values[c], "bin": bins[c],
                      "woe": float(points.loc[(c,bins[c]), "woe"]),
                      "points_exact": float(points.loc[(c,bins[c]), "points_exact"]),
                      "points_rounded": int(points.loc[(c,bins[c]), "points_rounded"])} for c in CARD_FEATURES]
    return {"kind": "hypothetical borrower, not a real person's record", "input": values,
            "contributions": contributions, "score": float(card.score_points(frame)[0]),
            "rounded_bin_points_sum": int(card.sum_bin_points(frame, rounded=True)[0]),
            "predicted_two_year_delinquency_probability": float(card.predict_proba(frame)[0,1])}


def synthetic_data(n=2000, seed=42):
    """Engineered fixture for executable demonstrations, not a credit benchmark."""
    from scipy.special import expit
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({UTIL: rng.beta(1.2, 3, n)*1.2, "age": rng.integers(21, 85, n),
                      INCOME: rng.lognormal(8.5, .65, n), "DebtRatio": rng.lognormal(-1, 1, n),
                      "NumberOfOpenCreditLinesAndLoans": rng.poisson(6, n),
                      "NumberRealEstateLoansOrLines": rng.poisson(1, n), "NumberOfDependents": rng.poisson(1.2, n),
                      **{c:rng.poisson(.3, n) for c in LATE}})
    linear = -3.5 + 4*X[UTIL] - .018*(X.age-40) + .5*X[LATE].sum(axis=1)
    X[TARGET] = rng.binomial(1, expit(linear))
    X.loc[rng.random(n) < .2, INCOME] = np.nan
    X.loc[rng.random(n) < .03, "NumberOfDependents"] = np.nan
    return X
