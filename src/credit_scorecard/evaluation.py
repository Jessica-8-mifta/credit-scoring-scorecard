"""Probability metrics, conditional bootstrap uncertainty and frozen score policies."""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


def discrimination(y, p):
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    return {"auc": float(auc), "gini": float(2*auc-1), "ks": float(np.max(tpr-fpr)),
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1]))}


def paired_bootstrap(y, card, forest, groups, repetitions=500, seed=42):
    """Resample feature-duplicate clusters, with the same draw for both models."""
    y, card, forest, groups = map(np.asarray, (y, card, forest, groups))
    rng = np.random.default_rng(seed)
    order = np.argsort(groups)
    boundaries = np.flatnonzero(groups[order][1:] != groups[order][:-1]) + 1
    members = np.split(order, boundaries)
    draws = []
    for _ in range(repetitions):
        chosen = rng.integers(0, len(members), size=len(members))
        idx = order[chosen] if len(members) == len(order) else np.concatenate([members[i] for i in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        a, b = roc_auc_score(y[idx], card[idx]), roc_auc_score(y[idx], forest[idx])
        draws.append((a, b, b-a))
    if len(draws) < max(10, repetitions*.8):
        raise ValueError("Too few valid bootstrap draws; increase sample size or class support")
    values = np.asarray(draws)
    return {"method": "paired feature-duplicate cluster percentile bootstrap",
            "requested_repetitions": repetitions, "valid_repetitions": len(draws), "seed": seed,
            "confidence_level": .95,
            **{key: [float(v) for v in np.quantile(values[:, i], [.025, .975])]
               for i, key in enumerate(("scorecard_auc_interval", "forest_auc_interval", "forest_minus_scorecard_auc_interval"))},
            "note": "Conditional on these fitted models and this split; does not include retraining uncertainty. Feature grouping is not verified borrower identity or temporal independence."}


def calibration_table(y, p, bins=10):
    frame = pd.DataFrame({"actual": np.asarray(y), "probability": np.asarray(p)})
    frame["probability_bin"] = pd.cut(frame.probability, np.linspace(0, 1, bins+1), include_lowest=True)
    return frame.groupby("probability_bin", observed=True).agg(
        applicants=("actual", "size"), bads=("actual", "sum"),
        observed_bad_rate=("actual", "mean"), mean_predicted_probability=("probability", "mean")).reset_index()


def approval_summary(scores, y, cutoff):
    accepted = np.asarray(scores) >= cutoff  # Keep every tied score at the boundary.
    y = np.asarray(y)
    return {"applicants": len(y), "approved": int(accepted.sum()),
            "rejected": int((~accepted).sum()), "approval_rate": float(accepted.mean()),
            "realized_reject_share": float((~accepted).mean()),
            "approved_bads": int(y[accepted].sum()), "rejected_bads": int(y[~accepted].sum()),
            "rejected_goods": int((1-y[~accepted]).sum()),
            "approved_bad_rate": float(y[accepted].mean()) if accepted.any() else None}


def cutoff_table(validation_scores, validation_y, test_scores, test_y, shares=None):
    shares = np.arange(0, .55, .05) if shares is None else shares
    rows = []
    for share in shares:
        # The zero-rejection policy must also admit test scores below validation's minimum.
        cutoff = -np.inf if share == 0 else float(np.quantile(validation_scores, share))
        row = {"requested_validation_reject_share": float(share), "cutoff": cutoff,
               "tie_policy": "accept score >= cutoff"}
        for prefix, scores, y in (("validation", validation_scores, validation_y), ("test", test_scores, test_y)):
            row.update({f"{prefix}_{k}": v for k, v in approval_summary(scores, y, cutoff).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def score_band_edges(training_scores):
    cuts = np.unique(np.quantile(training_scores, np.arange(.1, 1, .1)))
    return np.r_[-np.inf, cuts, np.inf]


def band_table(scores, y, edges):
    frame = pd.DataFrame({"score": np.asarray(scores), "bad": np.asarray(y)})
    frame["band"] = pd.cut(frame.score, edges, include_lowest=True)
    return frame.groupby("band", observed=True).agg(applicants=("bad", "size"), bads=("bad", "sum"),
        bad_rate=("bad", "mean"), avg_score=("score", "mean")).reset_index()


def score_psi(reference, current, edges):
    a = np.histogram(reference, edges)[0].astype(float) + .5
    b = np.histogram(current, edges)[0].astype(float) + .5
    a, b = a/a.sum(), b/b.sum()
    return float(np.sum((b-a)*np.log(b/a)))
