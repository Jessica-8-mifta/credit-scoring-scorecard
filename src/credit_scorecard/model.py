"""Training-only preprocessing, supervised WoE bins and exact score scaling."""

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

TARGET = "SeriousDlqin2yrs"
UTIL = "RevolvingUtilizationOfUnsecuredLines"
INCOME = "MonthlyIncome"
LATE = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"]
FEATURES = [UTIL, "age", *LATE, "DebtRatio", INCOME, "NumberOfOpenCreditLinesAndLoans",
            "NumberRealEstateLoansOrLines", "NumberOfDependents"]
CARD_FEATURES = [UTIL, "age", INCOME, *LATE]
CAP_COLUMNS = [UTIL, "DebtRatio", INCOME]
FACTOR = 20 / np.log(2)
OFFSET = 600 - FACTOR * np.log(50)


def score_from_log_odds(log_odds_bad):
    return OFFSET - FACTOR * np.asarray(log_odds_bad)


def probability_from_score(score):
    return expit((OFFSET - np.asarray(score)) / FACTOR)


def validate_features(frame):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required predictors: {sorted(missing)}")
    result = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Predictors cannot contain infinite values")
    for col in FEATURES:
        present = result[col].dropna()
        if (present < 0).any():
            raise ValueError(f"Negative values are not supported in {col}")
        if col in LATE + ["NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines", "NumberOfDependents"]:
            if not np.allclose(present, np.round(present)):
                raise ValueError(f"Expected whole-number counts in {col}")
    return result


class TrainPreprocessor(BaseEstimator, TransformerMixin):
    """Retain the original notebook's cleaning rules and income-missing indicator."""

    def _clean(self, X):
        X = validate_features(X).copy()
        if X["age"].isna().any() or (X["age"] < 18).any():
            raise ValueError("Missing/under-18 ages must be excluded before training or scoring")
        for col in LATE:
            X[col] = X[col].where(X[col] < 90)
        X["income_missing"] = X[INCOME].isna().astype(int)
        return X

    def fit(self, X, y=None):
        cleaned = self._clean(X)
        self.medians_ = cleaned.median()
        if self.medians_.isna().any():
            raise ValueError("A training predictor has no observed values")
        self.caps_ = cleaned.fillna(self.medians_)[CAP_COLUMNS].quantile(.99)
        return self

    def transform(self, X):
        cleaned = self._clean(X).fillna(self.medians_)
        cleaned[CAP_COLUMNS] = cleaned[CAP_COLUMNS].clip(upper=self.caps_, axis="columns")
        return cleaned


class WoeBinner(BaseEstimator, TransformerMixin):
    """Six fixed features; train-derived cutpoints, 0/1/2/3+ delinquency bins."""

    def __init__(self, smoothing=.5):
        self.smoothing = smoothing

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        if set(y) != {0, 1} or len(y) != len(X):
            raise ValueError("WoE fitting requires aligned labels containing both classes")
        if self.smoothing <= 0:
            raise ValueError("WoE smoothing must be positive")
        self.edges_ = {}
        # The original notebook merged the first three utilization deciles.
        # Here their exact cutpoints are recomputed from training, never typed from a prior run.
        quantiles = {UTIL: np.arange(.3, 1, .1), "age": np.arange(.1, 1, .1),
                     INCOME: np.arange(.125, 1, .125)}
        for col, qs in quantiles.items():
            observed = X.loc[X["income_missing"] == 0, col] if col == INCOME else X[col]
            cuts = np.unique(observed.quantile(qs).to_numpy())
            self.edges_[col] = np.r_[-np.inf, cuts, np.inf]
        for col in LATE:
            self.edges_[col] = np.array([-np.inf, .5, 1.5, 2.5, np.inf])
        B = self.bin(X)
        self.maps_, audit = {}, []
        total_good, total_bad = (y == 0).sum(), (y == 1).sum()
        for col in CARD_FEATURES:
            categories = [f"bin_{i:02}" for i in range(len(self.edges_[col])-1)]
            if col == INCOME:
                categories.append("missing")
            counts = pd.crosstab(B[col].to_numpy(), y).reindex(index=categories, columns=[0, 1], fill_value=0)
            supported = counts.sum(axis=1) > 0
            k = int(supported.sum())
            mapping = {}
            for i, category in enumerate(categories):
                good, bad = int(counts.loc[category, 0]), int(counts.loc[category, 1])
                if good + bad:
                    gd = (good + self.smoothing) / (total_good + k*self.smoothing)
                    bd = (bad + self.smoothing) / (total_bad + k*self.smoothing)
                    woe, iv = np.log(gd/bd), (gd-bd)*np.log(gd/bd)
                else:
                    gd = bd = woe = iv = 0.0  # Explicit neutral fallback; no evidence for this bin.
                mapping[category] = float(woe)
                edges = self.edges_[col]
                label = "missing" if category == "missing" else f"({edges[i]:.12g}, {edges[i+1]:.12g}]"
                audit.append({"variable": col, "bin": category, "interval": label,
                              "good": good, "bad": bad, "count": good+bad,
                              "bad_rate": bad/(good+bad) if good+bad else np.nan,
                              "good_distribution": gd, "bad_distribution": bd,
                              "woe": float(woe), "iv_contribution": float(iv),
                              "training_supported": bool(good+bad),
                              "sparse_bin": good+bad < max(50, .005*len(X)) or min(good, bad) < 5})
            self.maps_[col] = mapping
        self.audit_ = pd.DataFrame(audit)
        return self

    def bin(self, X):
        B = pd.DataFrame(index=X.index)
        for col in CARD_FEATURES:
            values = X[col].round().clip(upper=3) if col in LATE else X[col]
            codes = pd.cut(values, self.edges_[col], labels=False, include_lowest=True)
            if codes.isna().any():
                raise ValueError(f"Unbinned values in {col}")
            B[col] = codes.map(lambda c: f"bin_{int(c):02}")
            if col == INCOME:
                B.loc[X["income_missing"] == 1, col] = "missing"
        return B

    def transform(self, X):
        B = self.bin(X)
        W = B.apply(lambda s: s.map(self.maps_[s.name]))
        if W.isna().any().any():
            raise ValueError("Encountered a bin missing from the fitted WoE map")
        return W.astype(float)

    def unsupported_counts(self, X):
        B = self.bin(X)
        empty = self.audit_.loc[~self.audit_.training_supported]
        return {col: int(B[col].isin(empty.loc[empty.variable == col, "bin"]).sum()) for col in CARD_FEATURES}


class Scorecard(BaseEstimator):
    def fit(self, X, y):
        self.preprocessor_ = TrainPreprocessor().fit(X)
        cleaned = self.preprocessor_.transform(X)
        self.binner_ = WoeBinner().fit(cleaned, y)
        self.logistic_ = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(self.binner_.transform(cleaned), y)
        return self

    def log_odds(self, X):
        cleaned = self.preprocessor_.transform(X)
        return self.logistic_.decision_function(self.binner_.transform(cleaned))

    def predict_proba(self, X):
        p = expit(self.log_odds(X))
        return np.column_stack([1-p, p])

    def score_points(self, X):
        return score_from_log_odds(self.log_odds(X))

    def points_table(self):
        table = self.binner_.audit_.copy()
        coefficients = dict(zip(CARD_FEATURES, self.logistic_.coef_[0]))
        table["coefficient"] = table.variable.map(coefficients)
        table["allocated_intercept"] = float(self.logistic_.intercept_[0]) / len(CARD_FEATURES)
        table["allocated_offset"] = OFFSET / len(CARD_FEATURES)
        table["points_exact"] = table.allocated_offset - FACTOR*(table.coefficient*table.woe + table.allocated_intercept)
        table["points_rounded"] = np.rint(table.points_exact).astype(int)
        return table

    def sum_bin_points(self, X, rounded=False):
        B = self.binner_.bin(self.preprocessor_.transform(X))
        table = self.points_table()
        field = "points_rounded" if rounded else "points_exact"
        contributions = B.apply(lambda s: s.map(table.loc[table.variable == s.name].set_index("bin")[field]))
        if contributions.isna().any().any():
            raise ValueError("A scorecard contribution is missing")
        return contributions.sum(axis=1).to_numpy()
