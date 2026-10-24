# Credit Scoring Scorecard (WoE / Logistic Regression)

An interpretable credit scorecard that predicts serious delinquency within two years, built with Weight of Evidence (WoE) binning and logistic regression, benchmarked against a Random Forest, and converted into a points-based score with an approval cutoff analysis.

**Author:** Miftahul Jannat Jessica

---

## Objective

Build a transparent scorecard of the kind used in consumer and SME lending, evaluate it on a holdout set, and show the business trade-off between approval rate and bad rate at different score cutoffs.

## Data

- Source: Kaggle, *Give Me Some Credit* (training file; not redistributed here, download from Kaggle)
- 104,805 borrowers, 10 predictors, binary target `SeriousDlqin2yrs`
- Default (bad) rate: 6.63%
- Missing values: `MonthlyIncome` (19.8%), `NumberOfDependents` (2.6%)

## Method

1. **Split first:** 70/30 stratified train/test split (`random_state=42`). All fitted quantities (medians, caps, bin edges, WoE) come from the training set only, to avoid leakage.
2. **Cleaning:** removed ages under 18, handled 96/98 placeholder codes in late-payment counts (none present in this copy), capped `RevolvingUtilization`, `DebtRatio`, and `MonthlyIncome` at the training 99th percentile, filled missing values with training medians, and kept a missing-income flag.
3. **Binning and WoE/IV:** decile bins for continuous variables, categorical bins (0, 1, 2, 3+) for late-payment counts. Bins were checked for monotonic WoE and merged where the pattern was unstable (for example, the lowest utilization bins).
4. **Model:** logistic regression on WoE-transformed features.
5. **Benchmark:** Random Forest (300 trees, `min_samples_leaf=50`, untuned) on the raw features.
6. **Scaling:** 600 points at 50:1 good-to-bad odds, 20 points to double the odds. Higher score means lower risk.

### Information Value (training set)

| Variable | IV |
|---|---|
| Revolving utilization of unsecured lines | 1.10 |
| Times 90+ days late | 0.82 |
| Times 30-59 days past due | 0.67 |
| Times 60-89 days past due | 0.53 |
| Age | 0.28 |
| Monthly income | 0.07 |

Utilization IV is very high; it was sanity-checked and is consistent with utilization being a strong default predictor.

### Model coefficients

All six coefficients are negative, as expected with WoE defined as log(good/bad): utilization -0.65, age -0.47, income -0.35, 30-59 days past due -0.55, 60-89 days past due -0.40, 90+ days late -0.53.

## Results (holdout set)

| Model | AUC | Gini | KS |
|---|---|---|---|
| WoE logistic scorecard | 0.853 | 0.707 | 0.557 |
| Random Forest (benchmark) | 0.866 | n/a | n/a |

- Train-to-test score PSI: 0.0009. The split is random, so this is not an out-of-time validation.
- The scorecard gives up about 0.013 AUC against the Random Forest while remaining fully interpretable, which is why scorecards remain the standard in credit teams.
- Points table recomputed from bin points matches the model score within 1.91 points (rounding only).

### Score bands (holdout)

| Score band | Applicants | Bad rate |
|---|---|---|
| 411 - 540 | 3,167 | 34.95% |
| 540 - 557 | 3,128 | 11.06% |
| 557 - 571 | 3,162 | 7.81% |
| 571 - 582 | 3,145 | 4.55% |
| 582 - 592 | 3,130 | 2.14% |
| 592 - 599 | 3,137 | 1.88% |
| 599 - 604 | 3,164 | 1.26% |
| 604 - 610 | 3,171 | 0.95% |
| 610 - 618 | 3,118 | 0.77% |
| 618 - 628 | 3,120 | 0.71% |

The bad rate falls steadily across bands, so the score rank-orders risk well. The lowest-scoring decile holds about 53% of all defaults in the holdout set.

### Cutoff analysis

| Rejected | Cutoff | Approval rate | Bad rate among approved |
|---|---|---|---|
| 0% | 411 | 100% | 6.63% |
| 10% | 540 | 90% | 3.48% |
| 20% | 557 | 80% | 2.53% |
| 30% | 571 | 70% | 1.76% |
| 40% | 582 | 60% | 1.28% |
| 50% | 592 | 50% | 1.11% |

Returns diminish beyond about 30% rejection: moving from 40% to 50% rejection lowers the bad rate by only 0.17 percentage points while losing another 10% of applicants. The right cutoff depends on the lender's margin and loss assumptions.

## Dashboard

check outputs
![Dashboard](outputs/dashboard_preview.png)


## Limitations

- Random split, no out-of-time validation; PSI therefore cannot show real population drift.
- The cutoff was chosen on the holdout set; in practice a separate validation sample should be used.
- Only six variables are in the scorecard. `DebtRatio`, open credit lines, dependents, and real estate loans were left out and could be added with the same binning and monotonicity checks.
- No reject inference: the data contains only approved borrowers' outcomes.
- The data is anonymized consumer credit data, not SME lending, and the target is a two-year delinquency flag, not a regulatory probability of default.

## Repository structure

```
credit-scoring-scorecard/
├── notebooks/credit_scoring.ipynb
├── outputs/
│   ├── scorecard_points.csv
│   ├── score_bands.csv
│   ├── cutoff_table.csv
│   └── dashboard.png
├── requirements.txt
└── README.md
```

## How to run

1. Download `cs-training.csv` from the Kaggle competition page and place it in the project folder.
2. `pip install pandas numpy scikit-learn`
3. Run `notebooks/credit_scoring.ipynb` from top to bottom.
