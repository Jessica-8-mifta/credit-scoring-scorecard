# Methodology and interpretation

## Original versus revised work

The original notebook uses a 70/30 stratified split, training-only median imputation/capping/WoE, a six-variable logistic regression and a 300-tree Random Forest. It explores score cutoffs and score deciles on the holdout. The original fitted object, full-precision coefficients and points CSV were not retained. Rounded notebook output is preserved as evidence of the historical run, not used to invent full-precision fitted parameters.

The revised run uses approximately 60/20/20 train/validation/test partitions through five stratified feature-group folds. The utilization/age/income cutpoints are derived from training values for that run. The original design of pooling the lowest three utilization deciles is predeclared; other nonmonotonic patterns are flagged rather than manually changed after reading test outcomes. Income missingness has a separate bin. Logistic regularization is fixed (`C=1`); no claim of optimal hyperparameters is made.

For supported training bin j, good and bad distributions are `(good_j + 0.5)/(total_good + 0.5*K)` and `(bad_j + 0.5)/(total_bad + 0.5*K)`, with K supported bins for that variable. WoE is the log of their ratio; IV sums `(good_distribution - bad_distribution) * WoE`. Empty training bins have neutral WoE zero, no IV contribution and explicit usage counts. Such a fallback has no supporting risk evidence and should prompt investigation when used. Minimum event counts, sparse bins, monotonicity and high-IV flags are diagnostics, not automatic model approval criteria.

## Points and score probabilities

With factor `20/ln(2)` and offset `600 - factor*ln(50)`, score equals `offset - factor*logit(p_bad)`. Higher points imply lower modeled delinquency risk. Allocating intercept and offset across six feature contributions yields an exactly additive points table. Rounding each contribution separately can change the sum by at most 3 points. Policy membership uses exact model scores, never rounded-display scores. The hypothetical worked example contains no actual person's features.

## Thresholds and business interpretation

Requested validation rejection quantiles define numeric cutoffs. All tied scores at the cutoff are accepted; actual rejection fractions differ from nominal quantiles. The accept-all policy uses negative infinity so lower test scores are still accepted. Outcomes are reported for validation and test, but test outcomes do not select any cutoff. The default 30% illustration preserves the original project's discussion point; it is not an inferred business preference or an optimal policy.

Observed accepted bad rate alone cannot determine profitability. That analysis would need justified revenue, exposure, loss-given-default, operating-cost and constraint assumptions. The current project does not supply those inputs or a real lending decision. Score rank separation also does not establish probability calibration.

## Metrics, calibration and uncertainty

Report AUC, Gini (=2*AUC−1), KS, Brier score and log loss for both models and a constant training-prevalence baseline. Fixed probability intervals produce calibration tables with sample counts, observed rates and mean predictions. Empty calibration bins are omitted, not filled with zeros. Calibration assessment does not automatically recalibrate the predictions.

Bootstrap uncertainty uses paired draws for both model predictions. Feature-duplicate groups are sampled with replacement, retaining all members of each selected group. Percentile intervals describe test-sample uncertainty conditional on the fitted models and data split; they do not include variability from retraining, hyperparameter selection, or population/time shift. No statistical superiority claim should follow from a point AUC difference alone.

Score-band boundaries are learned from training scores, then reused on validation and test. PSI uses those fixed boundaries and 0.5 count smoothing. A low PSI between random partitions does not demonstrate out-of-time stability. Existing historical holdout bands were instead formed from that holdout's own score quantiles; do not combine their thresholds with revised-run metrics.

## Retain and review

Keep data checksums, all split assignments, model and version records, bin and points exports, test metrics and calibration tables together. Do not repeatedly adjust the system after inspecting test results and continue describing that set as untouched. Investigate sparse and unsupported bins, coefficient signs, shifts in event rates and high-error records. Additional temporal, borrower-level, domain, fairness and economic validation would be needed for a separate deployment project.
