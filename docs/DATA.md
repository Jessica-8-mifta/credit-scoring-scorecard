# Data requirements and lineage

The supplied original notebook loaded `train.csv.zip` with `index_col=0`; its saved output showed 104,805 rows and 11 non-index columns. It reported minimum age 21, a 6.631% target rate, approximately 19.8% missing income and 2.6% missing dependents. The original source ZIP has not been provided. Its earlier sampling, transformations and relationship to Kaggle's original training file cannot be recovered from aggregate outputs.

The revised loader uses named columns, never a positional assumption that the first column is an index. It accepts plain CSV or a ZIP containing one CSV. Supply `SeriousDlqin2yrs` as 0/1 with no missing target values and these ten predictors:

```text
RevolvingUtilizationOfUnsecuredLines
age
NumberOfTime30-59DaysPastDueNotWorse
NumberOfTime60-89DaysPastDueNotWorse
NumberOfTimes90DaysLate
DebtRatio
MonthlyIncome
NumberOfOpenCreditLinesAndLoans
NumberRealEstateLoansOrLines
NumberOfDependents
```

Predictors must be numeric, nonnegative and finite where observed. Count predictors must be whole numbers. Missing predictors are permitted; missing or under-18 ages are excluded from labeled training input and refused during scoring. Counts of 90 or more in delinquency histories are treated as special/missing values, retaining the original notebook's rule; the number affected is recorded. Investigate such codes against the data dictionary before using a different dataset. Negative or non-numeric values fail instead of being silently repaired.

Training requires at least 25 observations of each outcome class and five distinct feature groups per class. This is an execution guard, not a claim that such a small dataset is statistically sufficient. An all-missing predictor fails explicitly.

Each real run records the file checksum and the exact input/exclusion/eligible counts. The partition manifest indexes eligible rows after exclusions, not original file line numbers. The exact installed package versions and split seed are retained. Hash grouping keeps identical raw feature vectors together; different records that become equal after imputation are not thereby established to be the same borrower. If actual borrower IDs or dates become available, revise the grouping/time design before treating results as independent deployment evidence.

Keep source data and generated models local. Use only data you are authorized to process and follow the competition's terms. The synthetic generator is an explicit software fixture with an engineered outcome relation; its distributions and scores must not be represented as observed consumer-credit performance.
