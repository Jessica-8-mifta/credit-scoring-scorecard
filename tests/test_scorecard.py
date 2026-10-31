import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score

from credit_scorecard.cli import main
from credit_scorecard.evaluation import (approval_summary, cutoff_table, discrimination,
                                         paired_bootstrap, score_band_edges, score_psi)
from credit_scorecard.model import (CARD_FEATURES, INCOME, LATE, TARGET, UTIL, Scorecard,
                                    TrainPreprocessor, probability_from_score, score_from_log_odds)
from credit_scorecard.pipeline import load_data, run_experiment, split_groups, synthetic_data


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame = synthetic_data(1200, seed=16)
        cls.X, cls.y = frame.drop(columns=TARGET), frame[TARGET]
        cls.card = Scorecard().fit(cls.X.iloc[:900], cls.y.iloc[:900])

    def test_anchor_and_double_odds(self):
        p = 1/51
        self.assertAlmostEqual(float(score_from_log_odds(logit(p))), 600)
        p_double_good_odds = 1/101
        self.assertAlmostEqual(float(score_from_log_odds(logit(p_double_good_odds))), 620)

    def test_score_probability_round_trip_and_direction(self):
        scores = np.array([400, 600, 800])
        probabilities = probability_from_score(scores)
        np.testing.assert_allclose(score_from_log_odds(logit(probabilities)), scores)
        self.assertTrue(np.all(np.diff(probabilities) < 0))

    def test_exact_points_equal_model_and_rounded_points_obey_bound(self):
        X = self.X.iloc[900:]
        scores = self.card.score_points(X)
        np.testing.assert_allclose(scores, self.card.sum_bin_points(X), atol=1e-10)
        self.assertLessEqual(np.max(np.abs(scores-self.card.sum_bin_points(X, rounded=True))), 3+1e-10)

    def test_holdout_extremes_do_not_refit_preprocessing(self):
        pre = self.card.preprocessor_
        caps, medians = pre.caps_.copy(), pre.medians_.copy()
        new = self.X.iloc[:2].copy()
        new[UTIL], new[INCOME] = 1e8, np.nan
        clean = pre.transform(new)
        pd.testing.assert_series_equal(pre.caps_, caps)
        pd.testing.assert_series_equal(pre.medians_, medians)
        self.assertTrue((clean[UTIL] <= caps[UTIL]).all())
        self.assertTrue((clean.income_missing == 1).all())

    def test_missing_income_has_its_own_bin(self):
        new = self.X.iloc[:2].copy()
        new[INCOME] = np.nan
        clean = self.card.preprocessor_.transform(new)
        self.assertEqual(self.card.binner_.bin(clean)[INCOME].tolist(), ["missing", "missing"])

    def test_woe_formula_matches_raw_counts(self):
        table = self.card.binner_.audit_
        row = table[(table.variable == UTIL) & table.training_supported].iloc[0]
        t = table[(table.variable == UTIL) & table.training_supported]
        gd = (row.good+.5)/(t.good.sum()+.5*len(t))
        bd = (row.bad+.5)/(t.bad.sum()+.5*len(t))
        self.assertAlmostEqual(row.woe, np.log(gd/bd))

    def test_placeholders_are_imputed_using_training_medians(self):
        pre = TrainPreprocessor().fit(self.X)
        new = self.X.iloc[:2].copy()
        new[LATE[0]] = [96, 98]
        self.assertTrue((pre.transform(new)[LATE[0]] == pre.medians_[LATE[0]]).all())

    def test_unknown_count_bin_is_explicit_neutral_fallback(self):
        X = self.X.copy()
        X[LATE] = 0
        model = Scorecard().fit(X, self.y)
        new = X.iloc[:1].copy()
        new[LATE[0]] = 3
        clean = model.preprocessor_.transform(new)
        self.assertEqual(model.binner_.unsupported_counts(clean)[LATE[0]], 1)
        self.assertEqual(model.binner_.transform(clean)[LATE[0]].iloc[0], 0)


class EvaluationTests(unittest.TestCase):
    def test_ties_use_realized_rejection_and_frozen_cutoff(self):
        validation, y = np.array([10.,10.,10.,20.,20.]), np.array([1,0,1,0,0])
        test = np.array([0., 11., 19., 30., 40.])
        table = cutoff_table(validation, y, test, y, shares=[.4])
        self.assertEqual(table.cutoff.iloc[0], 10)
        self.assertEqual(table.validation_realized_reject_share.iloc[0], 0)
        self.assertEqual(table.test_realized_reject_share.iloc[0], .2)

    def test_accept_all_includes_unseen_lower_test_scores(self):
        table = cutoff_table([10,20], [1,0], [-10,50], [1,0], shares=[0])
        self.assertEqual(table.test_approval_rate.iloc[0], 1)

    def test_no_acceptance_has_no_invented_zero_bad_rate(self):
        result = approval_summary([1,2], [0,1], 5)
        self.assertEqual(result["approved"], 0)
        self.assertIsNone(result["approved_bad_rate"])

    def test_paired_identical_models_have_zero_auc_difference_interval(self):
        y = np.tile([0,1], 30)
        p = np.linspace(.01,.99,len(y))
        result = paired_bootstrap(y, p, p, np.arange(len(y)), repetitions=30)
        self.assertEqual(result["forest_minus_scorecard_auc_interval"], [0.,0.])

    def test_group_split_keeps_exact_feature_duplicates_together(self):
        frame = synthetic_data(800)
        frame = pd.concat([frame, frame.iloc[:20]], ignore_index=True)
        X, y = frame.drop(columns=TARGET), frame[TARGET]
        partitions, groups = split_groups(X, y, 42)
        sets = [set(groups[idx]) for idx in partitions.values()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(sum(map(len, partitions.values())), len(frame))

    def test_score_psi_finite_with_empty_bins_and_zero_for_identical_populations(self):
        edges = score_band_edges(np.arange(100))
        self.assertAlmostEqual(score_psi(np.arange(100), np.arange(100), edges), 0)
        self.assertTrue(np.isfinite(score_psi(np.arange(100), [200,201], edges)))

    def test_discrimination_constant_probability_baseline(self):
        metrics = discrimination([0,0,0,1], [.25]*4)
        self.assertEqual(metrics["auc"], .5)
        self.assertAlmostEqual(metrics["brier"], .1875)


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "fixture.csv"
        synthetic_data(900, seed=12).to_csv(self.data, index=False)

    def test_lineage_counts_and_invalid_target(self):
        frame = pd.read_csv(self.data)
        frame.loc[0,"age"], frame.loc[1,"age"] = 17, np.nan
        frame.to_csv(self.data, index=False)
        _, _, audit = load_data(self.data)
        self.assertEqual(audit["rows_read"], 900)
        self.assertEqual(audit["eligible_rows"], 898)
        frame.loc[2,TARGET] = 2
        frame.to_csv(self.data, index=False)
        with self.assertRaisesRegex(ValueError, "binary"):
            load_data(self.data)

    def test_complete_run_recomputes_metrics_and_serialized_scores(self):
        output = self.root / "run"
        run_experiment(self.data, output, trees=10, bootstrap=25, synthetic=True)
        metrics = json.loads((output / "metrics.json").read_text())
        self.assertEqual(metrics["status"], "computed_synthetic_demonstration")
        X,y,_ = load_data(self.data)
        split = pd.read_csv(output / "split_manifest.csv")
        test = split.loc[split.partition == "test", "eligible_row_position"].to_numpy()
        model = joblib.load(output / "model.joblib")["scorecard"]
        actual_auc = roc_auc_score(y.iloc[test], model.predict_proba(X.iloc[test])[:,1])
        self.assertAlmostEqual(actual_auc, metrics["splits"]["test"]["scorecard"]["auc"])
        self.assertEqual(metrics["policy"]["selection_split"], "validation")
        points = pd.read_csv(output / "scorecard_points.csv")
        self.assertEqual(set(points.variable), set(CARD_FEATURES))
        self.assertTrue((output / "dashboard.html").is_file())
        self.assertIn("SYNTHETIC DATA", (output / "dashboard.html").read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            main(["predict", "--model", str(output / "model.joblib"), "--data", str(self.data),
                  "--output", str(self.root / "scored.csv")])
        self.assertEqual(len(pd.read_csv(self.root / "scored.csv")), 900)

    def test_existing_run_preserved(self):
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "fresh"):
            run_experiment(self.data, output)

    def test_index_column_is_ignored_not_assumed_first_predictor(self):
        frame = pd.read_csv(self.data)
        frame.to_csv(self.data, index=True)
        X,_,audit = load_data(self.data)
        self.assertNotIn("Unnamed: 0", X)
        self.assertIn("Unnamed: 0", audit["extra_columns_ignored"])


if __name__ == "__main__":
    unittest.main()
