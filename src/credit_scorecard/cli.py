import argparse
from pathlib import Path
import tempfile

import joblib
import pandas as pd

from .model import validate_features
from .pipeline import run_experiment, synthetic_data


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description="Auditable WoE scorecard experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train/evaluate a new run on a local labeled CSV or single-file CSV ZIP")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--trees", type=positive, default=300)
    train.add_argument("--bootstrap", type=positive, default=500)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--reject-share", type=float, default=.3)
    demo = commands.add_parser("demo", help="Run the real workflow on clearly labeled synthetic data")
    demo.add_argument("--output", required=True)
    demo.add_argument("--rows", type=positive, default=2000)
    predict = commands.add_parser("predict", help="Score records with a trusted locally trained model")
    predict.add_argument("--model", required=True)
    predict.add_argument("--data", required=True)
    predict.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            output = run_experiment(args.data, args.output, seed=args.seed, trees=args.trees,
                                    bootstrap=args.bootstrap, reject_share=args.reject_share)
        elif args.command == "demo":
            if args.rows < 500:
                raise ValueError("Use at least 500 synthetic rows")
            with tempfile.TemporaryDirectory() as temp:
                data = Path(temp) / "synthetic_fixture.csv"
                synthetic_data(args.rows).to_csv(data, index=False)
                output = run_experiment(data, args.output, trees=30, bootstrap=100, synthetic=True)
        else:
            output = Path(args.output).resolve()
            if output.exists():
                raise ValueError("Choose a new output CSV")
            X = validate_features(pd.read_csv(args.data, compression="infer"))
            model = joblib.load(args.model)  # Only trusted, version-compatible local models.
            card = model["scorecard"]
            scores = card.score_points(X)
            result = pd.DataFrame({"input_row_position": range(len(X)), "score_exact": scores,
                                   "score_rounded_bin_sum": card.sum_bin_points(X, rounded=True),
                                   "predicted_two_year_delinquency_probability": card.predict_proba(X)[:, 1]})
            output.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output, index=False)
        print(f"Saved: {output}")
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"Error: {exc}\n")
