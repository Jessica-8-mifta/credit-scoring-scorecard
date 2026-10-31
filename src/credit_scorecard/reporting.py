"""Static dashboard generated from one run's machine-readable exports."""

import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def render_report(output):
    output = Path(output)
    metrics = json.loads((output / "metrics.json").read_text())
    audit = json.loads((output / "data_audit.json").read_text())
    synthetic = metrics["status"] == "computed_synthetic_demonstration"
    status = "SYNTHETIC DATA — WORKFLOW DEMONSTRATION ONLY" if synthetic else "NEW RUN — SUPPLIED DATA, GROUPED RANDOM TEST SPLIT"
    test = metrics["splits"]["test"]
    table = pd.DataFrame(test).T.loc[["scorecard", "random_forest", "constant_training_prevalence"],
                                    ["auc", "gini", "ks", "brier", "log_loss"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for model, label in (("scorecard", "WoE logistic"), ("random_forest", "Random Forest")):
        calibration = pd.read_csv(output / f"test_{model}_calibration.csv")
        axes[0].plot(calibration.mean_predicted_probability, calibration.observed_bad_rate, "o-", label=label)
    axes[0].plot([0,1], [0,1], "--", color="gray", label="Perfect calibration")
    axes[0].set(xlabel="Mean predicted probability", ylabel="Observed delinquency rate", title="Test calibration")
    axes[0].legend(fontsize=8)
    bands = pd.read_csv(output / "test_score_bands.csv")
    axes[1].bar(range(len(bands)), bands.bad_rate, color="#244d6a")
    axes[1].set(xlabel="Training-defined score band (low → high)", ylabel="Observed delinquency rate", title="Test risk separation")
    policies = pd.read_csv(output / "cutoff_table.csv")
    axes[2].plot(policies.test_approval_rate, policies.test_approved_bad_rate, "o-", color="#a24e32")
    axes[2].set(xlabel="Test acceptance fraction", ylabel="Bad rate in accepted test records", title="Validation-defined cutoffs")
    for ax in axes:
        ax.grid(alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(status, fontsize=12, fontweight="bold")
    fig.savefig(output / "dashboard.png", dpi=145)
    plt.close(fig)
    data_uri = base64.b64encode((output / "dashboard.png").read_bytes()).decode()
    intervals = metrics["test_auc_uncertainty"]
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Credit Scorecard — Run Report</title><style>body{{font:16px/1.6 system-ui,sans-serif;margin:35px auto;max-width:1200px;padding:0 20px;color:#172b3a;background:#fafbf9}}h1{{line-height:1.2}}.status{{background:#fff0d7;padding:14px;border-left:5px solid #a24e32;font-weight:700}}table{{border-collapse:collapse;font-size:14px}}td,th{{padding:8px 12px;border-bottom:1px solid #ccd5da;text-align:right}}img{{width:100%}}code{{overflow-wrap:anywhere}}.scroll{{overflow-x:auto}}</style>
<h1>Credit Scoring Scorecard</h1><p class="status">{html.escape(status)}</p>
<p>Test records: {audit['partitions']['test']['rows']:,}. Model parameters and preprocessing were fitted on training only.
Cutoff values were frozen on validation scores. This report does not establish a profitable or deployable lending policy.</p>
<h2>Computed test metrics</h2><div class="scroll">{table.to_html(float_format=lambda n: f'{n:.5f}')}</div>
<p>AUC measures ranking. Brier score and log loss evaluate probability predictions; lower is better. The calibration plot compares predictions with observed rates.</p>
<img alt="Calibration, score bands and validation-defined cutoff tradeoffs" src="data:image/png;base64,{data_uri}">
<h2>Paired bootstrap uncertainty</h2><p>95% AUC interval, scorecard: {intervals['scorecard_auc_interval']}; Random Forest: {intervals['forest_auc_interval']}.
Random Forest minus scorecard: {intervals['forest_minus_scorecard_auc_interval']}.</p><p>{html.escape(intervals['note'])}</p>
<h2>Illustrative policy, not an optimum</h2><pre>{html.escape(json.dumps(metrics['policy'], indent=2))}</pre>
<p>Zero-rejection means accept all. Boundary ties are all accepted; actual rejection shares may differ from requested validation quantiles. All other policies in cutoff_table.csv also use validation-derived cutoffs.</p>
<h2>Limitations</h2><ul>{''.join('<li>'+html.escape(s)+'</li>' for s in metrics['limitations'])}</ul>
<p>Input SHA-256: <code>{html.escape(metrics['data_sha256'])}</code>. See run.json, environment.txt, data_audit.json, bin_audit.csv and scorecard_points.csv for the calculation trail.</p></html>'''
    (output / "dashboard.html").write_text(text)
