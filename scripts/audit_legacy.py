"""Reconcile historical aggregates with saved notebook output, without refitting."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def audit():
    notebook_path = ROOT / 'notebooks/original_credit_scoring.ipynb'
    notebook = json.loads(notebook_path.read_text())
    texts = [''.join(''.join(o.get('text', [])) for o in c.get('outputs', [])) for c in notebook['cells']]
    rows = int(re.search(r'\((\d+),\s*11\)', texts[0]).group(1))
    auc, gini, ks = map(float, re.search(r'AUC ([\d.]+) Gini ([\d.]+) KS ([\d.]+)', texts[4]).groups())
    rf = float(re.search(r'RF AUC ([\d.]+)', texts[5]).group(1))
    psi = float(re.search(r'PSI ([\d.]+)', texts[5]).group(1))
    bands = pd.read_csv(ROOT / 'outputs/score_bands.csv')
    cutoffs = pd.read_csv(ROOT / 'outputs/cutoff_table.csv')
    n, bads = int(bands.applicants.sum()), int(bands.bads.sum())
    assert n == math.ceil(rows*.3), 'Holdout size does not match notebook split'
    assert ((bands.bads/bands.applicants-bands.bad_rate).abs() < 1e-12).all()
    assert abs(bads/n-cutoffs.bad_rate.iloc[0]) < 1e-12
    assert abs(2*auc-1-gini) <= .001001, 'Rounded AUC and Gini do not reconcile'
    policies = []
    for row in cutoffs.itertuples(index=False):
        approved = n*row.approval_rate
        approved_bads = approved*row.bad_rate
        assert abs(approved-round(approved)) < 1e-6
        assert abs(approved_bads-round(approved_bads)) < 1e-6
        policies.append({'requested_holdout_reject_share': row.reject_share,
                         'realized_reject_share': 1-row.approval_rate, 'cutoff': row.cutoff,
                         'approved': round(approved), 'approved_bads': round(approved_bads),
                         'approved_bad_rate': row.bad_rate})
    return {'status':'historical_saved_outputs_reconciled_not_refitted',
            'notebook_sha256':hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
            'input_filename_in_original_code':'train.csv.zip', 'input_rows_from_saved_output':rows,
            'raw_data_sha256':None, 'holdout_applicants':n,'holdout_bads':bads,'holdout_bad_rate':bads/n,
            'lowest_band_share_of_bads':float(bands.bads.iloc[0]/bads),
            'lowest_three_bands_share_of_bads':float(bands.bads.iloc[:3].sum()/bads),
            'reported_rounded_metrics':{'scorecard_auc':auc,'scorecard_gini':gini,'scorecard_ks':ks,
                                        'random_forest_auc':rf,'score_psi':psi},
            'policies_explored_on_holdout':policies,
            'notes':['Saved notebook outputs match the aggregate exports; this is not an independent model rerun.',
                     'The original code fits medians, caps and WoE maps on training data.',
                     'Utilization/age cutpoints were typed after training-only exploration; exact source-data provenance is unavailable.',
                     'Cutoffs were derived from holdout score quantiles; all tied scores at a cutoff were accepted.',
                     'Original fitted model, full-precision parameters and points CSV are unavailable; they are not reconstructed from rounded logs.']}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(ROOT/'reports/legacy_audit.json'))
    args=parser.parse_args()
    result=audit()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f"Historical aggregates reconciled: {result['holdout_applicants']:,} applicants, {result['holdout_bads']:,} bads. No model was refitted.")
