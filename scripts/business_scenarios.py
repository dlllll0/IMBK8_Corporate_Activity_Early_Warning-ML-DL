"""Illustrative campaign economics, NOT measured project revenue or causal uplift."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
original = json.loads((ROOT / 'results/original-metrics.json').read_text(encoding='utf-8'))
audit = json.loads((ROOT / 'results/supplementary-results.json').read_text(encoding='utf-8'))
cm = original['models'][2]['confusion_matrix']
alerts = cm[0][1] + cm[1][1]
baseline_alerts = audit['persistence']['flagged']
# All campaign inputs below are invented assumptions for a future, deduplicated pilot.
# They are not estimated from the original confusion matrix.
n, contact_cost, fixed_cost, margin = 1000, 20000, 5000000, 1000000
total_cost = n * contact_cost + fixed_cost
scenarios = []
for uplift in [0.01, 0.03, 0.05]:
    benefit = n * uplift * margin
    scenarios.append(dict(incremental_retention_rate=uplift,
                          expected_incremental_retained_customers=n * uplift,
                          incremental_contribution_krw=benefit,
                          net_benefit_krw=benefit-total_cost,
                          roi=(benefit-total_cost)/total_cost))
out = dict(
    status='HYPOTHETICAL economics, not observed business results',
    observed_workload=dict(unit='corporation-month',months=3,blend_alerts=alerts,
                           persistence_alerts=baseline_alerts,
                           aggregate_difference=alerts-baseline_alerts,
                           caveat='Aggregate comparison, not paired replication; no unique customer count.'),
    assumed_pilot=dict(unique_customers=n,follow_up_months=3,contact_cost_per_customer_krw=contact_cost,
                       fixed_cost_krw=fixed_cost,incremental_margin_per_retained_customer_krw=margin,
                       total_cost_krw=total_cost,
                       margin_definition='Three-month incremental contribution after variable/funding/risk costs; excludes campaign costs already counted separately.'),
    scenarios=scenarios,
    break_even_uplift=total_cost/(n*margin),
    formula='N * causal_incremental_retention_rate * incremental_margin - N * contact_cost - fixed_cost',
    exclusions=['Actual churn ground truth','Measured causal uplift','Measured bank margins','Annual extrapolation','Duplicate benefits across months'])
assert alerts == 5996 and alerts-baseline_alerts == 599
assert [s['net_benefit_krw'] for s in scenarios] == [-15000000,5000000,25000000]
(ROOT/'results/business-scenarios.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Hypothetical net benefit (KRW):', [s['net_benefit_krw'] for s in scenarios])
print('Break-even incremental retention:', out['break_even_uplift'])
