"""Read-only package checks plus synthetic label tests; no private data required."""
from pathlib import Path
import ast, json, re, sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.label_utils import parse_count_str, compute_rolling_change, rolling_cv

def check():
    # Synthetic unit examples; never used as project performance evidence.
    assert parse_count_str('10건초과 20건이하')==11
    assert parse_count_str('0건')==0
    assert parse_count_str(None)==0
    a=pd.Series([10,10,10,5,5,5,99],dtype=float)
    m=compute_rolling_change(a)
    assert np.isnan(m[:6]).all()
    assert np.isclose(m[6],-15/(30+1e-6))
    # Changing the next month cannot change the already-known decline component.
    b=a.copy();b.iloc[6]=100000
    assert compute_rolling_change(a)[6]==compute_rolling_change(b)[6]
    assert rolling_cv(a)[6]!=rolling_cv(b)[6]
    d=json.loads((ROOT/'results/supplementary-results.json').read_text(encoding='utf-8'))
    model=json.loads((ROOT/'results/original-metrics.json').read_text(encoding='utf-8'))
    for x in model['models']+[d['persistence'],d['known_decline'],d['frozen_persistence']]:
        (tn,fp),(fn,tp)=x['confusion_matrix']
        assert tn+fp+fn+tp==13521
        assert abs(2*tp/(2*tp+fp+fn)-x['f1'])<.000051
        assert abs(tp/(tp+fn)-x['recall'])<.000051
        assert abs(tp/(tp+fp)-x['precision'])<.000051
    assert d['clean_windows']['n']+d['imputed_exposure']['n']==d['test_rows']
    assert sum(x['positive'] for x in d['monthly'])==d['test_positive']
    personas=json.loads((ROOT/'results/persona-analysis.json').read_text(encoding='utf-8'))
    assert sum(g['assigned_corps'] for g in personas['groups'])==4507
    assert sum(g['financially_observed_corps'] for g in personas['groups'])==4504
    for metric in ['deposit','loan','transactions','cards']:
        assert np.isclose(sum(g['share_of_observed_value'][metric] for g in personas['groups']),1)
    for g in personas['hypothetical_group_campaigns']:
        assert np.isclose(g['n']*g['break_even_uplift']*1000000,g['cost_krw'])
        for s in g['scenarios']:
            assert np.isclose(s['net_benefit_krw'],g['n']*s['uplift']*1000000-g['cost_krw'])
    paired=json.loads((ROOT/'results/paired-revalidation.json').read_text(encoding='utf-8'))
    totals=set()
    for r in paired['results'].values():
        (tn,fp),(fn,tp)=r['confusion_matrix'];totals.add(tn+fp+fn+tp)
        assert np.isclose(r['f1'],2*tp/(2*tp+fp+fn))
        assert r['current_state_0']['n']+r['current_state_1']['n']==r['n']
        for i in range(2):
            for j in range(2):
                assert r['confusion_matrix'][i][j]==sum(r[f'current_state_{s}']['confusion_matrix'][i][j] for s in [0,1])
    assert totals=={11953}
    for c in paired['coverage'].values():assert c['candidate_rows']==c['evaluated_rows']+c['excluded_rows']
    for name,r in paired['results'].items():
        for i in range(2):
            for j in range(2):assert r['confusion_matrix'][i][j]==sum(m['metrics'][name]['confusion_matrix'][i][j] for m in paired['monthly'])
    adj=json.loads((ROOT/'results/adjusted-persona-comparison.json').read_text(encoding='utf-8'))
    assert sum(g['common_support'] for g in adj['groups'])==adj['common_support_corps']==3193
    assert adj['common_support_corps']<=adj['eligible_corps']<=4507
    failures=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or any(part in {'.git','.venv','__pycache__','local-results'} for part in p.parts):continue
        if p.suffix in {'.csv','.xlsx','.xls','.parquet','.pkl','.joblib','.zip'}:failures.append(f'Unexpected private-data type: {p.name}')
        if p.suffix not in {'.md','.py','.json','.ipynb','.txt'}:continue
        text=p.read_text(encoding='utf-8')
        if re.search(r'(?i)[A-Z]:[/\\](?:Users|Projects)[/\\]',text):failures.append(f'Absolute internal path: {p.name}')
        if re.search(r'\b[a-fA-F0-9]{32,64}\b',text):failures.append(f'Identifier-like token: {p.name}')
        if re.search(r'(?i)(?:api[_-]?key|password|secret)\s*=\s*[\"\x27][^\"\x27]{8,}',text):failures.append(f'Possible credential: {p.name}')
        if p.suffix=='.py':ast.parse(text,filename=str(p))
        if p.suffix=='.ipynb':
            nb=json.loads(text)
            for c in nb['cells']:
                if c['cell_type']=='code':
                    assert not c.get('outputs') and c.get('execution_count') is None
                    ast.parse(''.join(c['source']))
        if p.suffix=='.md':
            for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
                if link.startswith(('http','#')):continue
                if not (p.parent/link.split('#')[0]).exists():failures.append(f'Broken link: {p.name}: {link}')
    assert not failures,'\n'.join(failures)
    print('PASS: synthetic label tests, confusion-matrix metrics, notebook/Python syntax, links, output/privacy scan.')

if __name__=='__main__':check()
