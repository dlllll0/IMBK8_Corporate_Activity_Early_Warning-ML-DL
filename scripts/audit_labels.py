"""Local private CSV -> aggregate-only label audit. No network or raw-data output."""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.label_utils import build_corp_profile, parse_count_str, compute_rolling_change, rolling_cv, EPS
p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True,help='Private 70-column corporate CSV (not supplied)')
p.add_argument('--output',type=Path,required=True,help='New aggregate JSON path; overwrite refused')
args=p.parse_args()
if args.output.exists():raise SystemExit('Output already exists; choose a new file.')
if args.input.resolve()==args.output.resolve():raise SystemExit('Input/output must differ.')
raw=args.input

counts=['창구거래건수','인터넷뱅킹거래건수','스마트뱅킹거래건수','폰뱅킹거래건수','ATM거래건수','자동이체거래건수']
cols=['법인ID','기준년월','업종_대분류']+counts
header=pd.read_csv(raw,nrows=0)
d=pd.read_csv(raw,usecols=cols)
d['기준년월']=d['기준년월'].astype(str).str.replace('-','',regex=False).str[:6].astype(int)
months=[int(x.strftime('%Y%m')) for x in pd.period_range('2023-01','2025-12',freq='M')]
d=d[d['기준년월'].isin(months)].drop_duplicates(['법인ID','기준년월'],keep='last').sort_values(['법인ID','기준년월'])
print('Raw schema/rows/corps:',len(header.columns),len(d),d['법인ID'].nunique(),flush=True)
assert (len(header.columns),len(d),d['법인ID'].nunique())==(70,365988,15473), 'Raw data differs from source notebook; stop.'
q=1-len(d)/(d['법인ID'].nunique()*36)
p=build_corp_profile(d,months,q)
selected=p[p.final_status.isin(['완전관측_정상','중간랜덤누락_보간후보'])]['법인ID']
assert len(selected)==5723
d=d[d['법인ID'].isin(selected)].copy()
d['observed']=1
d=d.set_index(['법인ID','기준년월']).reindex(pd.MultiIndex.from_product([sorted(selected),months],names=['법인ID','기준년월'])).reset_index()
d['imputed']=d['observed'].isna().astype(int)
d['업종_대분류']=d.groupby('법인ID')['업종_대분류'].transform(lambda x:x.ffill().bfill())
d['총거래건수']=sum(d[c].fillna('0건').map(parse_count_str) for c in counts)
g=d.groupby('법인ID')['총거래건수']
d['M_freq']=g.transform(lambda x:compute_rolling_change(x.reset_index(drop=True)))
d['freq_CV']=g.transform(lambda x:rolling_cv(x.reset_index(drop=True)).values)
available=d.M_freq.notna() & d.freq_CV.notna()
threshold=d.loc[available,'freq_CV'].mean()
d['current']=np.where(available,((d.M_freq<=-.3)|(d.freq_CV>threshold)).astype(int),np.nan)
d['target']=d.groupby('법인ID')['current'].shift(-1)
d['next_decline']=d.groupby('법인ID')['M_freq'].shift(-1)<=-.3
recent=g.transform(lambda x:x.rolling(3).sum())
previous=d.groupby('법인ID')['총거래건수'].transform(lambda x:x.shift(3).rolling(3).sum())
d['known_decline_now']=(recent-previous)/(previous+EPS)<=-.3
d['next_imputed']=d.groupby('법인ID')['imputed'].shift(-1)
d['window_imputed']=d.groupby('법인ID')['imputed'].transform(lambda x:x.rolling(6,min_periods=6).sum())
# Freeze only CV definition at last training target month. Cohort remains original retrospectively selected cohort.
train_threshold=d.loc[available & (d['기준년월']<=202506),'freq_CV'].mean()
d['frozen_current']=np.where(available,((d.M_freq<=-.3)|(d.freq_CV>train_threshold)).astype(int),np.nan)
d['frozen_target']=d.groupby('법인ID')['frozen_current'].shift(-1)
m=d[available & d.target.notna() & d['업종_대분류'].isin(['제조업','도매 및 소매업','건설업','부동산업'])].copy()
t=m[m['기준년월'].isin([202509,202510,202511])].copy()
assert (len(t),int(t.target.sum()),m['법인ID'].nunique())==(13521,5458,4507), 'Target does not match saved model evaluation.'
assert (t.next_decline==t.known_decline_now).all()
print('Additional full-period cross-check:',len(m),int(d.loc[available & d.target.notna(),'target'].sum()),flush=True)
assert len(m)==130703
def metrics(y,p):
    return dict(confusion_matrix=confusion_matrix(y,p).tolist(),precision=float(precision_score(y,p,zero_division=0)),recall=float(recall_score(y,p,zero_division=0)),f1=float(f1_score(y,p,zero_division=0)),flagged=int(np.sum(p)))
result=dict(raw_rows=365988,raw_columns=70,raw_corps=15473,selected_corps=5723,top4_corps=4507,imputed_rows=int(d.imputed.sum()),test_rows=len(t),test_positive=int(t.target.sum()),cv_global=float(threshold),cv_through_last_training_target=float(train_threshold),changed_test_labels=int((t.target!=t.frozen_target).sum()),frozen_test_positive=int(t.frozen_target.sum()),persistence=metrics(t.target,t.current),known_decline=metrics(t.target,t.next_decline),target_imputed_rows=int(t.next_imputed.sum()),input_window_imputed_rows=int((t.window_imputed>0).sum()),monthly=[])
result['all_industries_model_rows']=int((available & d.target.notna()).sum())
result['all_industries_rerun_positive']=int(d.loc[available & d.target.notna(),'target'].sum())
result['all_industries_saved_positive']=64272
result['comparison_scope']='Aggregate-aligned comparison, not paired replication: full-period all-industry positives differ by 2; original row-level predictions unavailable.'
result['frozen_persistence']=metrics(t.frozen_target,t.frozen_current)
result['frozen_rule_against_original_target']=metrics(t.target,t.frozen_current)
for month,x in t.groupby('기준년월'):
    result['monthly'].append(dict(month=int(month),n=len(x),positive=int(x.target.sum()),persistence=metrics(x.target,x.current),known_decline=metrics(x.target,x.next_decline)))
for label,mask in [('clean_windows',(t.window_imputed==0)&(t.next_imputed==0)),('imputed_exposure',(t.window_imputed>0)|(t.next_imputed>0))]:
    x=t[mask]
    result[label]=dict(n=len(x),positive=int(x.target.sum()),positive_rate=float(x.target.mean()),persistence=metrics(x.target,x.current))
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('x',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
