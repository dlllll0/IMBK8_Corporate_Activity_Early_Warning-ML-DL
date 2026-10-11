"""Past-only, same-row LightGBM/rule comparison. Private CSV -> aggregates only.

This is a corrected internal-data experiment, NOT an exact replay of historical models.
"""
from pathlib import Path
import argparse, json, sys, time, warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (confusion_matrix,f1_score,precision_score,recall_score,
                             roc_auc_score,average_precision_score,precision_recall_curve)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.label_utils import build_corp_profile,parse_count_str,EPS

warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise SystemExit('Output already exists; choose a new file.')
start=time.monotonic()
def log(s):print(f'[{time.monotonic()-start:.0f}s] {s}',flush=True)
raw=pd.read_csv(a.input)
raw['기준년월']=raw['기준년월'].astype(int)
raw=raw[raw['기준년월'].between(202301,202512)].drop_duplicates(['법인ID','기준년월'],keep='last')
assert raw.shape==(365988,70)
counts=['창구거래건수','인터넷뱅킹거래건수','스마트뱅킹거래건수','폰뱅킹거래건수','ATM거래건수','자동이체거래건수']
industries=['제조업','도매 및 소매업','건설업','부동산업']
hist=raw[raw['기준년월']<=202505].sort_values(['법인ID','기준년월'])
months_past=[int(t.strftime('%Y%m')) for t in pd.period_range('2023-01','2025-05',freq='M')]
q=1-len(hist)/(hist['법인ID'].nunique()*len(months_past))
profile=build_corp_profile(hist,months_past,q)
eligible=profile.loc[profile.final_status.isin(['완전관측_정상','중간랜덤누락_보간후보']),'법인ID']
last_ind=hist.groupby('법인ID')['업종_대분류'].last()
eligible=sorted(set(eligible)&set(last_ind[last_ind.isin(industries)].index))
log(f'Past-only cohort selected: {len(eligible)} corporations')
months=[int(t.strftime('%Y%m')) for t in pd.period_range('2023-01','2025-12',freq='M')]
d=raw[raw['법인ID'].isin(eligible)].copy()
d['observed']=1
d=d.set_index(['법인ID','기준년월']).reindex(pd.MultiIndex.from_product([eligible,months],names=['법인ID','기준년월'])).reset_index()
d['industry_group']=d['법인ID'].map(last_ind)
valid_counts=d[counts].notna().all(axis=1)&d.observed.eq(1)
d['frequency']=sum(d[c].map(parse_count_str) for c in counts).where(valid_counts)
g=d.groupby('법인ID')['frequency']
recent=g.transform(lambda s:s.shift(1).rolling(3,min_periods=3).sum())
old=g.transform(lambda s:s.shift(4).rolling(3,min_periods=3).sum())
d['M']=(recent-old)/(old+EPS)
d['CV']=g.transform(lambda s:s.rolling(6,min_periods=6).std()/(s.rolling(6,min_periods=6).mean()+EPS))
d['count_available']=valid_counts.astype(int)
d['past7_observed']=d.groupby('법인ID')['count_available'].transform(lambda s:s.rolling(7,min_periods=7).sum()).eq(7)
cut=d.loc[(d['기준년월']<=202505)&d.past7_observed,'CV'].mean()
d['current']=((d.M<=-.3)|(d.CV>cut)).astype(float).where(d.past7_observed)
d['target']=d.groupby('법인ID')['current'].shift(-1)
valid=d.past7_observed & d.target.notna()
assert not d.loc[valid,['M','CV','current','target']].isna().any().any()
# Features: internal observed values and causal history only; no backward/linear filling.
numeric=raw.select_dtypes(include=np.number).columns.difference(['기준년월']).tolist()
cats=[c for c in raw.columns if c not in numeric+['법인ID','기준년월'] and '거래건수' not in c]
for c in cats:d[c]=d[c].fillna('Unknown').astype(str)
features=numeric.copy()
totals={'deposit':['요구불예금잔액','거치식예금잔액','적립식예금잔액'],
        'loan':['여신_운전자금대출잔액','여신_시설자금대출잔액'],
        'transactions':['창구거래금액','인터넷뱅킹거래금액','스마트뱅킹거래금액','폰뱅킹거래금액','ATM거래금액','자동이체금액'],
        'digital':['인터넷뱅킹거래금액','스마트뱅킹거래금액','폰뱅킹거래금액','ATM거래금액'],
        'cards':['신용카드사용금액','체크카드사용금액'],
        'fx':['외환_수출실적금액','외환_수입실적금액']}
for c,cols in totals.items():d[c]=d[cols].sum(axis=1,min_count=len(cols));features.append(c)
d['digital_ratio']=d.digital/(d.transactions+EPS)
d['transfer_ratio']=d['자동이체금액']/(d.transactions+EPS)
for c in list(totals)+['자동이체금액','digital_ratio','transfer_ratio']:
    g=d.groupby('법인ID')[c]
    for lag in [1,3,6]:d[f'{c}_lag{lag}']=g.shift(lag);features.append(f'{c}_lag{lag}')
    d[f'{c}_diff1']=g.diff(1)
    d[f'{c}_pct3']=(d[c]-g.shift(3))/(g.shift(3).abs()+EPS)
    d[f'{c}_ma3']=g.transform(lambda s:s.rolling(3,min_periods=3).mean())
    d[f'{c}_cv6']=g.transform(lambda s:s.rolling(6,min_periods=6).std()/(s.rolling(6,min_periods=6).mean()+EPS))
    features.extend(f'{c}_{s}' for s in ['diff1','pct3','ma3','cv6'])
features+=['digital_ratio','transfer_ratio']
for c in features:d[c]=pd.to_numeric(d[c],errors='raise').replace([np.inf,-np.inf],np.nan).astype('float32')
# Extra model receives the information already available to the persistence rule.
aug=['frequency','current','M','CV']
for lag in [1,3,6]:
    c=f'frequency_lag{lag}';d[c]=d.groupby('법인ID').frequency.shift(lag);aug.append(c)
frame=d.loc[valid].copy()
train=frame[frame['기준년월'].between(202307,202505)].copy()
val=frame[frame['기준년월'].isin([202506,202507,202508])].copy()
test=frame[frame['기준년월'].isin([202509,202510,202511])].copy()
assert len(train)>0 and len(val)>0 and len(test)>0
for f in [train,val,test]:f['target']=f.target.astype(int)
log(f'Same-row split sizes {len(train)}/{len(val)}/{len(test)}; CV cutoff {cut:.8f}')
params=dict(n_estimators=500,learning_rate=.05,max_depth=6,num_leaves=31,random_state=42,n_jobs=4,verbosity=-1,deterministic=True)
def fit_model(tr,va,te,nums,category_columns):
    assert not any('거래건수' in c for c in nums+category_columns)
    prep=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=True),category_columns),('num','passthrough',nums)])
    pipe=Pipeline([('prep',prep),('model',LGBMClassifier(**params))])
    pipe.fit(tr[nums+category_columns],tr.target)
    return pipe.predict_proba(va[nums+category_columns])[:,1],pipe.predict_proba(te[nums+category_columns])[:,1]
def threshold(y,s):
    pr,re,th=precision_recall_curve(y,s)
    f=2*pr[:-1]*re[:-1]/(pr[:-1]+re[:-1]+1e-12)
    return float(th[np.argmax(f)])
v_int,t_int=fit_model(train,val,test,features,cats)
log('Integrated LightGBM trained')
v_route=np.full(len(val),np.nan);t_route=np.full(len(test),np.nan)
for industry in industries:
    tr=train[train.industry_group==industry];vm=val.industry_group.eq(industry).to_numpy();tm=test.industry_group.eq(industry).to_numpy()
    vi,ti=fit_model(tr,val.loc[vm],test.loc[tm],features,cats)
    v_route[vm]=vi;t_route[tm]=ti
    log(f'Industry model trained: {industry}')
assert np.isfinite(v_route).all() and np.isfinite(t_route).all()
v_aug,t_aug=fit_model(train,val,test,features+aug,cats)
log('State/history-augmented model trained')
scores={'integrated':(v_int,t_int),'routed':(v_route,t_route),'blend':((v_int+v_route)/2,(t_int+t_route)/2),'with_state_history':(v_aug,t_aug)}
def metric(y,p,s=None):
    out=dict(n=len(y),positive=int(np.sum(y)),f1=float(f1_score(y,p,zero_division=0)),precision=float(precision_score(y,p,zero_division=0)),recall=float(recall_score(y,p,zero_division=0)),confusion_matrix=confusion_matrix(y,p,labels=[0,1]).tolist(),flagged=int(np.sum(p)))
    if s is not None:out.update(roc_auc=float(roc_auc_score(y,s)),average_precision=float(average_precision_score(y,s)))
    return out
y=test.target.to_numpy();base=test.current.to_numpy().astype(int)
predictions={'persistence':base};results={'persistence':metric(y,base)}
for name,(vs,ts) in scores.items():
    th=threshold(val.target,vs);pred=(ts>=th).astype(int)
    predictions[name]=pred;results[name]=metric(y,pred,ts);results[name]['validation_threshold']=th
for name,pred in predictions.items():
    for state in [0,1]:
        mask=base==state
        results[name][f'current_state_{state}']=metric(y[mask],pred[mask])
monthly=[]
for month,idx in test.groupby('기준년월').indices.items():
    row=dict(month=int(month),n=len(idx),positive=int(y[idx].sum()),metrics={},top500={})
    for name,pred in predictions.items():row['metrics'][name]=metric(y[idx],pred[idx])
    k=min(500,len(idx))
    for name,(_,ts) in scores.items():
        # Expected yield within tied scores, avoiding arbitrary customer-ID tie-breaking.
        s=ts[idx];yy=y[idx];cutscore=np.sort(s)[-k];above=s>cutscore;tie=s==cutscore
        expected=float(yy[above].sum()+(k-above.sum())*yy[tie].mean())
        row['top500'][name]=dict(k=k,expected_true_positive=expected,precision=expected/k,recall=expected/yy.sum())
    s=base[idx];yy=y[idx];cutscore=np.sort(s)[-k];above=s>cutscore;tie=s==cutscore
    expected=float(yy[above].sum()+(k-above.sum())*yy[tie].mean())
    row['top500']['persistence_random_tie']=dict(k=k,expected_true_positive=expected,precision=expected/k,recall=expected/yy.sum())
    monthly.append(row)
# Paired cluster bootstrap: aggregate confusion contributions per corporation, then resample corporations.
codes,unique_ids=pd.factorize(test['법인ID'],sort=True)
rng=np.random.default_rng(42);B=300;nids=len(unique_ids)
weights=rng.multinomial(nids,np.full(nids,1/nids),size=B)
boot={}
for name,pred in predictions.items():
    tp=np.bincount(codes,weights=((pred==1)&(y==1)),minlength=nids)
    fp=np.bincount(codes,weights=((pred==1)&(y==0)),minlength=nids)
    fn=np.bincount(codes,weights=((pred==0)&(y==1)),minlength=nids)
    t=weights@tp;f=weights@fp;miss=weights@fn
    boot[name]=2*t/np.maximum(2*t+f+miss,1)
cis={name:dict(delta_f1=results[name]['f1']-results['persistence']['f1'],percentile_95=np.quantile(values-boot['persistence'],[.025,.975]).tolist(),resamples=B) for name,values in boot.items() if name!='persistence'}
coverage={}
for label,ms in [('train',list(range(0))),('validation',[202506,202507,202508]),('test',[202509,202510,202511])]:
    mask=d['기준년월'].between(202307,202505) if label=='train' else d['기준년월'].isin(ms)
    coverage[label]=dict(candidate_rows=int(mask.sum()),evaluated_rows=int((mask&valid).sum()),excluded_rows=int((mask&~valid).sum()))
out=dict(experiment='Past-only internal-data revalidation; NOT historical exact reproduction',protocol='work/project4-revalidation-protocol.md (private review copy)',
         cohort=dict(cutoff=202505,corporations=len(eligible),test_corporations=nids,selection='Original missing-profile rule fitted only through May 2025; established top4 industries'),
         label=dict(cv_cutoff=float(cut),cv_fit_through=202505,definition='Original OR proxy with -30% decline, no missing-month zero imputation',required_observation_window='t-6 through t+1, 8 observed months for paired current/next state'),
         availability='Train target through June 2025 available at first validation June month-end; last validation target September available at first test September month-end.',
         coverage=coverage,feature_count=dict(numeric=len(features),categorical=len(cats),extra_state_history=len(aug)),parameters=params,
         changes=['Past-only cohort selection','CV cutoff fixed before validation','Missing month labels excluded rather than zero-filled','Causal internal features; no external indicators with unknown publication times','No whole-period preprocessing fits','LightGBM native numeric missing; train-fit categorical encoding'],
         results=results,monthly=monthly,paired_corporation_bootstrap=cis,
         limitations=['Proxy retains overlapping windows; not actual churn','Observation-complete target subset may be selective','Existing long-history corporations only','No held-out corporation validation','No measured intervention or profit','Historical all-period 2-label discrepancy not resolved by this new experiment'])
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
log('Completed: aggregate results only')
for name,r in results.items():print(name,round(r['f1'],6),round(r['recall'],6),flush=True)
