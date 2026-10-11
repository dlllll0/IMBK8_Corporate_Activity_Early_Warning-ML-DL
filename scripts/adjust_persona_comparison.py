"""Descriptive industry/past-size standardization, not a causal treatment analysis."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True)
p.add_argument('--personas',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise SystemExit('Choose a new output file.')
ids=pd.read_csv(a.personas,usecols=['법인ID','cluster'])
assert ids['법인ID'].is_unique and len(ids)==4507
deposit=['요구불예금잔액','거치식예금잔액','적립식예금잔액']
loan=['여신_운전자금대출잔액','여신_시설자금대출잔액']
channels=['창구거래금액','인터넷뱅킹거래금액','스마트뱅킹거래금액','폰뱅킹거래금액','ATM거래금액','자동이체금액']
raw=pd.read_csv(a.input,usecols=['법인ID','기준년월','업종_대분류']+deposit+loan+channels)
raw=raw[raw['법인ID'].isin(ids['법인ID'])].sort_values(['법인ID','기준년월'])
assert not raw.duplicated(['법인ID','기준년월']).any()
for key,cols in [('deposit',deposit),('loan',loan),('transactions',channels)]:
    assert raw[cols].notna().all().all()
    raw[key]=raw[cols].sum(axis=1)
hist=raw[raw['기준년월'].between(202501,202505)]
past=hist.groupby('법인ID').agg(past_deposit=('deposit','mean'),past_loan=('loan','mean'),past_months=('기준년월','size'),industry=('업종_대분류','last'))
past['past_size']=past.past_deposit+past.past_loan
now=raw[raw['기준년월'].isin([202509,202510,202511])]
cur=now.groupby('법인ID').agg(deposit=('deposit','mean'),loan=('loan','mean'),transactions=('transactions','mean'),observed_months=('기준년월','size'))
d=ids.merge(past,on='법인ID',how='left',validate='one_to_one').merge(cur,on='법인ID',how='left',validate='one_to_one')
d=d[(d.past_months>=3)&d.industry.isin(['제조업','도매 및 소매업','건설업','부동산업'])].copy()
# Derive size boundaries only from pre-assignment data, before outcome-completeness restriction.
boundaries={}
for ind,idx in d.groupby('industry').groups.items():
    edges=np.unique(np.quantile(d.loc[idx,'past_size'],[.2,.4,.6,.8]))
    d.loc[idx,'size_bin']=np.searchsorted(edges,d.loc[idx,'past_size'],side='right')
    boundaries[ind]=len(edges)+1
d['stratum']=d.industry.astype(str)+'|'+d.size_bin.astype(int).astype(str)
eligible=d[d.observed_months.eq(3)].copy()
tab=pd.crosstab(eligible.stratum,eligible.cluster).reindex(columns=[0,1,2],fill_value=0)
common=tab.index[(tab>=20).all(axis=1)]
matched=eligible[eligible.stratum.isin(common)].copy()
assert len(common)>0
weights=matched.groupby('stratum').size()/len(matched)
metrics=['deposit','loan','transactions']
names={0:'거래규모 변동성형',1:'안정 거래형',2:'디지털·자동이체 변동형'}
def mean_indices(frame):
    means=frame.groupby('cluster')[metrics].mean()
    return {str(k):{c:float(means.loc[k,c]/means.loc[1,c]) for c in metrics} for k in means.index}
standard={}
for k in [0,1,2]:
    means=matched[matched.cluster==k].groupby('stratum')[metrics].mean().reindex(weights.index)
    standard[k]=means.mul(weights,axis=0).sum().to_numpy()
rng=np.random.default_rng(42);B=300
boots={k:np.zeros((B,len(metrics))) for k in [0,1,2]}
for k in [0,1,2]:
    for s,w in weights.items():
        arr=matched[(matched.cluster==k)&(matched.stratum==s)][metrics].to_numpy()
        idx=rng.integers(0,len(arr),size=(B,len(arr)))
        boots[k]+=w*arr[idx].mean(axis=1)
groups=[]
for k in [2,0,1]:
    ratio=standard[k]/standard[1]
    br=boots[k]/boots[1]
    groups.append(dict(cluster=k,name=names[k],original_assigned=int(ids.cluster.eq(k).sum()),
                       eligible_complete=int(eligible.cluster.eq(k).sum()),common_support=int(matched.cluster.eq(k).sum()),
                       standardized_mean_index_vs_stable=dict(zip(metrics,map(float,ratio))),
                       bootstrap_percentile95={c:np.quantile(br[:,j],[.025,.975]).tolist() for j,c in enumerate(metrics)}))
out=dict(method='Direct standardization to pooled industry x past-size distribution on common support',
         past_size='202501-202505 mean(deposit+loan), at least 3 observed months; last past industry; within-industry quintiles',
         outcome='202509-202511 mean, all three months observed; unique corporations',
         support_rule='At least 20 corporations from each of the 3 personas in every retained stratum',
         candidate_strata=len(tab),retained_strata=len(common),eligible_corps=len(eligible),common_support_corps=len(matched),
         eligible_unadjusted_indices=mean_indices(eligible),common_support_unadjusted_indices=mean_indices(matched),groups=groups,
         bootstrap=dict(resamples=B,unit='corporation, within persona and stratum',fixed_strata_and_weights=True),
         scope='Descriptive adjusted association; overlapping assignment/outcome periods; not causal, prospective, profit or intervention validation',
         limitations=['Balances are size proxies, not assets or profit','Unobserved confounding and within-bin size differences remain','Common support exclusions restrict generalization','Heavy-tailed amounts; bootstrap intervals are exploratory'])
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(out,ensure_ascii=False,indent=2))
