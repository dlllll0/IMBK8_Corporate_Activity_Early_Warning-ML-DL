"""Read private saved persona assignments + raw CSV; export aggregate indices only.

Retrospective profiles, not an out-of-time policy evaluation. No individual output.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True)
p.add_argument('--personas',type=Path,required=True)
p.add_argument('--summary',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise SystemExit('Choose a new output path; overwrite refused.')
ids=pd.read_csv(a.personas)
assert ids['법인ID'].is_unique and len(ids)==4507
assert ids.n_months.eq(6).all(), 'Expected valid/test six-month assignment artifact.'
assert ids.cluster.value_counts().to_dict()=={0:1817,1:1391,2:1299}
saved=pd.read_csv(a.summary).set_index('cluster')
for k,g in ids.groupby('cluster'):
    assert len(g)==int(saved.loc[k,'corp_count'])
    for c in ['y_rate','blend_score_mean']:
        key='y_rate_mean' if c=='y_rate' else c
        assert np.isclose(g[c].mean(),saved.loc[k,key])
deposit=['요구불예금잔액','거치식예금잔액','적립식예금잔액']
loan=['여신_운전자금대출잔액','여신_시설자금대출잔액']
digital=['인터넷뱅킹거래금액','스마트뱅킹거래금액','폰뱅킹거래금액']
channels=['창구거래금액']+digital+['ATM거래금액','자동이체금액']
cards=['신용카드사용금액','체크카드사용금액']
cols=['법인ID','기준년월']+deposit+loan+channels+cards
raw=pd.read_csv(a.input,usecols=cols)
raw['기준년월']=raw['기준년월'].astype(int)
raw=raw[raw['기준년월'].isin([202509,202510,202511]) & raw['법인ID'].isin(ids['법인ID'])].copy()
assert not raw.duplicated(['법인ID','기준년월']).any()
numeric=list(dict.fromkeys(deposit+loan+channels+cards))
raw[numeric]=raw[numeric].apply(pd.to_numeric,errors='raise')
assert raw[numeric].notna().all().all(), 'Do not silently fill missing monetary cells.'
assert raw[numeric].ge(0).all().all()
for name,cols2 in [('deposit',deposit),('loan',loan),('transactions',channels),('digital',digital),('cards',cards)]:
    raw[name]=raw[cols2].sum(axis=1)
raw['transfer']=raw['자동이체금액']
metrics=['deposit','loan','transactions','digital','cards','transfer']
corp=raw.groupby('법인ID')[metrics].mean()
corp['observed_months']=raw.groupby('법인ID').size()
corp['digital_share']=corp.digital.div(corp.transactions.where(corp.transactions>0))
corp['transfer_share']=corp.transfer.div(corp.transactions.where(corp.transactions>0))
d=ids.merge(corp,on='법인ID',how='left',validate='one_to_one')
names={0:'거래규모 변동성형',1:'안정 거래형',2:'디지털·자동이체 변동형'}
def aggregate(frame):
    stable=frame[frame.cluster==1]
    denom=stable[metrics].mean()
    totals=frame[metrics].sum()
    rows=[]
    for k,g in frame.groupby('cluster'):
        valid=g[g.observed_months.notna()]
        r=dict(cluster=int(k),name=names[k],assigned_corps=len(g),financially_observed_corps=len(valid),
               complete_three_month_corps=int(g.observed_months.eq(3).sum()),
               saved_proxy_rate=float(g.y_rate.mean()),saved_score=float(g.blend_score_mean.mean()),
               saved_imputed_fraction=float(g['보간행비율'].mean()),
               mean_index_vs_stable={c:float(valid[c].mean()/denom[c]) for c in metrics},
               median_index_vs_stable={c:(float(valid[c].median()/stable[c].median()) if stable[c].median()>0 else None) for c in metrics},
               share_of_observed_value={c:float(valid[c].sum()/totals[c]) for c in metrics},
               mean_digital_share=float(valid.digital_share.mean()),mean_transfer_share=float(valid.transfer_share.mean()),
               positive_transactions_corps=int(valid.transactions.gt(0).sum()),
               card_users_fraction=float(valid.cards.gt(0).mean()),
               top_decile_deposit_share=float(valid.deposit.nlargest(int(np.ceil(len(valid)*.1))).sum()/valid.deposit.sum()))
        rows.append(r)
    return rows
groups=aggregate(d)
complete=aggregate(d[d.observed_months.eq(3)])
assumptions=dict(incremental_margin_krw=1000000,contact_cost_krw=20000,fixed_cost_per_separate_campaign_krw=5000000,
                 uplift_rates=[.01,.03,.05],label='HYPOTHETICAL equal economics across groups; no estimated uplift or profitability')
economics=[]
for k,g in ids.groupby('cluster'):
    n=len(g);cost=n*20000+5000000
    economics.append(dict(cluster=int(k),n=n,cost_krw=cost,break_even_uplift=cost/(n*1000000),
                          break_even_margin_at_3pp_krw=cost/(n*.03),
                          scenarios=[dict(uplift=u,net_benefit_krw=n*u*1000000-cost) for u in [.01,.03,.05]]))
out=dict(scope='Retrospective profiling of saved 202506-202511 assignments; observed financial months 202509-202511 only. Not prospective treatment effects.',
         source_version='Related FINAL 90-cell notebook; saved blend F1 0.7770, distinct from submitted 86-cell notebook F1 0.7793.',
         unit='Unique corporation; average of actually observed months per corporation. No missing-month zero fill.',
         period=[202509,202510,202511],groups=groups,complete_three_month_sensitivity=complete,
         monetary_units='Original currency scale not independently established; publish only dimensionless indices and shares, never assume KRW.',
         assumptions=assumptions,hypothetical_group_campaigns=economics)
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(out,ensure_ascii=False,indent=2))
