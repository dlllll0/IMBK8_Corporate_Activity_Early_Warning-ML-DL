"""Render aggregate figures only; overwrite the two derived PNGs under assets."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
names={f.name for f in font_manager.fontManager.ttflist}
for font in ['Malgun Gothic','AppleGothic','Noto Sans CJK KR','NanumGothic']:
    if font in names:
        plt.rcParams['font.family']=font
        break
plt.rcParams['axes.unicode_minus']=False
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
d=json.loads((ROOT/'results/supplementary-results.json').read_text(encoding='utf-8'))
models=json.loads((ROOT/'results/original-metrics.json').read_text(encoding='utf-8'))['models']
fig,ax=plt.subplots(figsize=(9,4.8))
values=[models[2]['f1'],d['persistence']['f1']]
bars=ax.barh(['원본 블렌딩\n저장된 평가','상태 유지 기준\n이번 재계산'],values,color=['#5478a8','#197c80'],height=.5)
ax.set(xlim=(0,1),xlabel='F1',title='모델의 활용 가치를 단순 기준과 함께 검토')
ax.invert_yaxis()
for b,v in zip(bars,values):ax.text(v+.013,b.get_y()+b.get_height()/2,f'{v:.4f}',va='center',fontweight='bold')
ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.text(.04,.025,'법인·월 13,521행 / 양성 5,458행 · 집계 수준 비교, 동일 실행의 짝지은 검증 아님',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.065,1,1))
fig.savefig(ROOT/'assets/baseline-comparison.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.8))
groups=[d['clean_windows'],d['imputed_exposure']]
vals=[x['positive_rate']*100 for x in groups]
bars=ax.bar(['생성된 누락 월 없음\n11,756행','생성된 누락 월 있음\n1,765행'],vals,color=['#5478a8','#d08948'],width=.48)
ax.set(ylim=(0,105),ylabel='대리 목표 양성 비율 (%)',title='누락 대체 노출에 따라 다른 활동 약화 라벨 비율')
for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+2,f'{v:.1f}%',ha='center',fontweight='bold')
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.text(.04,.025,'현재 포함 최근 6개월 + 목표 월 기준 · 관측 집단의 비교이며 인과효과가 아님',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.065,1,1))
fig.savefig(ROOT/'assets/imputation-audit.png',dpi=160);plt.close(fig)
print('Rendered 2 aggregate figures.')

persona_path=ROOT/'results/persona-analysis.json'
if persona_path.is_file():
    personas=json.loads(persona_path.read_text(encoding='utf-8'))
    lookup={x['cluster']:x for x in personas['groups']}
    ordered=[lookup[k] for k in [2,0,1]]
    fig,ax=plt.subplots(figsize=(10,5.2))
    keys=['deposit','loan','transactions']
    labels=['평균 예금잔액','평균 여신잔액','평균 채널 거래금액']
    colors=['#197c80','#5478a8','#d08948']
    for j,(key,label,color) in enumerate(zip(keys,labels,colors)):
        pos=[i+(j-1)*.24 for i in range(3)]
        values=[g['mean_index_vs_stable'][key] for g in ordered]
        bars=ax.bar(pos,values,width=.22,label=label,color=color)
        for b,v in zip(bars,values):
            ax.text(b.get_x()+b.get_width()/2,v+.025,f'{v:.2f}',ha='center',fontsize=10)
    ax.set_xticks(range(3),['디지털·자동이체 변동형','거래규모 변동성형','안정 거래형'])
    ax.set(ylim=(0,1.24),ylabel='안정 거래형 평균 = 1',title='활동 약화 위험과 거래 규모는 같은 순서가 아니었다')
    ax.legend(loc='upper left',ncol=3,fontsize=10,frameon=False)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.04,.025,'2025.09–11 실제 관측 월의 법인별 평균 · 사후 비교이며 수익·개입 효과가 아님',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.065,1,1))
    fig.savefig(ROOT/'assets/persona-value-comparison.png',dpi=160);plt.close(fig)
    print('Rendered persona aggregate figure.')

paired_path=ROOT/'results/paired-revalidation.json'
if paired_path.is_file():
    paired=json.loads(paired_path.read_text(encoding='utf-8'))
    keys=['persistence','blend','with_state_history']
    labels=['상태 유지 규칙','내부 거래 기반 블렌딩','현재 상태·건수 이력 추가']
    vals=[paired['results'][k]['f1'] for k in keys]
    fig,ax=plt.subplots(figsize=(10,5.2))
    bars=ax.barh(labels,vals,color=['#197c80','#8b9bab','#5478a8'],height=.52)
    ax.invert_yaxis()
    ax.set(xlim=(0,1.04),xlabel='F1',title='동일 평가 행 재검증: 추가 모델의 우위는 아직 불확실')
    for b,v in zip(bars,vals):ax.text(v+.012,b.get_y()+b.get_height()/2,f'{v:.4f}',va='center',fontsize=12)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ci=paired['paired_corporation_bootstrap']['with_state_history']
    fig.text(.04,.065,f"추가 모델 - 규칙: {ci['delta_f1']:+.4f} · 95% 부트스트랩 구간 [{ci['percentile_95'][0]:+.4f}, {ci['percentile_95'][1]:+.4f}]",fontsize=10)
    fig.text(.04,.025,'새 내부 데이터 실험 · 11,953 법인·월 · 실제 이탈·개입 효과를 평가한 수치가 아님',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.10,1,1))
    fig.savefig(ROOT/'assets/paired-model-comparison.png',dpi=160);plt.close(fig)
    print('Rendered paired model comparison.')
