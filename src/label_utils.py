"""Original cohort/label functions. Source cell numbers are zero-based. No file I/O."""

import re
import numpy as np
import pandas as pd
from scipy.stats import norm
EPS=1e-6
RUNS_ALPHA=.05
GAP_ALPHA=.01


# Original cell 4
def make_obs_vector(corp_df, all_months):
    actual_months = set(corp_df['기준년월'])
    return [1 if m in actual_months else 0 for m in all_months]

# Original cell 4
def runs_test_binary(obs_vector):
    """
    H0: 관측(1)과 누락(0)은 랜덤하게 배열되어 있다.
    H1: 관측/누락이 군집되어 있다.
    단측검정: runs가 기대보다 유의하게 적은 방향.
    """
    x = np.array([int(v) for v in obs_vector])
    n1 = np.sum(x == 1)
    n0 = np.sum(x == 0)

    if n0 == 0 or n1 == 0:
        return {
            'runs': np.nan,
            'expected_runs': np.nan,
            'var_runs': np.nan,
            'z_runs': np.nan,
            'p_runs_lower': np.nan,
            'runs_result': '검정불가'
        }

    runs = 1 + np.sum(x[1:] != x[:-1])
    n = len(x)
    expected_runs = 1 + (2 * n1 * n0) / n
    var_runs = ((2 * n1 * n0) * (2 * n1 * n0 - n)) / (n**2 * (n - 1))

    if var_runs <= 0:
        return {
            'runs': runs,
            'expected_runs': expected_runs,
            'var_runs': var_runs,
            'z_runs': np.nan,
            'p_runs_lower': np.nan,
            'runs_result': '검정불가'
        }

    z = (runs - expected_runs) / np.sqrt(var_runs)
    p_lower = norm.cdf(z)
    result = '비랜덤_군집누락' if (p_lower < RUNS_ALPHA and z < 0) else '랜덤누락_가능'

    return {
        'runs': runs,
        'expected_runs': expected_runs,
        'var_runs': var_runs,
        'z_runs': z,
        'p_runs_lower': p_lower,
        'runs_result': result
    }

# Original cell 4
def build_corp_profile(df, all_months, q_miss):
    N_MONTHS = len(all_months)
    FIRST_MONTH = all_months[0]
    LAST_MONTH = all_months[-1]

    rows = []

    for corp_id, corp_df in df.groupby('법인ID'):
        obs_vector_list = make_obs_vector(corp_df, all_months)
        obs_vector = ''.join(map(str, obs_vector_list))

        month_cnt = sum(obs_vector_list)
        total_gap = N_MONTHS - month_cnt

        if month_cnt == 0:
            continue

        first_idx = obs_vector_list.index(1)
        last_idx = len(obs_vector_list) - 1 - obs_vector_list[::-1].index(1)

        first_month = all_months[first_idx]
        last_month = all_months[last_idx]
        leading_gap = first_idx
        trailing_gap = N_MONTHS - 1 - last_idx

        # 중간 누락 구간
        internal_gaps = []
        current_gap = 0
        for i in range(first_idx, last_idx + 1):
            if obs_vector_list[i] == 0:
                current_gap += 1
            else:
                if current_gap > 0:
                    internal_gaps.append(current_gap)
                current_gap = 0
        if current_gap > 0:
            internal_gaps.append(current_gap)

        max_internal_gap = max(internal_gaps, default=0)
        total_internal_gap = sum(internal_gaps)
        internal_gap_cnt = len(internal_gaps)

        # 전체 최대 연속 누락
        all_gaps = []
        current_gap = 0
        for v in obs_vector_list:
            if v == 0:
                current_gap += 1
            else:
                if current_gap > 0:
                    all_gaps.append(current_gap)
                current_gap = 0
        if current_gap > 0:
            all_gaps.append(current_gap)

        max_gap = max(all_gaps, default=0)
        gap_cnt = len(all_gaps)

        runs_info = runs_test_binary(obs_vector_list)

        rows.append({
            '법인ID': corp_id,
            'obs_vector': obs_vector,
            'month_cnt': month_cnt,
            'total_gap': total_gap,
            'gap_cnt': gap_cnt,
            'max_gap': max_gap,
            'first_month': first_month,
            'last_month': last_month,
            'leading_gap': leading_gap,
            'trailing_gap': trailing_gap,
            'internal_gap_cnt': internal_gap_cnt,
            'max_internal_gap': max_internal_gap,
            'total_internal_gap': total_internal_gap,
            **runs_info
        })

    prof = pd.DataFrame(rows)

    def classify_life_type(row):
        first = row['first_month']
        last = row['last_month']
        max_internal_gap = row['max_internal_gap']

        if first == FIRST_MONTH and last == LAST_MONTH:
            return '완전관측형' if max_internal_gap == 0 else '중간누락형'
        elif first > FIRST_MONTH and last == LAST_MONTH:
            return '신규유입후보'
        elif first == FIRST_MONTH and last < LAST_MONTH:
            return '중도종료후보'
        elif first > FIRST_MONTH and last < LAST_MONTH:
            return '불완전관측형'
        return '기타'

    prof['life_type'] = prof.apply(classify_life_type, axis=1)

    prof['p_leading_gap'] = q_miss ** prof['leading_gap']
    prof['p_trailing_gap'] = q_miss ** prof['trailing_gap']
    prof.loc[prof['leading_gap'] == 0, 'p_leading_gap'] = 1
    prof.loc[prof['trailing_gap'] == 0, 'p_trailing_gap'] = 1

    prof['missing_randomness'] = np.where(
        prof['runs_result'] == '비랜덤_군집누락',
        '비랜덤_군집누락',
        '랜덤누락_가능'
    )

    def final_status(row):
        lt = row['life_type']
        lead_sig = row['p_leading_gap'] < GAP_ALPHA
        trail_sig = row['p_trailing_gap'] < GAP_ALPHA
        internal_nonrandom = row['runs_result'] == '비랜덤_군집누락'

        if lt == '완전관측형':
            return '완전관측_정상'

        if lt == '신규유입후보':
            return '신규유입' if lead_sig else '초반단기누락_보간후보'

        if lt == '중도종료후보':
            return '중도종료_이탈후보' if trail_sig else '후반단기누락_보간후보'

        if lt == '중간누락형':
            return '중간군집누락_비활동후복귀후보' if internal_nonrandom else '중간랜덤누락_보간후보'

        if lt == '불완전관측형':
            if lead_sig and trail_sig:
                return '신규유입후_중도종료후보'
            elif lead_sig and not trail_sig:
                return '신규유입후_후반단기누락보간후보'
            elif not lead_sig and trail_sig:
                return '초반단기누락보간후_중도종료후보'
            else:
                return '초반후반_단기누락보간후보'

        return '기타'

    prof['final_status'] = prof.apply(final_status, axis=1)
    return prof

# Original cell 6
def parse_count_str(s):
    if pd.isna(s) or str(s) in ['', '-']: return 0
    s = str(s)
    e = re.match(r'^(\d+)[개건]$', s)
    if e: return int(e.group(1))
    b = re.match(r'^(\d+)[개건]초과\s*(\d+)[개건]이하$', s)
    if b: return int(b.group(1)) + 1
    o = re.match(r'^(\d+)[개건]\s*초과$', s)
    if o: return int(o.group(1)) + 1
    return 0

# Original cell 8
def compute_rolling_change(series, past=3, recent=3):
    n = len(series)
    M = np.full(n, np.nan)
    for i in range(past + recent, n):
        ps = series.iloc[i-past-recent : i-recent].sum()   # 이전 3개월
        rs = series.iloc[i-recent : i].sum()               # 최근 3개월
        M[i] = (rs - ps) / (ps + EPS)
    return M

# Original cell 8
def rolling_cv(series, window=6):
    m = series.rolling(window, min_periods=3).mean()
    sd = series.rolling(window, min_periods=3).std()
    return sd / (m + EPS)
