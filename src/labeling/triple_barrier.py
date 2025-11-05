import numpy as np
import pandas as pd

def triple_barrier(
    df: pd.DataFrame,
    atr_col: str = "atr_14",
    up_mult: float = 1.5,
    dn_mult: float = 1.0,
    max_holding: int = 20,
    high: str = "high",
    low: str = "low",
    close: str = "close",
    conservative_tie_to_sl: bool = True,
) -> pd.DataFrame:
    """
    트리플 배리어 라벨링:
      +1 : 상단(익절) 장벽이 먼저 도달
      -1 : 하단(손절) 장벽이 먼저 도달
       0 : max_holding(수직장벽) 내 어느 쪽도 도달 못함

    주의: 일봉만 있을 때 같은 날 high>=TP & low<=SL 이 동시에 가능한데,
         일중 순서를 알 수 없으므로 tie 규칙이 필요하다.
         conservative_tie_to_sl=True 이면 보수적으로 -1(손절 우선) 처리.

    반환 컬럼:
      label(Int8), t_hit(int, 이벤트 종료 인덱스), tp(float), sl(float), event_end(datetime64[ns])
    """
    idx = df.index
    n = len(df)
    label = np.zeros(n, dtype=np.int8)
    t_hit = np.full(n, -1, dtype=np.int32)

    c = df[close].to_numpy(dtype=float).reshape(-1)
    h = df[high].to_numpy(dtype=float).reshape(-1)
    l = df[low].to_numpy(dtype=float).reshape(-1)
    atr = df[atr_col].to_numpy(dtype=float).reshape(-1)

    tp = c + up_mult * atr
    sl = c - dn_mult * atr

    for i in range(n):
        if not np.isfinite(atr[i]):
            label[i] = 0
            t_hit[i] = i
            continue

        end = min(n - 1, i + max_holding)
        if end <= i:
            label[i] = 0
            t_hit[i] = end
            continue

        rng = slice(i + 1, end + 1)
        up_idx_arr = np.where(h[rng] >= tp[i])[0]
        dn_idx_arr = np.where(l[rng] <= sl[i])[0]

        up_idx = (i + 1 + up_idx_arr[0]) if up_idx_arr.size else -1
        dn_idx = (i + 1 + dn_idx_arr[0]) if dn_idx_arr.size else -1

        if up_idx == -1 and dn_idx == -1:
            label[i] = 0
            t_hit[i] = end
        elif up_idx != -1 and dn_idx == -1:
            label[i] = +1
            t_hit[i] = up_idx
        elif up_idx == -1 and dn_idx != -1:
            label[i] = -1
            t_hit[i] = dn_idx
        else:
            # 같은 캔들에서 둘 다 터진 경우(일봉만 있을 때 순서 모름)
            if up_idx < dn_idx:
                label[i] = +1; t_hit[i] = up_idx
            elif dn_idx < up_idx:
                label[i] = -1; t_hit[i] = dn_idx
            else:  # 완전 동시(같은 인덱스)
                if conservative_tie_to_sl:
                    label[i] = -1; t_hit[i] = dn_idx
                else:
                    label[i] = +1; t_hit[i] = up_idx

    event_end = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns]")
    valid = (t_hit >= 0) & (t_hit < n)
    event_end.iloc[np.where(valid)[0]] = idx[t_hit[valid]]

    out = pd.DataFrame(
        {
            "label": pd.Series(label, index=idx, dtype="Int8"),
            "t_hit": pd.Series(t_hit, index=idx, dtype="int32"),
            "tp": pd.Series(tp, index=idx),
            "sl": pd.Series(sl, index=idx),
            "event_end": event_end,
        }
    )
    return out
