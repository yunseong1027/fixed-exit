import numpy as np

def sigmoid(x): return 1/(1+np.exp(-x))

def size_riskaware(q_mae, e_r, B_t, theta, kappa=0.5, f_max=0.2, f_total=0.6,
                   scale_er=None, eps=1e-4):
    """
    q_mae, e_r : np.ndarray (테스트 타임라인 기준, 후보일 순서와 동일)
    B_t        : float or np.ndarray (잔여 DD 예산; 계정 레벨에서 갱신 가능)
    theta      : 기대수익 임계(절대 bp 또는 상대 퍼센타일에서 선정값)
    kappa      : 민감도
    f_max      : 트레이드별 최대 비중
    f_total    : 동시보유 총합 상한
    """
    if scale_er is None:
        # 검증 구간 E[R]의 IQR 등으로 스케일 잡는 것을 권장
        scale_er = max(1e-4, np.nanstd(e_r))

    # 품질 스케일러: (E_R - theta) 양수일 때만 크게
    s_er = sigmoid((e_r - theta) / (scale_er + 1e-8))
    raw = kappa * (B_t / (q_mae + eps)) * s_er
    f = np.clip(raw, 0.0, f_max)
    # 동시보유 총합 제한은 시뮬레이터에서 하루 단위로 적용(Σ f_i ≤ f_total)
    return f
