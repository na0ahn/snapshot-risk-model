# losses/snapshot_losses.py
import torch
import torch.nn.functional as F

def binary_kl_div(p, q, eps=1e-7):
    """
    Teacher(q)와 Student(p) 간의 KL Divergence
    D_KL(q || p) = q * log(q/p) + (1-q) * log((1-q)/(1-p))
    """
    p = torch.clamp(p, eps, 1 - eps)
    q = torch.clamp(q, eps, 1 - eps)
    return (q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p)))

def monotone_and_smooth_loss(
    probs: torch.Tensor,
    group_ids: torch.Tensor,
    time_to_deadline: torch.Tensor,
    lambda_mono: float = 1.0,
    lambda_smooth: float = 1.0,
):
    """
    Vectorized Snapshot Regularization (No Loop)
    - Monotonicity: 마감에 가까워질수록(time_to_deadline이 작을수록) 위험도(prob)가 급격히 줄어드는 것을 방지
    - Smoothness: 인접 시점 간 급격한 변화 방지
    """
    device = probs.device
    if lambda_mono == 0 and lambda_smooth == 0:
        return torch.tensor(0.0, device=device)

    # 1. 정렬 키 생성 (Group 우선, 그 다음 Time Descending)
    # Scale을 곱해 group_id가 섞이지 않게 함
    # Time 내림차순: 7일전 -> 3일전 -> 0일전 순서로 정렬됨
    max_time = 10000 
    sort_keys = group_ids.long() * max_time + time_to_deadline.long()
    sorted_idx = torch.argsort(sort_keys, descending=True)
    
    p_sorted = probs[sorted_idx]
    g_sorted = group_ids[sorted_idx]
    
    # 2. 인접 시점 차이 계산 (Past - Future)
    diffs = p_sorted[:-1] - p_sorted[1:]
    
    # 3. 같은 그룹끼리만 비교하도록 마스킹
    valid_mask = (g_sorted[:-1] == g_sorted[1:])
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    valid_diffs = diffs[valid_mask]

    # [Monotone Logic]
    # 상식: 시간이 지나도 아무것도 안 하면 위험도는 증가하거나 유지되어야 함 (Past <= Future)
    # 즉, p_past - p_future <= 0 이어야 정상.
    # 만약 p_past - p_future > 0 (과거엔 위험했는데 갑자기 안전해짐) -> Penalty (노력 없이)
    # *주의: 학생이 '제출'을 해버리면 위험도가 0이 되므로 정당한 감소임.
    # 하지만 여기서는 '미제출 상태'에서의 경향성을 학습하려는 의도가 큼.
    # (실험적으로는 F.relu(valid_diffs)가 '갑작스런 안심'을 방지하는 효과가 있음)
    loss_mono = F.relu(valid_diffs).mean() # 양수일 때(위험도 감소)만 페널티
    
    # [Smooth Logic]
    # 변화량이 너무 크면 페널티
    loss_smooth = (valid_diffs ** 2).mean()

    return (lambda_mono * loss_mono) + (lambda_smooth * loss_smooth)