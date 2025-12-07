# utils/metrics.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score, precision_score, recall_score

def calculate_metrics(y_true_tensor, y_logits_tensor, threshold=None):
    """
    threshold가 None이면, 데이터 비율에 맞춰 자동으로 설정하거나 
    AUPRC 같은 Threshold-independent metric 위주로 리턴
    """
    y_true = y_true_tensor.detach().cpu().numpy()
    probs = torch.sigmoid(y_logits_tensor).detach().cpu().numpy()
    
    # 1. ROC-AUC (일반적인 성능)
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = 0.5

    # 2. AUPRC (불균형 데이터 핵심 지표)
    try:
        auprc = average_precision_score(y_true, probs)
    except ValueError:
        auprc = 0.0

    # 3. Hard Prediction (Threshold 조정)
    # 지각률이 4.6%라면, 상위 10~20% 정도를 위험군으로 보는 게 맞음 -> Threshold 0.2
    if threshold is None:
        threshold = 0.2 
    
    preds = (probs > threshold).astype(int)

    f1 = f1_score(y_true, preds, zero_division=0)
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)

    return {
        "auc": auc, 
        "auprc": auprc,
        "f1": f1, 
        "acc": acc,
        "prec": prec,
        "rec": rec
    }
