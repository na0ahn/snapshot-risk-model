# utils/metrics.py
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def calculate_metrics(y_true_tensor, y_logits_tensor):
    y_true = y_true_tensor.detach().cpu().numpy()
    probs = torch.sigmoid(y_logits_tensor).detach().cpu().numpy()
    preds = (probs > 0.5).astype(int)

    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = 0.5  # 한 클래스만 있을 경우

    f1 = f1_score(y_true, preds, zero_division=0)
    acc = accuracy_score(y_true, preds)

    return {"auc": auc, "f1": f1, "acc": acc}
