# scripts/eval_student_skku.py

import os
import sys
import pickle

import torch
import pandas as pd
from torch.utils.data import DataLoader

# 프로젝트 루트 경로 추가
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from datasets.snapshot_dataset import SnapshotDataset
from models.snapshot_risk_model import SnapshotRiskModelFT
from configs.feature_config import NUMERIC_COLS, CAT_COLS, LABEL_COL, GROUP_COLS

# 만약 utils/metrics.py 이미 만들어뒀다면 이걸 사용
try:
    from utils.metrics import calculate_metrics
    USE_UTIL_METRICS = True
except ImportError:
    USE_UTIL_METRICS = False
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    def calculate_metrics(y_true_tensor, y_logits_tensor):
        y_true = y_true_tensor.detach().cpu().numpy()
        probs = torch.sigmoid(y_logits_tensor).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)

        try:
            auc = roc_auc_score(y_true, probs)
        except ValueError:
            auc = 0.5
        f1 = f1_score(y_true, preds, zero_division=0)
        acc = accuracy_score(y_true, preds)
        return {"auc": auc, "f1": f1, "acc": acc}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SKKU_SNAPSHOT_WITH_TEACHER = os.path.join(
    "data", "skku", "processed", "skku_snapshots_with_teacher.csv"
)
ENCODER_PATH = os.path.join("checkpoints", "cat_encoders.pkl")
STUDENT_CHECKPOINT_PATH = os.path.join("checkpoints", "student_skku.pt")


def eval_student_skku():
    print(f"[Device] {DEVICE}")

    # 1. Encoders 로드 (Teacher와 동일 인코더 사용)
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"[ERR] 인코더 파일이 없습니다: {ENCODER_PATH}")

    with open(ENCODER_PATH, "rb") as f:
        cat_encoders = pickle.load(f)

    cat_cardinalities = [len(cat_encoders[c]) for c in CAT_COLS]
    print(f"[Info] Cardinalities: {dict(zip(CAT_COLS, cat_cardinalities))}")

    # 2. SKKU snapshot + teacher_prob CSV 로드
    if not os.path.exists(SKKU_SNAPSHOT_WITH_TEACHER):
        raise FileNotFoundError(
            f"[ERR] {SKKU_SNAPSHOT_WITH_TEACHER} 가 없습니다.\n"
            "먼저 scripts/inference_teacher_on_skku.py 를 실행해서 teacher_prob를 생성하세요."
        )

    dataset = SnapshotDataset(
        csv_path=SKKU_SNAPSHOT_WITH_TEACHER,
        numeric_cols=NUMERIC_COLS,
        cat_cols=CAT_COLS,
        label_col=LABEL_COL,
        group_cols=GROUP_COLS,
        teacher_prob_col="teacher_prob",  # 읽기만 하고 사용 안 할 수도 있음
        fit_encoders=False,
        cat_encoders=cat_encoders,
    )

    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,   # 전체 평가이므로 셔플 X
        num_workers=0,
    )

    print(f"[Dataset] n_samples = {len(dataset)}")

    # 3. Student 모델 로드
    if not os.path.exists(STUDENT_CHECKPOINT_PATH):
        raise FileNotFoundError(f"[ERR] Student 체크포인트가 없습니다: {STUDENT_CHECKPOINT_PATH}")

    model = SnapshotRiskModelFT(
        num_numeric=len(NUMERIC_COLS),
        cat_cardinalities=cat_cardinalities,
        d_token=32,
        n_heads=4,
        n_layers=3,
        dim_feedforward=128,
        dropout=0.1,
    ).to(DEVICE)

    state = torch.load(STUDENT_CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print("[OK] Student 모델 로드 완료")

    # 4. 전체 스냅샷에 대한 예측 수집
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            numeric = batch["numeric"].to(DEVICE)
            categorical = batch["categorical"].to(DEVICE)

            # index 범위 clamp (UNK 방어)
            for i, card in enumerate(cat_cardinalities):
                categorical[:, i] = categorical[:, i].clamp(max=card - 1)

            labels = batch["label"].to(DEVICE).float()
            logits, probs = model(numeric, categorical)

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 5. 메트릭 계산 (AUC, F1, Acc)
    metrics = calculate_metrics(all_labels, all_logits)
    print("\n[Student on SKKU snapshots]")
    print(f"  AUC = {metrics['auc']:.4f}")
    print(f"  F1  = {metrics['f1']:.4f}")
    print(f"  Acc = {metrics['acc']:.4f}")


if __name__ == "__main__":
    eval_student_skku()
