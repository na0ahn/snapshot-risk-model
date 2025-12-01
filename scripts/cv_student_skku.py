# scripts/cv_student_skku.py

import os
import sys
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

# 루트 경로 추가
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from datasets.snapshot_dataset import SnapshotDataset
from datasets.samplers import GroupBatchSampler
from models.snapshot_risk_model import SnapshotRiskModelFT
from losses.snapshot_losses import binary_kl_div, monotone_and_smooth_loss
from configs.feature_config import NUMERIC_COLS, CAT_COLS, LABEL_COL, GROUP_COLS
from utils.metrics import calculate_metrics
from utils.seed import seed_everything

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SKKU_SNAPSHOT_WITH_TEACHER = os.path.join("data", "skku", "processed", "skku_snapshots_with_teacher.csv")
ENCODER_PATH = os.path.join("checkpoints", "cat_encoders.pkl")

def train_one_fold(train_indices, test_indices, full_dataset, teacher_encoders,
                   d_token=32, n_heads=4, n_layers=3,
                   lr=5e-4, batch_size=256, epochs=10,
                   lambda_kl=0.5, lambda_mono=0.1, lambda_smooth=0.1):

    # --- Subset 만들기 ---
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset  = Subset(full_dataset, test_indices)

    # GroupBatchSampler에는 train 쪽 group_ids만 전달
    full_group_ids = np.array(full_dataset.group_ids)  # SnapshotDataset 안에 있다고 가정
    train_group_ids = full_group_ids[train_indices]

    sampler = GroupBatchSampler(train_group_ids, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- 모델 구성 (Teacher encoders 기준으로 cardinality 고정) ---
    cat_cardinalities = [len(teacher_encoders[c]) for c in CAT_COLS]

    model = SnapshotRiskModelFT(
        num_numeric=len(NUMERIC_COLS),
        cat_cardinalities=cat_cardinalities,
        d_token=d_token,
        n_heads=n_heads,
        n_layers=n_layers
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_task = nn.BCEWithLogitsLoss()

    # --- 학습 ---
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            numeric = batch["numeric"].to(DEVICE)
            categorical = batch["categorical"].to(DEVICE)
            labels = batch["label"].to(DEVICE).float()
            group_ids = batch["group_id"].to(DEVICE)
            ttd = batch["time_to_deadline"].to(DEVICE)
            teacher_probs = batch["teacher_prob"].to(DEVICE)

            # Teacher cardinality 범위 밖 인덱스는 UNK(마지막)로 clamp
            for i, card in enumerate(cat_cardinalities):
                categorical[:, i] = categorical[:, i].clamp(max=card - 1)

            optimizer.zero_grad()
            logits, probs = model(numeric, categorical)

            loss_task = criterion_task(logits, labels)

            # KL (teacher_prob가 NaN이 아닐 때만)
            mask_kl = ~torch.isnan(teacher_probs)
            if mask_kl.any():
                loss_kl = binary_kl_div(probs[mask_kl], teacher_probs[mask_kl]).mean()
            else:
                loss_kl = torch.tensor(0.0, device=DEVICE)

            loss_reg = monotone_and_smooth_loss(
                probs, group_ids, ttd,
                lambda_mono=lambda_mono,
                lambda_smooth=lambda_smooth
            )

            loss = loss_task + lambda_kl * loss_kl + loss_reg
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"    [Epoch {epoch+1}] train loss = {avg_loss:.4f}")

    # --- 평가 ---
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            numeric = batch["numeric"].to(DEVICE)
            categorical = batch["categorical"].to(DEVICE)
            labels = batch["label"].to(DEVICE).float()

            for i, card in enumerate(cat_cardinalities):
                categorical[:, i] = categorical[:, i].clamp(max=card - 1)

            logits, _ = model(numeric, categorical)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = calculate_metrics(all_labels, all_logits)
    return metrics


def run_cv(k_folds: int = 4, seed: int = 42):
    print(f"[Device] {DEVICE}")
    seed_everything(seed)

    # --- Teacher encoders 로드 ---
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder 파일 없음: {ENCODER_PATH}")
    with open(ENCODER_PATH, "rb") as f:
        teacher_encoders = pickle.load(f)

    # --- 전체 Dataset 로드 (한 번만) ---
    full_dataset = SnapshotDataset(
        csv_path=SKKU_SNAPSHOT_WITH_TEACHER,
        numeric_cols=NUMERIC_COLS,
        cat_cols=CAT_COLS,
        label_col=LABEL_COL,
        group_cols=GROUP_COLS,
        teacher_prob_col="teacher_prob",
        fit_encoders=False,
        cat_encoders=teacher_encoders
    )

    df = full_dataset.df  # SnapshotDataset 안에 pandas DataFrame이 있다고 가정
    unique_students = df["id_student"].unique()
    print(f"[Info] n_students = {len(unique_students)}, n_samples = {len(full_dataset)}")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    fold_results = []
    for fold_idx, (train_stu_idx, test_stu_idx) in enumerate(kf.split(unique_students), start=1):
        train_ids = set(unique_students[train_stu_idx])
        test_ids  = set(unique_students[test_stu_idx])

        # 각 snapshot의 인덱스를 학생 ID 기준으로 분할
        all_student_ids = df["id_student"].values
        train_indices = [i for i, sid in enumerate(all_student_ids) if sid in train_ids]
        test_indices  = [i for i, sid in enumerate(all_student_ids) if sid in test_ids]

        print(f"\n[Fold {fold_idx}/{k_folds}] "
              f"train_students={len(train_ids)}, test_students={len(test_ids)}, "
              f"train_samples={len(train_indices)}, test_samples={len(test_indices)}")

        metrics = train_one_fold(
            train_indices, test_indices, full_dataset, teacher_encoders
        )
        print(f"  ↳ Fold {fold_idx} metrics: AUC={metrics['auc']:.4f}, "
              f"F1={metrics['f1']:.4f}, Acc={metrics['acc']:.4f}")
        fold_results.append(metrics)

    # --- Fold 평균/표준편차 요약 ---
    avg = defaultdict(float)
    std = defaultdict(float)
    for key in fold_results[0].keys():
        vals = np.array([m[key] for m in fold_results])
        avg[key] = vals.mean()
        std[key] = vals.std()

    print("\n[CV Summary]")
    for k in avg.keys():
        print(f"  {k.upper()}: {avg[k]:.4f} ± {std[k]:.4f}")

if __name__ == "__main__":
    run_cv(k_folds=4, seed=42)
