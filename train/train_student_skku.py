# train/train_student_skku.py

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from datasets.snapshot_dataset import SnapshotDataset
from datasets.samplers import GroupBatchSampler
from models.snapshot_risk_model import SnapshotRiskModelFT
from configs.feature_config import NUMERIC_COLS, CAT_COLS, LABEL_COL, GROUP_COLS
from utils.metrics import calculate_metrics
from utils.seed import seed_everything
from losses.snapshot_losses import binary_kl_div, monotone_and_smooth_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SKKU_SNAPSHOT_WITH_TEACHER = os.path.join("data", "skku", "processed", "skku_snapshots_with_teacher.csv")
ENCODER_PATH = os.path.join("checkpoints", "cat_encoders.pkl")
STUDENT_CKPT = os.path.join("checkpoints", "student_skku.pt")

def train_student_skku(
    batch_size: int = 256,
    lr: float = 5e-4,
    epochs: int = 10,
    lambda_kl: float = 0.5,
    lambda_mono: float = 0.1,
    lambda_smooth: float = 0.1,
    val_ratio: float = 0.2,
    seed: int = 42,
):

    print(f"[Device] {DEVICE}")
    seed_everything(seed)

    # 1. Encoders 로드 (Teacher 기준)
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"[ERR] {ENCODER_PATH} 없음 (Teacher 먼저 학습 필요)")
    with open(ENCODER_PATH, "rb") as f:
        teacher_encoders = pickle.load(f)

    teacher_cardinalities = {c: len(teacher_encoders[c]) for c in CAT_COLS}
    print(f"[Info] Teacher cardinalities: {teacher_cardinalities}")

    # 2. 전체 SKKU Dataset 로드 (teacher_prob 포함)
    if not os.path.exists(SKKU_SNAPSHOT_WITH_TEACHER):
        raise FileNotFoundError(f"[ERR] {SKKU_SNAPSHOT_WITH_TEACHER} 없음. "
                                f"먼저 Teacher inference를 수행하세요.")
    
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
    print(f"[Dataset] SKKU snapshots(with teacher) loaded. n_samples = {len(full_dataset)}")

    df = full_dataset.df
    student_ids = df["id_student"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(student_ids)

    n_students = len(student_ids)
    n_val = int(n_students * val_ratio)
    val_students = set(student_ids[:n_val])
    train_students = set(student_ids[n_val:])

    train_indices = np.where(df["id_student"].isin(train_students))[0]
    val_indices = np.where(df["id_student"].isin(val_students))[0]

    print(f"[Split] #students train={len(train_students)}, val={len(val_students)}")
    print(f"[Split] #samples  train={len(train_indices)}, val={len(val_indices)}")

    # 3. Subset Dataset 생성
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # 4. GroupBatchSampler는 "train subset" 기준 group_ids 필요
    train_group_ids = full_dataset.group_ids[train_indices]
    train_sampler = GroupBatchSampler(train_group_ids, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)

    # val은 그냥 sequential loader
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 5. Student 모델 (Teacher encoder 기준 cardinality 사용)
    cat_cardinalities = [teacher_cardinalities[c] for c in CAT_COLS]

    model = SnapshotRiskModelFT(
        num_numeric=len(NUMERIC_COLS),
        cat_cardinalities=cat_cardinalities,
        d_token=32,
        n_heads=4,
        n_layers=3,
        dim_feedforward=128,
        dropout=0.1
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_task = nn.BCEWithLogitsLoss()

    print("[Start] Student training...")

    best_val_auc = 0.0

    for epoch in range(1, epochs + 1):
        # ---------- Train ----------
        model.train()
        total_loss = total_task = total_kl = total_reg = 0.0
        n_batches = 0

        for batch in train_loader:
            numeric = batch["numeric"].to(DEVICE)
            categorical = batch["categorical"].to(DEVICE)
            labels = batch["label"].to(DEVICE).float()
            group_ids = batch["group_id"].to(DEVICE)
            ttd = batch["time_to_deadline"].to(DEVICE)
            teacher_probs = batch["teacher_prob"].to(DEVICE)

            # 범주 인덱스 클램핑 (UNK 포함)
            for i, card in enumerate(cat_cardinalities):
                categorical[:, i] = categorical[:, i].clamp(max=card - 1)

            optimizer.zero_grad()
            logits, probs = model(numeric, categorical)

            # Task loss
            loss_task = criterion_task(logits, labels)

            # KL loss (Teacher distillation, NaN 마스킹)
            mask_kl = ~torch.isnan(teacher_probs)
            if mask_kl.any():
                loss_kl = binary_kl_div(probs[mask_kl], teacher_probs[mask_kl]).mean()
            else:
                loss_kl = torch.tensor(0.0, device=DEVICE)

            # Snapshot regularization
            loss_reg = monotone_and_smooth_loss(
                probs, group_ids, ttd,
                lambda_mono=lambda_mono,
                lambda_smooth=lambda_smooth
            )

            loss = loss_task + lambda_kl * loss_kl + loss_reg
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_task += loss_task.item()
            total_kl += loss_kl.item()
            total_reg += loss_reg.item()
            n_batches += 1

        avg_total = total_loss / n_batches
        avg_task = total_task / n_batches
        avg_kl = total_kl / n_batches
        avg_reg = total_reg / n_batches

        # ---------- Validation ----------
        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                numeric = batch["numeric"].to(DEVICE)
                categorical = batch["categorical"].to(DEVICE)
                labels = batch["label"].to(DEVICE).float()

                for i, card in enumerate(cat_cardinalities):
                    categorical[:, i] = categorical[:, i].clamp(max=card - 1)

                logits, _ = model(numeric, categorical)
                all_logits.append(logits)
                all_labels.append(labels)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = calculate_metrics(all_labels, all_logits)
        val_auc, val_f1, val_acc = metrics["auc"], metrics["f1"], metrics["acc"]

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"TrainTotal={avg_total:.4f} (Task={avg_task:.4f}, KL={avg_kl:.4f}, Reg={avg_reg:.4f}) "
            f"| Val AUC={val_auc:.4f}, F1={val_f1:.4f}, Acc={val_acc:.4f}"
        )

        # 베스트 모델 저장
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            os.makedirs(os.path.dirname(STUDENT_CKPT), exist_ok=True)
            torch.save(model.state_dict(), STUDENT_CKPT)
            print(f"  ↳ [BEST] 모델 갱신! (Val AUC={best_val_auc:.4f}) → {STUDENT_CKPT}")

    print(f"[Done] Training finished. Best Val AUC = {best_val_auc:.4f}")


if __name__ == "__main__":
    train_student_skku()
