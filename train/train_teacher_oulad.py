import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from configs.train_config import default_config as cfg
from configs.feature_config import NUMERIC_COLS, CAT_COLS, LABEL_COL, GROUP_COLS
from datasets.snapshot_dataset import SnapshotDataset
from datasets.samplers import GroupBatchSampler
from models.snapshot_risk_model import SnapshotRiskModelFT
from losses.snapshot_losses import monotone_and_smooth_loss
from utils.seed import seed_everything
from utils.metrics import calculate_metrics

def make_dataloaders_oulad(dataset: SnapshotDataset, val_ratio: float = 0.2):
    """
    OULAD SnapshotDataset를 받아서
    - 학생 단위로 Train/Val split
    - GroupBatchSampler까지 적용된 DataLoader 반환
    """
    df = dataset.df
    assert "id_student" in df.columns, "id_student 컬럼이 필요합니다."

    # 1. 학생 단위 분할
    all_students = df["id_student"].unique()
    np.random.shuffle(all_students)

    n_val = int(len(all_students) * val_ratio)
    val_students = set(all_students[:n_val])
    train_students = set(all_students[n_val:])

    idx_all = np.arange(len(df))
    student_ids = df["id_student"].values

    train_indices = [i for i in idx_all if student_ids[i] in train_students]
    val_indices = [i for i in idx_all if student_ids[i] in val_students]

    print(f"[Split] #students train={len(train_students)}, val={len(val_students)}")
    print(f"[Split] #samples  train={len(train_indices)}, val={len(val_indices)}")

    # 2. Subset 생성
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # 3. GroupBatchSampler 생성 (subset 기준 group_ids)
    train_group_ids = dataset.group_ids[train_indices]
    val_group_ids = dataset.group_ids[val_indices]

    train_batch_sampler = GroupBatchSampler(
        group_ids=train_group_ids,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
    )
    val_batch_sampler = GroupBatchSampler(
        group_ids=val_group_ids,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_subset,
        batch_sampler=train_batch_sampler,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_sampler=val_batch_sampler,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_teacher_oulad():
    seed_everything(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 1. Dataset 로드
    dataset = SnapshotDataset(
        csv_path=cfg.OULAD_CSV,
        numeric_cols=NUMERIC_COLS,
        cat_cols=CAT_COLS,
        label_col=LABEL_COL,
        group_cols=GROUP_COLS,
        teacher_prob_col=None,
        fit_encoders=True,
        cat_encoders=None,
    )
    print(f"[Dataset] OULAD snapshots loaded. shape={len(dataset.df)}")

    # 2. Train / Val DataLoader
    train_loader, val_loader = make_dataloaders_oulad(dataset, val_ratio=0.2)

    # 3. 모델 생성
    model = SnapshotRiskModelFT(
        num_numeric=len(NUMERIC_COLS),
        cat_cardinalities=dataset.cat_cardinalities,
        d_token=cfg.D_TOKEN,
        n_heads=cfg.N_HEADS,
        n_layers=cfg.N_LAYERS,
        dim_feedforward=cfg.DIM_FEEDFORWARD,
        dropout=cfg.DROPOUT,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.LR_TEACHER)

    # [변경 1] pos_weight 고정값 (2.5) 적용
    # OULAD 데이터 비율(약 2.5:1)을 고려하여 Positive Class에 가중치 부여
    pos_weight = torch.tensor([2.5]).to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0.0

    # [변경 2] Epoch 증가 (20 -> 100) 및 Early Stopping 변수 설정
    epochs = 100        # 넉넉하게 설정
    patience = 15       # 15 epoch 동안 개선 없으면 중단
    patience_counter = 0 # 카운터 초기화

    print(f"[Training] Start (Max Epochs: {epochs}, Patience: {patience}, pos_weight: 2.5)")

    # 4. 학습 루프
    for epoch in range(1, epochs + 1):
        # ----- Train -----
        model.train()
        train_losses = []
        train_bce_losses = []
        train_reg_losses = []

        for batch in tqdm(train_loader, desc=f"[Epoch {epoch}] Train", leave=False):
            numeric = batch["numeric"].to(device)
            categorical = batch["categorical"].to(device)
            labels = batch["label"].to(device)
            group_ids = batch["group_id"].to(device)
            time_to_deadline = batch["time_to_deadline"].to(device)

            optimizer.zero_grad()
            logits, probs = model(numeric, categorical)

            loss_bce = bce_loss_fn(logits, labels)
            loss_reg = monotone_and_smooth_loss(
                probs=probs,
                group_ids=group_ids,
                time_to_deadline=time_to_deadline,
                lambda_mono=cfg.LAMBDA_MONO,
                lambda_smooth=cfg.LAMBDA_SMOOTH,
            )
            loss = loss_bce + loss_reg

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_bce_losses.append(loss_bce.item())
            train_reg_losses.append(loss_reg.item())

        avg_train_loss = float(np.mean(train_losses))
        avg_train_bce = float(np.mean(train_bce_losses))
        avg_train_reg = float(np.mean(train_reg_losses))

        # ----- Validation -----
        model.eval()
        val_losses = []
        all_val_labels = []
        all_val_logits = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Epoch {epoch}] Val", leave=False):
                numeric = batch["numeric"].to(device)
                categorical = batch["categorical"].to(device)
                labels = batch["label"].to(device)
                group_ids = batch["group_id"].to(device)
                time_to_deadline = batch["time_to_deadline"].to(device)

                logits, probs = model(numeric, categorical)

                loss_bce = bce_loss_fn(logits, labels)
                loss_reg = monotone_and_smooth_loss(
                    probs=probs,
                    group_ids=group_ids,
                    time_to_deadline=time_to_deadline,
                    lambda_mono=cfg.LAMBDA_MONO,
                    lambda_smooth=cfg.LAMBDA_SMOOTH,
                )
                loss = loss_bce + loss_reg

                val_losses.append(loss.item())
                all_val_labels.append(labels)
                all_val_logits.append(logits)

        avg_val_loss = float(np.mean(val_losses))
        all_val_labels = torch.cat(all_val_labels, dim=0)
        all_val_logits = torch.cat(all_val_logits, dim=0)
        metrics = calculate_metrics(all_val_labels, all_val_logits)
        val_auc, val_f1, val_acc = metrics["auc"], metrics["f1"], metrics["acc"]

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"TrainLoss={avg_train_loss:.4f} (BCE={avg_train_bce:.4f}) | "
            f"ValLoss={avg_val_loss:.4f} | "
            f"AUC={val_auc:.4f}, F1={val_f1:.4f} (Counter: {patience_counter}/{patience})"
        )

        # Early Stopping 로직 적용
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0  # 개선되었으므로 카운터 초기화

            # Best Model 저장
            ckpt_path = cfg.TEACHER_MODEL_PATH
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            
            import pickle
            with open(cfg.ENCODER_PATH, "wb") as f:
                pickle.dump(dataset.cat_encoders, f)
            
            print(f"  ↳ [BEST] 갱신 (AUC={best_val_auc:.4f}) -> 저장 완료")
        else:
            patience_counter += 1
            print(f"  ↳ 개선 없음 ({patience_counter}/{patience})")

        # Patience 초과 시 학습 중단
        if patience_counter >= patience:
            print(f"[Early Stopping] {patience} epoch 동안 개선이 없어 학습을 조기 종료합니다.")
            break

    print(f"[Done] Training finished. Best Val AUC = {best_val_auc:.4f}")

if __name__ == "__main__":
    train_teacher_oulad()
