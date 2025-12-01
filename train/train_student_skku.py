import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle

from datasets.snapshot_dataset import SnapshotDataset
from datasets.samplers import GroupBatchSampler
from models.snapshot_risk_model import SnapshotRiskModelFT
from losses.snapshot_losses import binary_kl_div, monotone_and_smooth_loss
from configs.feature_config import NUMERIC_COLS, CAT_COLS, LABEL_COL, GROUP_COLS

def train_student():
    BATCH_SIZE = 256
    LR = 5e-4
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters for Loss
    LAMBDA_KL = 0.5
    LAMBDA_MONO = 0.1
    LAMBDA_SMOOTH = 0.1

    # 1. Load Encoders (from Teacher)
    try:
        with open("cat_encoders.pkl", "rb") as f:
            teacher_encoders = pickle.load(f)
    except FileNotFoundError:
        print("Error: 'cat_encoders.pkl' not found. Train Teacher first.")
        return

    # 2. Dataset (Teacher's encoders 적용)
    # Student 데이터에는 'teacher_prob' 컬럼이 있다고 가정 (없으면 NaN)
    dataset = SnapshotDataset(
        csv_path="data/skku_snapshots.csv", 
        numeric_cols=NUMERIC_COLS,
        cat_cols=CAT_COLS,
        label_col=LABEL_COL,
        group_cols=GROUP_COLS,
        teacher_prob_col="teacher_prob", 
        fit_encoders=False,
        cat_encoders=teacher_encoders
    )

    sampler = GroupBatchSampler(dataset.group_ids, batch_size=BATCH_SIZE)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    # 3. Model
    model = SnapshotRiskModelFT(
        num_numeric=len(NUMERIC_COLS),
        cat_cardinalities=dataset.cat_cardinalities,
        d_token=32, n_heads=4, n_layers=3
    ).to(DEVICE)
    
    # (선택) Teacher Weight로 초기화하려면:
    # model.load_state_dict(torch.load("teacher_oulad.pt"))

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_task = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            numeric = batch["numeric"].to(DEVICE)
            categorical = batch["categorical"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            # For Regularization
            group_ids = batch["group_id"].to(DEVICE)
            ttd = batch["time_to_deadline"].to(DEVICE)
            teacher_probs = batch["teacher_prob"].to(DEVICE) # NaN 포함 가능

            optimizer.zero_grad()
            logits, probs = model(numeric, categorical)

            # (1) Task Loss
            loss_task = criterion_task(logits, labels)

            # (2) KL Loss (Teacher Prob이 있는 샘플만)
            mask_kl = ~torch.isnan(teacher_probs)
            if mask_kl.any():
                loss_kl = binary_kl_div(probs[mask_kl], teacher_probs[mask_kl]).mean()
            else:
                loss_kl = torch.tensor(0.0, device=DEVICE)

            # (3) Temporal Regularization (Monotone + Smooth)
            # GroupSampler 덕분에 같은 그룹이 한 배안에 있음이 보장됨
            loss_reg = monotone_and_smooth_loss(
                probs, group_ids, ttd, 
                lambda_mono=LAMBDA_MONO, 
                lambda_smooth=LAMBDA_SMOOTH
            )

            # Final Loss
            loss = loss_task + (LAMBDA_KL * loss_kl) + loss_reg
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Total Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "student_skku.pt")
    print("Student Model Saved!")

if __name__ == "__main__":
    train_student()