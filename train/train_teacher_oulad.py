import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle # 인코더 저장용

from datasets.snapshot_dataset import SnapshotDataset
from datasets.samplers import GroupBatchSampler
from models.snapshot_risk_model import SnapshotRiskModelFT
from configs.feature_config import NUMERIC_COLS, CAT_COLS, LABEL_COL, GROUP_COLS

def train_teacher():
    # 설정
    BATCH_SIZE = 256
    LR = 1e-3
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")

    # 1. Dataset & DataLoader
    # Teacher는 fit_encoders=True로 스스로 인코더를 만듦
    dataset = SnapshotDataset(
        csv_path="data/oulad_snapshots.csv",
        numeric_cols=NUMERIC_COLS,
        cat_cols=CAT_COLS,
        label_col=LABEL_COL,
        group_cols=GROUP_COLS,
        fit_encoders=True 
    )
    
    # Teacher도 GroupSampler를 쓰면 학습이 안정적 (선택사항)
    sampler = GroupBatchSampler(dataset.group_ids, batch_size=BATCH_SIZE)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0) # Windows에선 workers=0 권장

    # 2. Model
    model = SnapshotRiskModelFT(
        num_numeric=len(NUMERIC_COLS),
        cat_cardinalities=dataset.cat_cardinalities,
        d_token=32, n_heads=4, n_layers=3
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # 3. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            numeric = batch["numeric"].to(DEVICE)
            categorical = batch["categorical"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(numeric, categorical)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

    # 4. 저장
    # 모델 가중치 저장
    torch.save(model.state_dict(), "teacher_oulad.pt")
    
    # [중요] Student가 쓸 수 있도록 카테고리 인코더 저장
    with open("cat_encoders.pkl", "wb") as f:
        pickle.dump(dataset.cat_encoders, f)
        
    print("Teacher Model & Encoders Saved!")

if __name__ == "__main__":
    train_teacher()