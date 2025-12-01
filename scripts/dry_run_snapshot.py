# scripts/dry_run_snapshot.py
import torch
from torch.utils.data import DataLoader

from datasets.snapshot_dataset import SnapshotDataset
from datasets.samplers import GroupBatchSampler
from models.snapshot_risk_model import SnapshotRiskModelFT
from losses.snapshot_losses import monotone_and_smooth_loss
from configs.feature_config import NUMERIC_COLS, CAT_COLS, LABEL_COL, GROUP_COLS

def dry_run(csv_path: str, cat_encoders=None):
    dataset = SnapshotDataset(
        csv_path=csv_path,
        numeric_cols=NUMERIC_COLS,
        cat_cols=CAT_COLS,
        label_col=LABEL_COL,
        group_cols=GROUP_COLS,
        teacher_prob_col=None,
        fit_encoders=(cat_encoders is None),
        cat_encoders=cat_encoders,
    )

    sampler = GroupBatchSampler(dataset.group_ids, batch_size=64)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    batch = next(iter(loader))
    numeric = batch["numeric"]
    categorical = batch["categorical"]
    labels = batch["label"]
    group_ids = batch["group_id"]
    ttd = batch["time_to_deadline"]

    print("numeric shape:", numeric.shape)
    print("categorical shape:", categorical.shape)

    model = SnapshotRiskModelFT(
        num_numeric=len(NUMERIC_COLS),
        cat_cardinalities=[int(dataset.df[c].nunique()) for c in CAT_COLS],
    )

    logits, probs = model(numeric, categorical)
    print("logits shape:", logits.shape)
    print("probs min/max:", probs.min().item(), probs.max().item())

    # 간단한 BCE + monotone loss 한 번 계산
    bce = torch.nn.BCEWithLogitsLoss()(logits, labels)
    reg = monotone_and_smooth_loss(probs, group_ids, ttd, lambda_mono=0.1, lambda_smooth=0.1)
    print("BCE loss:", bce.item())
    print("Reg loss:", reg.item())


if __name__ == "__main__":
    # 1) Teacher dry-run (OULAD, 인코더 자체 생성)
    print("=== OULAD dry-run ===")
    dry_run("data/oulad/processed/oulad_snapshots.csv")

    # 2) Student dry-run (SKKU, Teacher 인코더 재사용)
    try:
        import pickle
        with open("cat_encoders.pkl", "rb") as f:
            encs = pickle.load(f)
        print("\n=== SKKU dry-run (with teacher encoders) ===")
        dry_run("data/skku/processed/skku_snapshots.csv", cat_encoders=encs)
    except FileNotFoundError:
        print("\n[주의] cat_encoders.pkl이 없습니다. Teacher를 먼저 학습시키세요.")
