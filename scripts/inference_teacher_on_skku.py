# scripts/inference_teacher_on_skku.py

import os
import sys
import pickle
from typing import List

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# 프로젝트 루트 경로를 sys.path에 추가 (직접 실행해도 import 되도록)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from datasets.snapshot_dataset import SnapshotDataset
from models.snapshot_risk_model import SnapshotRiskModelFT
from configs.feature_config import NUMERIC_COLS, CAT_COLS, LABEL_COL, GROUP_COLS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 경로 상수
SKKU_SNAPSHOT_PATH = os.path.join("data", "skku", "processed", "skku_snapshots.csv")
SKKU_SNAPSHOT_WITH_TEACHER_PATH = os.path.join(
    "data", "skku", "processed", "skku_snapshots_with_teacher.csv"
)
TEACHER_CHECKPOINT_PATH = os.path.join("checkpoints", "teacher_oulad.pt")
ENCODER_PATH = os.path.join("checkpoints", "cat_encoders.pkl")


def run_inference_teacher_on_skku():
    print(f"[Device] {DEVICE}")

    # ---------------------------------------------------
    # 1. Teacher 쪽에서 학습해 둔 카테고리 인코더 로드
    # ---------------------------------------------------
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"[ERR] Encoder 파일이 없습니다: {ENCODER_PATH}")

    with open(ENCODER_PATH, "rb") as f:
        cat_encoders = pickle.load(f)
    print(f"[OK] Encoders 로드 완료. Keys: {list(cat_encoders.keys())}")

    # Teacher 기준 카디널리티(embedding 크기)를 고정
    #  -> Teacher가 학습한 카테고리 공간을 그대로 사용
    teacher_cardinalities: List[int] = [len(cat_encoders[c]) for c in CAT_COLS]
    print(f"[Info] Teacher cardinalities: {dict(zip(CAT_COLS, teacher_cardinalities))}")

    # ---------------------------------------------------
    # 2. SKKU snapshot CSV 로드 → Dataset 구성
    # ---------------------------------------------------
    if not os.path.exists(SKKU_SNAPSHOT_PATH):
        raise FileNotFoundError(f"[ERR] SKKU snapshot CSV가 없습니다: {SKKU_SNAPSHOT_PATH}")

    dataset = SnapshotDataset(
        csv_path=SKKU_SNAPSHOT_PATH,
        numeric_cols=NUMERIC_COLS,
        cat_cols=CAT_COLS,
        label_col=LABEL_COL,
        group_cols=GROUP_COLS,
        teacher_prob_col=None,         # 아직 없음
        fit_encoders=False,            # Teacher 인코더 재사용
        cat_encoders=cat_encoders,
    )

    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,   # 순서를 유지해야 teacher_prob를 그대로 붙일 수 있음
        num_workers=0,
    )

    print(f"[Dataset] SKKU snapshots loaded. n_samples = {len(dataset)}")

    # ---------------------------------------------------
    # 3. Teacher 모델 초기화 & 가중치 로드
    #    ⭐중요: cat_cardinalities는 'Teacher 기준'으로 세팅해야 함
    # ---------------------------------------------------
    if not os.path.exists(TEACHER_CHECKPOINT_PATH):
        raise FileNotFoundError(f"[ERR] Teacher 체크포인트가 없습니다: {TEACHER_CHECKPOINT_PATH}")

    model = SnapshotRiskModelFT(
        num_numeric=len(NUMERIC_COLS),
        cat_cardinalities=teacher_cardinalities,
        d_token=32,
        n_heads=4,
        n_layers=3,
        dim_feedforward=128,
        dropout=0.1,
    ).to(DEVICE)

    try:
        state = torch.load(TEACHER_CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state)
    except RuntimeError as e:
        print("[Fatal] Teacher state_dict 로드 실패 (Embedding 크기 불일치 가능성)")
        print(e)
        return

    model.eval()
    print("[OK] Teacher 모델 로드 완료")

    # ---------------------------------------------------
    # 4. Inference loop → teacher_prob 리스트 생성
    #    🔒 clamp: SKKU에서 새로 등장한 카테고리(UNK)를 범위 안으로 맞추기
    # ---------------------------------------------------
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Teacher inference on SKKU"):
            numeric = batch["numeric"].to(DEVICE)
            categorical = batch["categorical"].to(DEVICE)

            # 각 범주형 피처별로 index를 [0, card-1] 범위로 clamp
            #  - SnapshotDataset에서 UNK는 len(mapping)으로 할당되어 있음
            #  - Teacher는 len(mapping)까지만 embedding을 가지고 있으므로
            #    UNK를 마지막 유효 index(card-1)로 "붙여"준다.
            for i, card in enumerate(teacher_cardinalities):
                categorical[:, i] = categorical[:, i].clamp(max=card - 1)

            _, probs = model(numeric, categorical)  # (B,)
            all_probs.extend(probs.cpu().tolist())

    # 길이 체크
    if len(all_probs) != len(dataset):
        raise RuntimeError(
            f"[ERR] teacher_prob 길이 불일치: probs={len(all_probs)}, dataset={len(dataset)}"
        )

    # ---------------------------------------------------
    # 5. 원본 SKKU snapshot CSV에 teacher_prob 열 추가해서 저장
    # ---------------------------------------------------
    df = pd.read_csv(SKKU_SNAPSHOT_PATH)
    df["teacher_prob"] = all_probs

    os.makedirs(os.path.dirname(SKKU_SNAPSHOT_WITH_TEACHER_PATH), exist_ok=True)
    df.to_csv(SKKU_SNAPSHOT_WITH_TEACHER_PATH, index=False)
    print(f"[Done] Teacher inference 결과 저장 → {SKKU_SNAPSHOT_WITH_TEACHER_PATH}")


if __name__ == "__main__":
    run_inference_teacher_on_skku()
