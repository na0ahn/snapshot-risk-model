# inference_demo.py

import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# 프로젝트 모듈 임포트
from datasets.snapshot_dataset import SnapshotDataset
from configs.feature_config import NUMERIC_COLS, CAT_COLS, LABEL_COL, GROUP_COLS
from configs.best_params import DEFAULT_CONFIG as CFG  # Best Params
from models.snapshot_risk_model import MLPStudentWithTeacher, EfficientStudentModel, SnapshotRiskModelFT

def main():
    print("="*60)
    print("   AI Risk Prediction Demo (Team Share Version)")
    print("="*60)

    # 1. 설정
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 경로 확인 필요
    DATA_PATH = "data/skku/processed/skku_snapshots_with_teacher.csv"
    ENCODER_PATH = "checkpoints/cat_encoders.pkl"
    # 학습된 가중치 파일 (없으면 랜덤 예측됨)
    MODEL_WEIGHT_PATH = "checkpoints/best_mlp_student.pt" 

    print(f"[Info] Device: {DEVICE}")
    print(f"[Info] Model Type: {CFG['model_type']}")

    # 2. 데이터셋 준비
    if not os.path.exists(DATA_PATH) or not os.path.exists(ENCODER_PATH):
        print(f"[Error] 데이터 파일이나 인코더가 없습니다.\n -> {DATA_PATH}\n -> {ENCODER_PATH}")
        return

    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)

    dataset = SnapshotDataset(
        csv_path=DATA_PATH,
        numeric_cols=NUMERIC_COLS,
        cat_cols=CAT_COLS,
        label_col=LABEL_COL,
        group_cols=GROUP_COLS,
        teacher_prob_col="teacher_prob",
        fit_encoders=False,
        cat_encoders=encoders
    )
    
    # 데모용으로 10개 샘플만 로드
    loader = DataLoader(dataset, batch_size=10, shuffle=False)
    batch = next(iter(loader))

    # 3. 모델 초기화
    cat_cardinalities = [len(encoders[c]) for c in CAT_COLS]
    
    if CFG['model_type'] == 'mlp':
        model = MLPStudentWithTeacher(
            len(NUMERIC_COLS), cat_cardinalities,
            teacher_d_token=CFG['teacher_d_token'],
            hidden_dim=CFG['hidden_dim'],
            dropout=CFG['dropout']
        )
    elif CFG['model_type'] == 'efficient':
        model = EfficientStudentModel(
            len(NUMERIC_COLS), cat_cardinalities,
            teacher_d_token=CFG['teacher_d_token'],
            student_d_token=CFG['d_token'],
            dropout=CFG['dropout']
        )
    else:
        model = SnapshotRiskModelFT(
            len(NUMERIC_COLS), cat_cardinalities,
            d_token=CFG['d_token'], n_layers=CFG['n_layers'],
            dropout=CFG['dropout']
        )

    model.to(DEVICE)

    # 4. 가중치 로드
    if os.path.exists(MODEL_WEIGHT_PATH):
        try:
            state = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE)
            model.load_state_dict(state, strict=False)
            print("[Success] 학습된 가중치를 로드했습니다.")
        except Exception as e:
            print(f"[Warning] 가중치 로드 실패: {e}")
            print(" -> 랜덤 초기화 상태로 예측합니다.")
    else:
        print("[Warning] 가중치 파일(.pt)이 없습니다. 랜덤 초기화 상태로 예측합니다.")

    # 5. 추론 실행
    model.eval()
    with torch.no_grad():
        num = batch['numeric'].to(DEVICE)
        cat = batch['categorical'].to(DEVICE)
        
        # Forward
        logits, probs = model(num, cat)
        probs = probs.cpu().numpy().flatten()
        
        # 실제 정답 (비교용)
        labels = batch['label'].numpy().flatten()

    # 6. 결과 리포트 출력
    print("\n[Prediction Report (Top 10 Samples)]")
    print(f"{'Idx':<5} | {'Risk Prob':<12} | {'Status':<10} | {'True Label'}")
    print("-" * 50)
    
    for i in range(len(probs)):
        p = probs[i]
        true_l = labels[i]
        
        # Threshold 0.55 기준
        status = "Danger" if p > 0.55 else ("Warning" if p > 0.3 else "Safe")
        
        print(f"{i:<5} | {p:.4f}       | {status:<10} | {int(true_l)}")

    print("-" * 50)
    print("Demo Finished.")

if __name__ == "__main__":
    main()
