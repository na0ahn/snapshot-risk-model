# configs/best_params.py

# 1. MLP Student
# Teacher 지식을 증류받은 단순 MLP 구조. 28명 데이터에서 가장 성능이 좋음.
MLP_BEST_CONFIG = {
    "model_type": "mlp",
    "hidden_dim": 64,       # 실험으로 찾은 최적 Hidden Size
    "dropout": 0.3,         # 과적합 방지용
    "lr": 0.0009,           # 학습률
    "teacher_d_token": 32,  # Teacher 임베딩 차원 (고정)
    "focal_alpha": 0.82,    # 불균형 데이터 가중치
    "focal_gamma": 1.16,    # Focal Loss 강도
    "epochs": 50            # 충분한 학습
}

# 2. Lightweight Student (경량화 모델)
# Identity Attention을 사용한 초경량 모델.
LIGHTWEIGHT_BEST_CONFIG = {
    "model_type": "efficient",
    "d_token": 8,           # 32 -> 8 압축
    "n_layers": 1,
    "dropout": 0.1,
    "lr": 0.0005,
    "teacher_d_token": 32,
    "focal_alpha": 0.85,
    "focal_gamma": 1.5,
    "epochs": 50
}

# 3. Teacher-Like Student (기본)
# Teacher와 유사한 Transformer 구조 (비교군).
TRANSFORMER_BEST_CONFIG = {
    "model_type": "transformer",
    "d_token": 32,
    "n_layers": 2,
    "dropout": 0.1,
    "lr": 0.00035,
    "focal_alpha": 0.82,
    "focal_gamma": 1.15,
    "epochs": 30
}

# 기본 사용 설정
DEFAULT_CONFIG = MLP_BEST_CONFIG
