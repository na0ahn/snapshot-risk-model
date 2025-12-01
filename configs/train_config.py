# configs/train_config.py
from dataclasses import dataclass

@dataclass
class TrainConfig:
    # === Paths ===
    OULAD_CSV: str = "data/oulad/processed/oulad_snapshots.csv"
    SKKU_CSV: str = "data/skku/processed/skku_snapshots.csv"
    SKKU_CSV_WITH_TEACHER: str = "data/skku/processed/skku_snapshots_with_teacher.csv"

    TEACHER_MODEL_PATH: str = "checkpoints/teacher_oulad.pt"
    ENCODER_PATH: str = "checkpoints/cat_encoders.pkl"
    STUDENT_MODEL_PATH: str = "checkpoints/student_skku.pt"

    # === Model Hyperparameters ===
    D_TOKEN: int = 32
    N_HEADS: int = 4
    N_LAYERS: int = 3
    DIM_FEEDFORWARD: int = 128
    DROPOUT: float = 0.1

    # === Training Hyperparameters ===
    BATCH_SIZE: int = 256
    LR_TEACHER: float = 1e-3
    LR_STUDENT: float = 5e-4
    EPOCHS: int = 20
    SEED: int = 42

    # === Loss Weights (Student) ===
    LAMBDA_KL: float = 0.5
    LAMBDA_MONO: float = 0.1
    LAMBDA_SMOOTH: float = 0.1

default_config = TrainConfig()
