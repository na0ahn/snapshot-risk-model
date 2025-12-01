import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class SnapshotDataset(Dataset):
    """
    snapshot CSV를 읽어서 모델 입력 텐서로 변환.
    OULAD(fit_encoders=True)와 SKKU(fit_encoders=False) 모두 지원.
    """

    def __init__(
        self,
        csv_path: str,
        numeric_cols: list[str],
        cat_cols: list[str],
        label_col: str,
        group_cols: list[str],
        teacher_prob_col: str | None = None,
        fit_encoders: bool = True,
        cat_encoders: dict[str, dict] | None = None,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)

        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols
        self.label_col = label_col
        self.group_cols = group_cols
        self.teacher_prob_col = teacher_prob_col

        # 1. 결측치 및 타입 처리 (안전장치)
        # Numeric: NaN -> 0.0
        for c in self.numeric_cols:
            if c not in self.df.columns:
                self.df[c] = 0.0
            self.df[c] = pd.to_numeric(self.df[c], errors='coerce').fillna(0.0)

        # Categorical: NaN -> "UNK"
        for c in self.cat_cols:
            if c not in self.df.columns:
                self.df[c] = "UNK"
            self.df[c] = self.df[c].astype(str).replace('nan', 'UNK').fillna("UNK")

        # Label
        if self.label_col not in self.df.columns:
            self.df[self.label_col] = 0.0
        self.df[self.label_col] = self.df[self.label_col].fillna(0.0)

        # Teacher Prob (성대 Student 학습용)
        if self.teacher_prob_col:
            if self.teacher_prob_col not in self.df.columns:
                self.df[self.teacher_prob_col] = np.nan
            # NaN은 그대로 둠 (Loss 계산 시 마스킹 처리)

        # 2. Group ID 생성 (Monotone Loss용 식별자)
        # group_cols가 없으면 더미 생성
        missing_groups = [c for c in self.group_cols if c not in self.df.columns]
        if missing_groups:
            for c in missing_groups: self.df[c] = 0
            
        # (student_id, assignment_id) 조합을 고유 정수로 매핑
        self.df["group_id_str"] = self.df[self.group_cols].astype(str).agg("_".join, axis=1)
        self.df["group_id"], uniques = pd.factorize(self.df["group_id_str"])
        # Sampler에서 사용하기 위해 배열로 저장
        self.group_ids = self.df["group_id"].values 

        # 3. 범주형 인코딩 (Label Encoding)
        self.fit_encoders = fit_encoders
        if fit_encoders:
            # Teacher 학습 시: 인코더 생성
            self.cat_encoders = {}
            for c in self.cat_cols:
                self.df[c] = self.df[c].astype("category")
                self.cat_encoders[c] = {v: i for i, v in enumerate(self.df[c].cat.categories)}
                self.df[c] = self.df[c].cat.codes.astype("int64")
        else:
            # Student 학습 시: Teacher의 인코더 재사용
            if cat_encoders is None:
                raise ValueError("fit_encoders=False일 때는 cat_encoders를 반드시 주어야 합니다.")
            self.cat_encoders = cat_encoders
            for c in self.cat_cols:
                mapping = self.cat_encoders.get(c, {})
                unk_idx = len(mapping) # 모르는 범주는 마지막 인덱스(UNK)로
                # map 후 NaN(UNK) 처리
                self.df[c] = self.df[c].map(mapping).fillna(unk_idx).astype("int64")

        # 4. Cardinality 계산 (모델 Embedding Layer 크기 결정)
        self.cat_cardinalities = []
        for c in self.cat_cols:
            # Student의 경우 UNK가 추가될 수 있으므로 인코더 크기 + 1(UNK) 고려
            # 하지만 간단히 max code + 1로 처리
            max_code = self.df[c].max()
            self.cat_cardinalities.append(int(max_code) + 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        item = {
            "numeric": torch.tensor([row[c] for c in self.numeric_cols], dtype=torch.float32),
            "categorical": torch.tensor([row[c] for c in self.cat_cols], dtype=torch.long),
            "label": torch.tensor(float(row[self.label_col]), dtype=torch.float32),
            "group_id": torch.tensor(int(row["group_id"]), dtype=torch.long),
        }

        # 선택적 컬럼 처리
        if "time_to_deadline_days" in row.index:
            item["time_to_deadline"] = torch.tensor(float(row["time_to_deadline_days"]), dtype=torch.float32)
        else:
            item["time_to_deadline"] = torch.tensor(0.0, dtype=torch.float32)

        if self.teacher_prob_col and self.teacher_prob_col in row.index:
            item["teacher_prob"] = torch.tensor(float(row[self.teacher_prob_col]), dtype=torch.float32)
        else:
            item["teacher_prob"] = torch.tensor(float('nan'), dtype=torch.float32)
            
        return item