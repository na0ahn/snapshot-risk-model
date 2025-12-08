# models/snapshot_risk_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------
# 1. MLP Student Model
# ------------------------------------------------------
class MLPStudentWithTeacher(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, teacher_d_token=32, hidden_dim=64, dropout=0.3):
        super().__init__()
        
        # Teacher와 동일한 임베딩 구조 (Warm Start 호환)
        self.num_embedding = nn.Linear(1, teacher_d_token)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, teacher_d_token) for card in cat_cardinalities
        ])
        
        # 입력 차원: (수치형 + 범주형 개수) * 임베딩 차원
        self.input_dim = (num_numeric + len(cat_cardinalities)) * teacher_d_token
        
        # Simple MLP Backbone
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, numeric, categorical):
        # Embeddings
        x_list = []
        for i in range(numeric.shape[1]):
            x_list.append(self.num_embedding(numeric[:, i].unsqueeze(1)))
        for i, layer in enumerate(self.cat_embeddings):
            x_list.append(layer(categorical[:, i]))
            
        # Flatten [Batch, Features, Emb] -> [Batch, Features * Emb]
        x = torch.stack(x_list, dim=1).view(numeric.size(0), -1)
        
        logits = self.mlp(x)
        return logits, torch.sigmoid(logits)


# ------------------------------------------------------
# 2. Lightweight Efficient Model
# ------------------------------------------------------
class EfficientStudentModel(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, teacher_d_token=32, student_d_token=8, dropout=0.1):
        super().__init__()
        self.num_embedding = nn.Linear(1, teacher_d_token)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, teacher_d_token) for card in cat_cardinalities
        ])
        
        # Dimension Reduction
        self.projector = nn.Linear(teacher_d_token, student_d_token)
        self.dropout = nn.Dropout(dropout)
        
        self.head = nn.Linear(student_d_token, 1)
        self.norm = nn.LayerNorm(student_d_token)

    def forward(self, numeric, categorical):
        x_list = []
        for i in range(numeric.shape[1]):
            x_list.append(self.num_embedding(numeric[:, i].unsqueeze(1)))
        for i, layer in enumerate(self.cat_embeddings):
            x_list.append(layer(categorical[:, i]))
            
        x = torch.stack(x_list, dim=1) # [B, Seq, 32]
        
        # Compress
        x = self.projector(x) # [B, Seq, 8]
        x = self.dropout(x)

        # Identity Attention (Param-Free)
        attn_scores = torch.matmul(x, x.transpose(-2, -1)) * (x.size(-1)**-0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        x_attn = torch.matmul(attn_probs, x)
        
        x = self.norm(x + x_attn) # Residual
        x = x.mean(dim=1)         # Pooling
        
        logits = self.head(x)
        return logits, torch.sigmoid(logits)


# ------------------------------------------------------
# 3. Standard Transformer Student
# ------------------------------------------------------
class SnapshotRiskModelFT(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, d_token=32, n_heads=4, n_layers=2, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.num_embedding = nn.Linear(1, d_token)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, d_token) for card in cat_cardinalities
        ])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_token, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, numeric, categorical):
        x_list = []
        for i in range(numeric.shape[1]):
            x_list.append(self.num_embedding(numeric[:, i].unsqueeze(1)))
        for i, layer in enumerate(self.cat_embeddings):
            x_list.append(layer(categorical[:, i]))
            
        x = torch.stack(x_list, dim=1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        
        logits = self.head(x)
        return logits, torch.sigmoid(logits)
