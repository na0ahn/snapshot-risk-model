# models/snapshot_risk_model.py
import torch
import torch.nn as nn
from .ft_transformer import TabFeatureTokenizerFT, FTTransformerEncoder

class SnapshotRiskModelFT(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: list,
        d_token: int = 32,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()
        
        # 1. Tokenizer
        self.tokenizer = TabFeatureTokenizerFT(
            num_numeric=num_numeric,
            cat_cardinalities=cat_cardinalities,
            d_token=d_token,
            use_cls_token=use_cls_token
        )
        
        # 2. Backbone (Transformer)
        self.backbone = FTTransformerEncoder(
            d_token=d_token,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # 3. Risk Head (Classification)
        self.use_cls_token = use_cls_token
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, 1) # Logit 출력
        )

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor):
        # 1. Tokenize
        x = self.tokenizer(numeric, categorical) # (B, Seq, d)
        
        # 2. Contextualize (Self-Attention)
        x = self.backbone(x) # (B, Seq, d)
        
        # 3. Pool (Use CLS or Mean)
        if self.use_cls_token:
            x_pool = x[:, 0, :] # [CLS] token
        else:
            x_pool = x.mean(dim=1) # Mean pooling
            
        # 4. Predict
        logits = self.head(x_pool).squeeze(-1) # (B,)
        probs = torch.sigmoid(logits)          # (B,)
        
        return logits, probs