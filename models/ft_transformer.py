# models/ft_transformer.py
import torch
import torch.nn as nn

class TabFeatureTokenizerFT(nn.Module):
    """
    Numeric -> Linear Embedding -> Token
    Categorical -> Table Embedding -> Token
    [CLS] Token 추가
    """
    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: list,
        d_token: int = 32,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.num_numeric = num_numeric
        self.d_token = d_token
        self.use_cls_token = use_cls_token

        # 1. Numeric Feature Embedding
        # 각 피처마다 고유한 weight(w)와 bias(b)를 가짐: token = w * x + b
        self.num_weight = nn.Parameter(torch.randn(num_numeric, d_token))
        self.num_bias = nn.Parameter(torch.zeros(num_numeric, d_token))

        # 2. Categorical Feature Embedding
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, d_token) for card in cat_cardinalities
        ])

        # 3. CLS Token (전체 정보를 요약할 토큰)
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        """
        Args:
            numeric: (Batch, num_numeric)
            categorical: (Batch, num_cats)
        Returns:
            tokens: (Batch, Seq_Len, d_token)
        """
        B = numeric.size(0)

        # (1) Numeric -> Token
        # numeric.unsqueeze(-1): (B, N_num, 1)
        # weight: (N_num, d)
        # broadcasting으로 (B, N_num, d) 생성
        num_tokens = numeric.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)

        # (2) Categorical -> Token
        cat_tokens_list = []
        for i, emb in enumerate(self.cat_embeddings):
            cat_tokens_list.append(emb(categorical[:, i])) # (B, d)
        
        if cat_tokens_list:
            cat_tokens = torch.stack(cat_tokens_list, dim=1) # (B, N_cat, d)
            x = torch.cat([num_tokens, cat_tokens], dim=1)
        else:
            x = num_tokens

        # (3) Add CLS Token
        if self.use_cls_token:
            cls_token = self.cls_token.expand(B, -1, -1) # (B, 1, d)
            x = torch.cat([cls_token, x], dim=1)

        return x


class FTTransformerEncoder(nn.Module):
    """
    Standard Transformer Encoder Stack
    """
    def __init__(
        self,
        d_token: int = 32,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True, # (Batch, Seq, Feature)
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        return self.encoder(x)