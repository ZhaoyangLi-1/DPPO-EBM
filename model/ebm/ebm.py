import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make timm import optional
try:
    import timm
    from timm.layers import DropPath
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Image processing features will be disabled.")

# ------------------------------------------------------------------
# 0.  Sinusoidal Positional Encoding
# ------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x:(B,) or (T,)
        device = x.device
        half   = self.dim // 2
        freq   = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / (half - 1)
        )                                                    # (half,)
        emb = x.unsqueeze(-1) * freq                         # (B,half)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)      # (B,dim)
        if self.dim % 2 == 1:                                # odd dim pad
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb                                           # (B,dim)

# ------------------------------------------------------------------
# 1.  Attention Pool over multi-view features
# ------------------------------------------------------------------
class AttentionPool(nn.Module):
    def __init__(self, hid: int):
        super().__init__()
        self.q    = nn.Linear(hid, hid)
        self.k    = nn.Linear(hid, hid)
        self.v    = nn.Linear(hid, hid)
        self.norm = nn.LayerNorm(hid)
        self.proj = nn.Linear(hid, hid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x:(B,V,H)
        q = self.q(x[:, :1])                                 # (B,1,H)
        k = self.k(x)                                        # (B,V,H)
        v = self.v(x)                                        # (B,V,H)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(x.size(-1))  # (B,1,V)
        w    = attn.softmax(dim=-1)                          # (B,1,V)
        pooled = (w @ v).squeeze(1)                          # (B,H)
        return self.norm(self.proj(pooled))                  # (B,H)

# ------------------------------------------------------------------
# 2.  ConvNeXt backbone -> view embedding (only if timm is available)
# ------------------------------------------------------------------
class ConvNeXtMultiView(nn.Module):
    def __init__(self, num_views: int, hid: int = 512):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for ConvNeXtMultiView. Please install timm or set num_views to None.")
        
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=True, features_only=True
        )
        c_out = self.backbone.feature_info[-1]["num_chs"]
        self.proj = nn.Linear(c_out, hid, bias=False)
        self.ln   = nn.LayerNorm(hid)
        self.pool = AttentionPool(hid)
        for p in self.backbone.parameters():        # freeze
            p.requires_grad = False
        self.num_views = num_views
        self.hid       = hid

    def forward(self, views: torch.Tensor) -> torch.Tensor:  # (B,V,C,H,W)
        B, V, C, H, W = views.shape
        x = views.view(B * V, C, H, W)
        feat = self.backbone(x)[-1].mean(dim=(-2, -1))       # (B*V,C_out)
        feat = self.proj(feat).view(B, V, -1)                # (B,V,H)
        return self.ln(self.pool(feat))                      # (B,H)

# ------------------------------------------------------------------
# 3.  Residual MLP block
# ------------------------------------------------------------------
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, expansion=4, drop=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.drop = nn.Dropout(drop)
        if TIMM_AVAILABLE:
            self.drop_path = DropPath(drop)
        else:
            self.drop_path = nn.Dropout(drop)

    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = self.drop_path(self.drop(x))
        return residual + x

# ------------------------------------------------------------------
# 4.  State / Action encoders
# ------------------------------------------------------------------
class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, embed_dim: int, depth: int = 3, p_drop: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, embed_dim)
        self.blocks = nn.Sequential(*[
            ResidualMLPBlock(embed_dim, expansion=4, drop=p_drop)
            for _ in range(depth)
        ])

    def forward(self, s: torch.Tensor) -> torch.Tensor:  # (B, D_s)
        x = self.input_proj(s)
        return self.blocks(x)  # (B, E)

class MultiKernelDilatedConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch, out_ch, 1, padding=0, dilation=1),
            nn.Conv1d(in_ch, out_ch, 3, padding=1, dilation=1),
            nn.Conv1d(in_ch, out_ch, 3, padding=2, dilation=2),
            nn.Conv1d(in_ch, out_ch, 3, padding=4, dilation=4),
        ])
        self.ln = nn.LayerNorm(out_ch * len(self.convs))
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):  # x: (B, C_in, T)
        feats = [F.gelu(conv(x)) for conv in self.convs]     # each: (B, out_ch, T)
        x = torch.cat(feats, dim=1)                          # (B, out_ch*4, T) = (B, 2E, T) 若 out_ch=E//2
        x = x.transpose(1, 2)                                # (B, T, 2E)
        return self.ln(self.drop(x))                         # (B, T, 2E)

class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int, p_drop: float = 0.1, **_):
        super().__init__()
        self.input_proj = nn.Linear(action_dim, embed_dim)
        self.multi_conv = MultiKernelDilatedConv1D(embed_dim, embed_dim // 2, p_drop)
        self.proj_out = nn.Linear(embed_dim * 3, embed_dim)

    def forward(self, acts: torch.Tensor) -> torch.Tensor:   # acts: (B, T, C_a)
        assert acts.dim() == 3, f"expects (B,T,Ca), got {acts.shape}"
        x0 = self.input_proj(acts)                           # (B, T, E)
        x1 = self.multi_conv(x0.transpose(1, 2))            # (B, T, 2E)
        out = torch.cat([x0, x1], dim=-1)                   # (B, T, 3E)
        return self.proj_out(out)                           # (B, T, E)

# ------------------------------------------------------------------
# 5.  Main Energy-Based Model (no t_idx)
# ------------------------------------------------------------------
class EBM(nn.Module):
    """Transformer-based Action-State-Image energy model (no t_idx)."""
    def __init__(self, cfg):
        super().__init__()
        # ---------- cfg fields ----------
        E              = cfg.ebm.embed_dim
        self.embed_dim = E
        self.state_dim = cfg.ebm.state_dim
        self.action_dim= cfg.ebm.action_dim
        self.nhead     = cfg.ebm.nhead
        self.depth     = cfg.ebm.depth
        self.dropout   = cfg.ebm.dropout
        self.use_cls   = cfg.ebm.use_cls_token
        self.num_views = cfg.ebm.num_views

        # ---------- sub-modules ----------
        self.state_proj  = StateEncoder(self.state_dim,  E, p_drop=0.1)
        self.action_proj = ActionEncoder(self.action_dim, E, p_drop=0.1)

        # Only create view encoder if num_views is specified and timm is available
        if self.num_views and TIMM_AVAILABLE:
            self.view_encoder = ConvNeXtMultiView(self.num_views, hid=E)
        else:
            self.view_encoder = None

        # positional / denoise timestep embeddings
        self.k_embed   = SinusoidalPosEmb(E)     # 去噪时间线
        self.temp_pos  = SinusoidalPosEmb(E)     # 片段内相对位置 0..T-1

        if self.use_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, E))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=E, nhead=self.nhead, dim_feedforward=4*E,
            dropout=self.dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.depth)

        self.energy_head = nn.Sequential(
            nn.LayerNorm(E),
            nn.Linear(E, 1)
        )

    # ------------------------------------------------------------------
    # forward (no t_idx)
    # ------------------------------------------------------------------
    def forward(
        self,
        k_idx: torch.Tensor,            # (B,)  去噪时间线索引
        views:  torch.Tensor,           # (B,V,C,H,W) or None
        poses:  torch.Tensor,           # (B,D_s)
        actions:torch.Tensor,           # (B,T,D_a)
    ):
        """
        Returns
        -------
        energy : (B,)             negative score
        token  : (B,E)            pooled representation (before head)
        """
        B, T, _ = actions.size()
        device  = actions.device

        # (1)  encode action sequence + temporal position (片段内位置)
        x = self.action_proj(actions)                       # (B,T,E)
        t_range = torch.arange(T, device=device, dtype=torch.float32)
        pos_emb = self.temp_pos(t_range).unsqueeze(0)       # (1,T,E)
        x = x + pos_emb                                     # (B,T,E)

        # (2) add denoising-time embeddings (仅 k_idx)
        x = x + self.k_embed(k_idx).unsqueeze(1)            # (B,T,E)

        # (3) encode state (pose) + image token
        pose_tok = self.state_proj(poses).unsqueeze(1)      # (B,1,E)
        if self.view_encoder and views is not None:
            img_tok = self.view_encoder(views).unsqueeze(1) # (B,1,E)
        else:
            img_tok = torch.zeros_like(pose_tok)
        s_token = pose_tok + img_tok                        # (B,1,E)

        # (4) concat sequence
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)          # (B,1,E)
            seq = torch.cat([cls, s_token, x], dim=1)       # (B,2+T,E)
            out_idx = 0
        else:
            seq = torch.cat([s_token, x], dim=1)            # (B,1+T,E)
            out_idx = 0

        # (5) transformer + head
        y   = self.transformer(seq)                         # (B,L,E)
        y0  = y[:, out_idx, :]                              # (B,E)
        en  = self.energy_head(y0).view(B)                  # (B,)
        return en, y0

