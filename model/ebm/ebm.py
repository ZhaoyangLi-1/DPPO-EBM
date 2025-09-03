import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def _rope_build_inv_freq(dim: int, base: float = 10000.0, device=None):
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

def _rope_angles_from_pos(pos: torch.Tensor, dim: int, base: float = 10000.0):
    pos = pos.to(torch.float32).view(-1, 1)
    inv_freq = _rope_build_inv_freq(dim, base=base, device=pos.device)
    theta = torch.matmul(pos, inv_freq.view(1, -1))
    cos = torch.cos(theta).unsqueeze(1)
    sin = torch.sin(theta).unsqueeze(1)
    return cos, sin

def _rope_rotate_vec(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    B, D = x.shape
    x1 = x.view(B, D // 2, 2)[..., 0]
    x2 = x.view(B, D // 2, 2)[..., 1]
    xr0 = x1 * cos.squeeze(1) - x2 * sin.squeeze(1)
    xr1 = x1 * sin.squeeze(1) + x2 * cos.squeeze(1)
    return torch.stack([xr0, xr1], dim=-1).reshape(B, D)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half = self.dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / (half - 1))
        emb = x.unsqueeze(-1) * freq
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class AttentionPool(nn.Module):
    def __init__(self, hid: int):
        super().__init__()
        self.q = nn.Linear(hid, hid)
        self.k = nn.Linear(hid, hid)
        self.v = nn.Linear(hid, hid)
        self.proj = nn.Linear(hid, hid)
        self.norm = nn.LayerNorm(hid)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x[:, :1])
        k = self.k(x)
        v = self.v(x)
        w = (q @ k.transpose(-2, -1)) / math.sqrt(x.size(-1))
        pooled = (w.softmax(-1) @ v).squeeze(1)
        return self.norm(self.proj(pooled))

class ConvNeXtMultiView(nn.Module):
    def __init__(self, num_views: int, hid: int = 512):
        super().__init__()
        self.backbone = timm.create_model("convnext_tiny", pretrained=True, features_only=True)
        c_out = self.backbone.feature_info[-1]["num_chs"]
        self.proj = nn.Linear(c_out, hid, bias=False)
        self.ln = nn.LayerNorm(hid)
        self.pool = AttentionPool(hid)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.num_views = num_views
        self.hid = hid
    def forward(self, views: torch.Tensor) -> torch.Tensor:
        B, V, C, H, W = views.shape
        x = views.view(B * V, C, H, W)
        feat = self.backbone(x)[-1].mean(dim=(-2, -1))
        feat = self.proj(feat).view(B, V, -1)
        return self.ln(self.pool(feat))

class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, embed_dim: int, depth: int = 4, p_drop: float = 0.1, hidden: int = None):
        super().__init__()
        hidden = hidden or (4 * embed_dim)
        layers = [nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden), nn.GELU(), nn.Dropout(p_drop)]
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p_drop)]
        layers += [nn.Linear(hidden, embed_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)

class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int, depth: int = 4, p_drop: float = 0.1, hidden: int = None):
        super().__init__()
        hidden = hidden or (4 * embed_dim)
        layers = [nn.LayerNorm(action_dim), nn.Linear(action_dim, hidden), nn.GELU(), nn.Dropout(p_drop)]
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p_drop)]
        layers += [nn.Linear(hidden, embed_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, acts: torch.Tensor) -> torch.Tensor:
        assert acts.dim() == 3
        return self.net(acts)

class EBM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        E = cfg.ebm.embed_dim
        self.E = E
        self.state_dim = cfg.ebm.state_dim
        self.action_dim = cfg.ebm.action_dim
        self.nhead = cfg.ebm.nhead
        self.depth = cfg.ebm.depth
        self.dropout = cfg.ebm.dropout
        self.use_cls = cfg.ebm.use_cls_token
        self.num_views = cfg.ebm.num_views
        self.state_proj = StateEncoder(self.state_dim, E, p_drop=self.dropout, depth=self.depth)
        self.action_proj = ActionEncoder(self.action_dim, E, p_drop=self.dropout, depth=self.depth)
        self.view_encoder = (ConvNeXtMultiView(self.num_views, hid=E) if self.num_views else None)
        self.temp_pos = SinusoidalPosEmb(E)
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, E))
        enc_layer = nn.TransformerEncoderLayer(d_model=E, nhead=self.nhead, dim_feedforward=4 * E, dropout=self.dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.depth)
        self.pool = AttentionPool(E)
        self.energy_head = nn.Sequential(
            nn.LayerNorm(E),
            nn.Linear(E, E),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(E, 1),
        )
        self.log_temp = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def encode_tokens(self, k_idx: torch.Tensor, t_idx: torch.Tensor, views: torch.Tensor, poses: torch.Tensor, actions: torch.Tensor):
        # Accept actions as [B, H, A] or [B, T, H, A]; if 4D, collapse T by mean
        if actions.dim() == 4:
            actions = actions.mean(dim=1)
        B, T, _ = actions.size()
        device = actions.device
        x = self.action_proj(actions)
        t_range = torch.arange(T, device=device, dtype=torch.float32)
        pos_emb = self.temp_pos(t_range).unsqueeze(0)
        x = x + pos_emb

        pose_tok = self.state_proj(poses).unsqueeze(1)
        if self.view_encoder and views is not None:
            img_tok = self.view_encoder(views).unsqueeze(1)
        else:
            img_tok = torch.zeros_like(pose_tok)
        s_token = pose_tok + img_tok

        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            seq = torch.cat([cls, s_token, x], dim=1)
        else:
            seq = torch.cat([s_token, x], dim=1)

        y = self.transformer(seq)
        if self.use_cls:
            y0 = y[:, 0, :]
        else:
            y0 = y[:, 0, :]

        cos_t, sin_t = _rope_angles_from_pos(t_idx.to(y0.device), dim=self.E, base=10000.0)
        y0 = _rope_rotate_vec(y0, cos_t, sin_t)

        cos_k, sin_k = _rope_angles_from_pos(k_idx.to(y0.device), dim=self.E, base=10000.0)
        y0 = _rope_rotate_vec(y0, cos_k, sin_k)

        pooled = self.pool(y)

        z = 0.5 * (y0 + pooled)
        return z

    def forward(self, k_idx: torch.Tensor, t_idx: torch.Tensor, views: torch.Tensor, poses: torch.Tensor, actions: torch.Tensor):
        z = self.encode_tokens(k_idx, t_idx, views, poses, actions)
        en = self.energy_head(z).view(-1)
        return en, z