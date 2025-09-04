import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)

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

        enc_layer = nn.TransformerEncoderLayer(
            d_model=E, nhead=self.nhead, dim_feedforward=4 * E,
            dropout=self.dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.depth)

        self.pool = AttentionPool(E)
        self.pool_action = AttentionPool(E)
        
        # 禁用SDPA高效注意力，以支持梯度惩罚的二阶梯度计算（在所有模块创建后调用）
        self._disable_sdpa_for_gradient_penalty()       

        P = getattr(cfg.ebm, "proj_dim", E)
        self.proj_s = nn.Sequential(nn.LayerNorm(E), nn.Linear(E, P, bias=False))
        self.proj_a = nn.Sequential(nn.LayerNorm(E), nn.Linear(E, P, bias=False))

        self.energy_head = nn.Sequential(
            nn.LayerNorm(E),
            nn.Linear(E, E),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(E, 1),
        )
    
    def _disable_sdpa_for_gradient_penalty(self):
        def disable_sdpa_recursive(module):
            for _, child in module.named_children():
                if hasattr(child, '_use_sdpa'):
                    child._use_sdpa = False
                if hasattr(child, 'enable_nested_tensor'):
                    child.enable_nested_tensor = False
                disable_sdpa_recursive(child)
        
        disable_sdpa_recursive(self.transformer)
        # AttentionPool使用手动实现，不需要特别处理
        disable_sdpa_recursive(self.pool)
        disable_sdpa_recursive(self.pool_action)

    def encode_tokens(self, k_idx: torch.Tensor, t_idx: torch.Tensor,
                      views: torch.Tensor, poses: torch.Tensor, actions: torch.Tensor):
        B, T, _ = actions.size()
        device = actions.device

        a_tokens = self.action_proj(actions)
        t_range = torch.arange(T, device=device, dtype=torch.float32)
        pos_emb = self.temp_pos(t_range).unsqueeze(0)
        a_tokens = a_tokens + pos_emb                          # [B,T,E]

        s_token = self.state_proj(poses).unsqueeze(1)          # [B,1,E]
        if self.view_encoder and views is not None:
            img_tok = self.view_encoder(views).unsqueeze(1)    # [B,1,E]
        else:
            img_tok = torch.zeros_like(s_token)
        s_token = s_token + img_tok                            # [B,1,E]

        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            seq = torch.cat([cls, s_token, a_tokens], dim=1)   # [B,1+1+T,E]
        else:
            seq = torch.cat([s_token, a_tokens], dim=1)        # [B,1+T,E]

        y = self.transformer(seq)                              # [B,L,E]
        y0 = y[:, 0, :]
        cos_t, sin_t = _rope_angles_from_pos(t_idx.to(y0.device), dim=self.E, base=10000.0)
        y0 = _rope_rotate_vec(y0, cos_t, sin_t)
        cos_k, sin_k = _rope_angles_from_pos(k_idx.to(y0.device), dim=self.E, base=10000.0)
        y0 = _rope_rotate_vec(y0, cos_k, sin_k)

        pooled = self.pool(y)                                  # [B,E]
        z = 0.5 * (y0 + pooled)                                # fusion 表征

        # 取 state/action 专属嵌入用于对齐损失
        z_s = s_token.squeeze(1)                               # [B,E]
        z_a = self.pool_action(a_tokens)                       # [B,E]
        z_s = self.proj_s(z_s)                                 # [B,P]
        z_a = self.proj_a(z_a)                                 # [B,P]
        z_s = F.normalize(z_s, dim=-1)
        z_a = F.normalize(z_a, dim=-1)

        return z, z_s, z_a

    def forward(self, k_idx: torch.Tensor, t_idx: torch.Tensor,
                views: torch.Tensor, poses: torch.Tensor, actions: torch.Tensor):
        z, z_s, z_a = self.encode_tokens(k_idx, t_idx, views, poses, actions)
        en = self.energy_head(z).view(-1)
        return en, z, z_s, z_a
    
    def forward_batch(self, k_idx: torch.Tensor, t_idx: torch.Tensor,
                      views: torch.Tensor, poses: torch.Tensor, actions: torch.Tensor):
        """Process batch of 4 actions and return 4 energies.
        
        Args:
            k_idx: [B] tensor of guidance step indices
            t_idx: [B] tensor of time step indices  
            views: [B,V,C,H,W] multi-view images or None
            poses: [B,D_s] state/pose information
            actions: [B,4,T,D_a] batch of 4 action sequences
            
        Returns:
            energies: [B,4] tensor of 4 energy values
            z: [B,4,E] fused representations
            z_s: [B,4,P] state representations
            z_a: [B,4,P] action representations
        """
        B, num_actions = actions.shape[0], actions.shape[1]
        assert num_actions == 4, f"Expected 4 actions, got {num_actions}"
        
        # Prepare batch inputs by repeating k_idx, t_idx, views, poses for each action
        k_idx_batch = k_idx.unsqueeze(1).repeat(1, 4).view(-1)  # [B*4]
        t_idx_batch = t_idx.unsqueeze(1).repeat(1, 4).view(-1)  # [B*4]
        poses_batch = poses.unsqueeze(1).repeat(1, 4, 1).view(-1, poses.shape[-1])  # [B*4, D_s]
        
        # Handle views if present
        if views is not None:
            views_batch = views.unsqueeze(1).repeat(1, 4, 1, 1, 1, 1).view(-1, *views.shape[1:])  # [B*4, V, C, H, W]
        else:
            views_batch = None
            
        # Reshape actions for batch processing
        actions_batch = actions.view(-1, *actions.shape[2:])  # [B*4, T, D_a]
        
        # Process all 4 actions in parallel
        z_batch, z_s_batch, z_a_batch = self.encode_tokens(
            k_idx_batch, t_idx_batch, views_batch, poses_batch, actions_batch
        )
        
        # Compute energies for all 4 actions
        en_batch = self.energy_head(z_batch).view(B, 4)  # [B, 4]
        
        # Reshape outputs back to [B, 4, ...]
        z = z_batch.view(B, 4, -1)  # [B, 4, E]
        z_s = z_s_batch.view(B, 4, -1)  # [B, 4, P]
        z_a = z_a_batch.view(B, 4, -1)  # [B, 4, P]
        
        return en_batch, z, z_s, z_a
