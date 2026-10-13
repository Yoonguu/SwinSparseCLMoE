"""
SwinSparseCLMOE: Sparsely-Gated Mixture-of-Experts with Contrastive Learning
for Efficient Brain Tumor Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- helpers --------------------
def to_3tuple(x):
    if isinstance(x, tuple): return x
    return (x, x, x)


def window_partition_3d(x, window_size):
    B, D, H, W, C = x.shape
    Wd, Wh, Ww = window_size
    assert D % Wd == 0 and H % Wh == 0 and W % Ww == 0, \
        f"Input (D,H,W)=({D},{H},{W}) must be multiples of window_size ({Wd},{Wh},{Ww})"
    x = x.reshape(B, D // Wd, Wd, H // Wh, Wh, W // Ww, Ww, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, Wd * Wh * Ww, C)
    return x


def window_reverse_3d(windows, window_size, B, D, H, W, C):
    Wd, Wh, Ww = window_size
    x = windows.reshape(B, D // Wd, H // Wh, W // Ww, Wd, Wh, Ww, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, D, H, W, C)
    return x


# -------------------- attention --------------------
class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size=(2, 7, 7), num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = to_3tuple(window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        Wd, Wh, Ww = self.window_size
        coords_d = torch.arange(Wd)
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flat = torch.flatten(coords, 1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += Wd - 1
        rel[:, :, 1] += Wh - 1
        rel[:, :, 2] += Ww - 1
        rel[:, :, 0] *= (2 * Wh - 1) * (2 * Ww - 1)
        rel[:, :, 1] *= (2 * Ww - 1)
        rel_index = rel.sum(-1)
        self.register_buffer("rel_pos_index", rel_index, persistent=False)
        self.rel_pos_bias = nn.Parameter(torch.zeros((2 * Wd - 1) * (2 * Wh - 1) * (2 * Ww - 1), num_heads))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rel_bias = self.rel_pos_bias[self.rel_pos_index.reshape(-1)].reshape(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -------------------- MLP / DropPath --------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand.floor_()
        return x.div(keep_prob) * rand


# -------------------- Sparsely-Gated MoE --------------------
class SparselyGatedMoE(nn.Module):
    """
    Sparsely-Gated Mixture-of-Experts Layer with Top-k Routing and Load-Balancing Loss
    """
    def __init__(self, input_dim, out_channels, num_experts=4, top_k=2, aux_loss_weight=1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.aux_loss_weight = aux_loss_weight
        self.out_channels = out_channels

        # 전문가 네트워크
        self.experts = nn.ModuleList([
            nn.Conv3d(input_dim, out_channels, kernel_size=1) for _ in range(num_experts)
        ])

        # 게이팅 네트워크
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(input_dim, num_experts)
        )

    def forward(self, x):
        B, _, D, H, W = x.shape

        # 1. Gate Logits & Top-k Routing
        gate_logits = self.gate(x)  # (B, num_experts)
        top_k_weights, top_k_indices = torch.topk(gate_logits, self.top_k, dim=1)
        top_k_weights = F.softmax(top_k_weights, dim=1).to(x.dtype)  # (B, top_k)

        # 2. Auxiliary Load-Balancing Loss (Eq. 5 논문 수식 반영)
        router_probs = F.softmax(gate_logits, dim=1)           # (B, num_experts)
        expert_importance = router_probs.sum(dim=0)            # 배치 내 Importance

        expert_mask = torch.zeros_like(gate_logits).scatter_(1, top_k_indices, 1.0)
        expert_load = expert_mask.mean(dim=0)                  # 실제 선택된 비율 (Load)

        aux_loss = self.num_experts * torch.sum(expert_importance * expert_load)
        aux_loss = aux_loss * self.aux_loss_weight

        # 3. Memory & Calculation Efficient Dynamic Aggregation
        final_output = torch.zeros((B, self.out_channels, D, H, W), device=x.device, dtype=x.dtype)

        # 배치별, top-k별 안전한 인덱싱 및 가중치 합산
        for b in range(B):
            for k in range(self.top_k):
                expert_idx = top_k_indices[b, k].item()
                weight = top_k_weights[b, k]
                expert_out = self.experts[expert_idx](x[b:b+1])  # (1, out_channels, D, H, W)
                final_output[b:b+1] += expert_out * weight

        return final_output, aux_loss


# -------------------- Swin Block --------------------
class SwinBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=None, mlp_ratio=4.0, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.window_size = to_3tuple(window_size)
        if shift_size is None:
            self.shift_size = tuple(w // 2 for w in self.window_size)
        else:
            self.shift_size = to_3tuple(shift_size)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim=dim, window_size=self.window_size, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)

        # DropPath 적용
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def _build_attn_mask(self, device, Dp, Hp, Wp):
        Wd, Wh, Ww = self.window_size
        Sd, Sh, Sw = self.shift_size
        img_mask = torch.zeros((1, Dp, Hp, Wp, 1), device=device)
        if any(s > 0 for s in (Sd, Sh, Sw)):
            d_slices = (slice(0, -Wd), slice(-Wd, -Sd), slice(-Sd, None)) if Sd > 0 else (slice(0, Dp),)
            h_slices = (slice(0, -Wh), slice(-Wh, -Sh), slice(-Sh, None)) if Sh > 0 else (slice(0, Hp),)
            w_slices = (slice(0, -Ww), slice(-Ww, -Sw), slice(-Sw, None)) if Sw > 0 else (slice(0, Wp),)
        else:
            d_slices = (slice(0, Dp),)
            h_slices = (slice(0, Hp),)
            w_slices = (slice(0, Wp),)
        cnt = 0
        for d in d_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
        mask_windows = window_partition_3d(img_mask, self.window_size)
        mask_windows = mask_windows.reshape(-1, Wd * Wh * Ww)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x).reshape(B, D, H, W, C)
        Wd, Wh, Ww = self.window_size
        d_pad = (Wd - D % Wd) % Wd
        h_pad = (Wh - H % Wh) % Wh
        w_pad = (Ww - W % Ww) % Ww
        if d_pad or h_pad or w_pad:
            x = F.pad(x, (0, 0, 0, w_pad, 0, h_pad, 0, d_pad))
            Dp, Hp, Wp = D + d_pad, H + h_pad, W + w_pad
        else:
            Dp, Hp, Wp = D, H, W

        Sd, Sh, Sw = self.shift_size
        if any(s > 0 for s in (Sd, Sh, Sw)):
            shifted = torch.roll(x, shifts=(-Sd, -Sh, -Sw), dims=(1, 2, 3))
            attn_mask = self._build_attn_mask(x.device, Dp, Hp, Wp)
        else:
            shifted = x
            attn_mask = None

        x_windows = window_partition_3d(shifted, self.window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        shifted_back = window_reverse_3d(attn_windows, self.window_size, B, Dp, Hp, Wp, C)
        if any(s > 0 for s in (Sd, Sh, Sw)):
            x = torch.roll(shifted_back, shifts=(Sd, Sh, Sw), dims=(1, 2, 3))
        else:
            x = shifted_back

        if d_pad or h_pad or w_pad:
            x = x[:, :D, :H, :W, :]

        x = x.reshape(B, N, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# -------------------- Patch Merging & Embed --------------------
class PatchMerging3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(8 * dim)
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.reshape(B, D, H, W, C)
        pd, ph, pw = D % 2, H % 2, W % 2
        if pd or ph or pw:
            x = F.pad(x, (0, 0, 0, pw, 0, ph, 0, pd))
            D += pd; H += ph; W += pw
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
        x = x.reshape(B, -1, 8 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, (D // 2, H // 2, W // 2)


class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans=4, embed_dim=96, patch_size=(2, 2, 2)):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().reshape(B, -1, C)
        return x, (D, H, W)


# -------------------- Decoder Block --------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch * 2, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )

    def _align_to(self, x, ref):
        Dz, Dy, Dx = x.size(2), x.size(3), x.size(4)
        Rz, Ry, Rx = ref.size(2), ref.size(3), ref.size(4)
        if Dz > Rz: x = x[:, :, :Rz, :, :]
        if Dy > Ry: x = x[:, :, :, :Ry, :]
        if Dx > Rx: x = x[:, :, :, :, :Rx]
        pad_z = max(Rz - x.size(2), 0)
        pad_y = max(Ry - x.size(3), 0)
        pad_x = max(Rx - x.size(4), 0)
        if pad_z or pad_y or pad_x:
            x = F.pad(x, (0, pad_x, 0, pad_y, 0, pad_z))
        return x

    def forward(self, x, skip):
        x = self.up(x)
        x = self._align_to(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# -------------------- Main Model: SwinSparseCLMOE --------------------
class SwinSparseCLMOE(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_dim=24,
                 depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                 window_size=(2, 7, 7), patch_size=(2, 2, 2),
                 expert_count=4, top_k=2, aux_loss_weight=1e-2,
                 projection_dim=128, drop_path_rate=0.0):
        super().__init__()
        self.embed = PatchEmbed3D(in_channels, base_dim, patch_size)
        self.window_size = to_3tuple(window_size)
        self.depths = depths
        self.num_heads = num_heads

        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        idx = 0

        # Encoder stages
        self.stages = nn.ModuleList()
        for si, (dim, depth, heads) in enumerate(zip(dims, depths, num_heads)):
            blocks = []
            for bi in range(depth):
                shift = (0, 0, 0) if (bi % 2 == 0) else tuple(w // 2 for w in self.window_size)
                blocks.append(SwinBlock3D(dim=dim, num_heads=heads, window_size=self.window_size, shift_size=shift, drop_path=dpr[idx]))
                idx += 1
            self.stages.append(nn.ModuleList(blocks))

        # Downsampling
        self.downs = nn.ModuleList()
        for si in range(len(depths) - 1):
            self.downs.append(PatchMerging3D(dim=dims[si]))

        # Decoder
        self.up4 = UpBlock(2 * dims[3], 2 * dims[2])
        self.up3 = UpBlock(2 * dims[2], 2 * dims[1])
        self.up2 = UpBlock(2 * dims[1], 2 * dims[0])
        self.up1 = nn.ConvTranspose3d(2 * dims[0], dims[0] // 2, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv3d(dims[0] // 2, dims[0] // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(dims[0] // 2), nn.ReLU(inplace=True)
        )

        # Sparsely-Gated MoE Head
        self.moe_head = SparselyGatedMoE(
            input_dim=dims[0] // 2,
            out_channels=out_channels,
            num_experts=expert_count,
            top_k=top_k,
            aux_loss_weight=aux_loss_weight
        )

        # Contrastive Projection Head
        self.projection = nn.Sequential(
            nn.Linear(dims[3], dims[3] // 2), nn.ReLU(inplace=True),
            nn.Linear(dims[3] // 2, projection_dim)
        )

    def forward(self, x1, x2=None):
        # x2가 주어지지 않은 단일 입력 경우 처리
        if x2 is None:
            x2 = x1

        # Patch Embed
        x1, (D, H, W) = self.embed(x1)
        x2, (D, H, W) = self.embed(x2)
        B, N, C = x1.shape
        cur_dim = C

        cur_res1, cur_res2 = (D, H, W), (D, H, W)
        encoder_feats1, encoder_feats2 = [], []

        # Encoder Stage Forward
        for si, blocks in enumerate(self.stages):
            for blk in blocks:
                x1 = blk(x1, *cur_res1)
                x2 = blk(x2, *cur_res2)
            x3d_1 = x1.reshape(B, cur_res1[0], cur_res1[1], cur_res1[2], cur_dim).permute(0, 4, 1, 2, 3).contiguous()
            x3d_2 = x2.reshape(B, cur_res2[0], cur_res2[1], cur_res2[2], cur_dim).permute(0, 4, 1, 2, 3).contiguous()

            encoder_feats1.append(x3d_1)
            encoder_feats2.append(x3d_2)
            if si < len(self.stages) - 1:
                x1, cur_res1 = self.downs[si](x1, *cur_res1)
                x2, cur_res2 = self.downs[si](x2, *cur_res2)
                cur_dim *= 2

        # Decoder Forward
        x1_enc4, x1_enc3, x1_enc2, x1_enc1 = encoder_feats1[-1], encoder_feats1[-2], encoder_feats1[-3], encoder_feats1[-4]
        x2_enc4, x2_enc3, x2_enc2, x2_enc1 = encoder_feats2[-1], encoder_feats2[-2], encoder_feats2[-3], encoder_feats2[-4]

        x_enc4 = torch.cat([x1_enc4, x2_enc4], dim=1)
        x_enc3 = torch.cat([x1_enc3, x2_enc3], dim=1)
        x_enc2 = torch.cat([x1_enc2, x2_enc2], dim=1)
        x_enc1 = torch.cat([x1_enc1, x2_enc1], dim=1)

        d3 = self.up4(x_enc4, x_enc3)
        d2 = self.up3(d3, x_enc2)
        d1 = self.up2(d2, x_enc1)
        u0 = self.up1(d1)
        seg_feat = self.final_conv(u0)

        # Sparsely-Gated MoE Output & Auxiliary Loss
        logits, aux_loss = self.moe_head(seg_feat)

        # Bottleneck Contrastive Projection
        deep_pool1 = F.adaptive_avg_pool3d(x1_enc4, 1).view(x1_enc4.size(0), -1)
        deep_pool2 = F.adaptive_avg_pool3d(x2_enc4, 1).view(x2_enc4.size(0), -1)
        proj1 = F.normalize(self.projection(deep_pool1), dim=1)
        proj2 = F.normalize(self.projection(deep_pool2), dim=1)

        return seg_feat, logits, proj1, proj2, aux_loss