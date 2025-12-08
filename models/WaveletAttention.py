import torch
import torch.nn as nn
import torch.nn.functional as F

class HaarDWT(nn.Module):
    """ 2D Haar wavelet transform """
    def __init__(self):
        super().__init__()
        # shape: [out_ch, in_ch, 2, 2]
        self.register_buffer('f_LL', torch.tensor([[1, 1], [1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 2)
        self.register_buffer('f_LH', torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 2)
        self.register_buffer('f_HL', torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 2)
        self.register_buffer('f_HH', torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 2)

    def forward(self, x):
        # x: [B, C, H, W]
        LL = F.conv2d(x, self.f_LL.expand(x.size(1), -1, 2, 2), stride=2, groups=x.size(1))
        LH = F.conv2d(x, self.f_LH.expand(x.size(1), -1, 2, 2), stride=2, groups=x.size(1))
        HL = F.conv2d(x, self.f_HL.expand(x.size(1), -1, 2, 2), stride=2, groups=x.size(1))
        HH = F.conv2d(x, self.f_HH.expand(x.size(1), -1, 2, 2), stride=2, groups=x.size(1))
        return LL, LH, HL, HH  # [B, C, H//2, W//2]

class GCBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 1, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.conv2 = nn.Conv2d(in_ch, in_ch, 1)
        self.norm = nn.LayerNorm(in_ch)  # 改成只对channel归一化
        self.act = nn.ReLU()
        self.out_proj = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x):
        w = self.conv1(x)  # [B, 1, H, W]
        w = w.view(x.shape[0], -1)
        w = self.softmax(w)
        w = w.view(x.shape[0], 1, x.shape[2], x.shape[3])
        attn = x * w
        attn = self.conv2(attn)
        # LayerNorm over channel, permute to [B, H, W, C]
        attn = attn.permute(0, 2, 3, 1)  # [B, H, W, C]
        attn = self.norm(attn)
        attn = attn.permute(0, 3, 1, 2)  # [B, C, H, W]
        attn = self.act(attn)
        attn = self.out_proj(attn)
        return x + attn


class WaveletSpatialAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.dwt = HaarDWT()
        self.conv1 = nn.Conv2d(in_ch, in_ch // 2, 1)
        self.branch_convs = nn.ModuleList([
            nn.Conv2d(in_ch // 2, 1, (7, 3), padding=(3, 1), dilation=1),
            nn.Conv2d(in_ch // 2, 1, (3, 7), padding=(1, 3), dilation=1),
            nn.Conv2d(in_ch // 2, 1, (5, 2), padding=(2, 0), dilation=1),
            nn.Conv2d(in_ch // 2, 1, (2, 5), padding=(1, 2), dilation=1),
            nn.Conv2d(in_ch // 2, 1, (1, 1), padding=(0, 0), dilation=1),
        ])
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.conv7 = nn.Conv2d(6, 1, 7, padding=3)  # 5分支+1池化
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # 1. DWT
        LL, LH, HL, HH = self.dwt(x)  # 全 [B, C, H//2, W//2]
        high = LH + HL + HH  # [B, C, H//2, W//2]
        low = LL  # [B, C, H//2, W//2]
        # 2. Fuse features and restore original size
        high = F.interpolate(self.conv1(high), size=(H, W), mode='bilinear', align_corners=False)
        low = F.interpolate(self.conv1(low), size=(H, W), mode='bilinear', align_corners=False)
        wavelet_feat = high * low

        # 3. multi-scale branch
        branches = []
        for conv in self.branch_convs:
            out = conv(wavelet_feat)
            if out.shape[2:] != (H, W):
                out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            branches.append(out)
        maxp = self.maxpool(wavelet_feat).mean(dim=1, keepdim=True)
        maxp = maxp.expand(-1, 1, H, W)  # [B, 1, H, W]
        branches.append(maxp)
        multi_scale = torch.cat(branches, dim=1)  # [B, 6, H, W]
        attn = self.sigmoid(self.conv7(multi_scale))
        out = x * attn
        return out

class WaveletGlobalAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, in_ch)
        self.wavelet_sa = WaveletSpatialAttention(in_ch)
        self.norm2 = nn.GroupNorm(1, in_ch)
        self.gc_block = GCBlock(in_ch)

    def forward(self, x):
        x0 = x
        x = self.norm1(x)
        x = self.wavelet_sa(x)
        x = x + x0
        x = self.norm2(x)
        x = self.gc_block(x)
        return x