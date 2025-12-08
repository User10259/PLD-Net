import torchvision.models as tv
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from .WaveletAttention import WaveletGlobalAttention
from .UND import UND_Block

class ConvNeXtBlock(nn.Module):
    """
      1) Depthwise convolution (kernel_size × kernel_size, groups=dim)
      2) Permute to NHWC format
      3) LayerNorm (perform normalization on the dim dimension after Depthwise Conv and before MLP)
      4) MLP (two nn.Linear layers with GELU in between)
      5) Permute back to NCHW format
      6) Residual connection: out = identity + out
    """

    def __init__(self, dim, expansion: int = 4, kernel_size: int = 7):
        super().__init__()
        self.dim = dim
        self.expansion = expansion

        # 1) Depthwise conv
        self.dwconv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=True
        )

        # 2) LayerNorm
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # 3) MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim)
        )

    def forward(self, x):
        # x: [B, dim, H, W]
        identity = x

        # 1) Depthwise
        out = self.dwconv(x)  # [B, dim, H, W]

        # 2) Permute to NHWC format for LayerNorm and MLP operations
        out = out.permute(0, 2, 3, 1)  # [B, H, W, dim]

        # 3) LayerNorm
        out = self.norm(out)  # 对最后一维 (dim) 做 LayerNorm

        # 4) MLP
        out = self.mlp(out)  # [B, H, W, dim]

        # 5) Permute back to NCHW format for residual connection
        out = out.permute(0, 3, 1, 2)  # [B, dim, H, W]

        # 6) residual connection
        return identity + out  # [B, dim, H, W]


#Downsample
class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)  # LayerNorm on C
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = self.conv(x)  # Conv2d
        return x


class PLDNet_Backbone(nn.Module):
    """
    Consistent with the standard timm ConvNeXt-Tiny (pre-trained on ImageNet-1K),
    """

    def __init__(self, pretrained=True, pretrained_path=None):
        super().__init__()
        # ConvNeXt-Tiny structure parameters
        # Official ConvNeXt-Tiny parameters: dims=[96, 192, 384, 768], depths=[3, 3, 9, 3]

        # ===== Stage 1 (Stem + Stage 0 Blocks): Output 128×128×96 =====
        # Corresponding to the stem and stages[0] of timm ConvNeXt
        # The stem of timm includes Conv2d and LayerNorm (channels_last)
        # For simplification and matching purposes, we only copy the weights of Conv2d, while BatchNorm remains unchanged
        self.st1_down = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=4, stride=4, padding=0, bias=True),  # ConvNeXt stem 通常有 bias
            nn.BatchNorm2d(96),
            nn.GELU()
        )
        self.st1_blocks = nn.Sequential(*[
            ConvNeXtBlock(dim=96, expansion=4, kernel_size=7)
            for _ in range(3)
        ])

        # ===== Stage 2 (Stage 1 downsample + Stage 1 Blocks): Output 64×64×192 =====
        # Corresponding to stages[1].downsample and stages[1].blocks of timm ConvNeXt
        self.st2_down = DownsampleLayer(96, 192, stride=2)  # 使用自定义的 DownsampleLayer
        self.st2_blocks = nn.Sequential(*[
            ConvNeXtBlock(dim=192, expansion=4, kernel_size=7)
            for _ in range(3)  # ConvNeXt-Tiny Stage 1 有 3 个 Block
        ])

        # ===== Stage 3 (Stage 2 downsample + Stage 2 Blocks): Output 32×32×384 =====
        self.st3_down = DownsampleLayer(192, 384, stride=2)
        self.st3_blocks = nn.Sequential(*[
            ConvNeXtBlock(dim=384, expansion=4, kernel_size=7)
            for _ in range(9)
        ])

        # ===== Stage 4 (Stage 3 downsample + Stage 3 Blocks): Output 16×16×768 =====
        self.st4_down = DownsampleLayer(384, 768, stride=2)
        self.st4_blocks = nn.Sequential(*[
            ConvNeXtBlock(dim=768, expansion=4, kernel_size=7)
            for _ in range(3)
        ])

        # ===== Stage 5: ConvNeXt-5th stage，Output 16×16×384 =====
        self.st5_proj = nn.Conv2d(768, 384, kernel_size=1, bias=False)
        self.st5_blocks = nn.Sequential(*[
            ConvNeXtBlock(dim=384, expansion=4, kernel_size=7)
            for _ in range(9)
        ])

        # Stack multiple layers of Attention on the 384-dimensional features output in Stage 5
        self.attention_neck = nn.Sequential(*[
            WaveletGlobalAttention(384) for _ in range(4)
        ])
        # === Define the UND module in the last stage ===
        self.und_block = UND_Block(384)

        # --- Pre-trained weight loading logic ---
        if pretrained:
            try:
                if pretrained_path:
                    convnext_timm = timm.create_model('convnext_tiny', pretrained=False)
                    state = torch.load(pretrained_path, map_location='cpu')
                    convnext_timm.load_state_dict(state, strict=False)
                    # print(f"Loaded pretrained weights from {pretrained_path} into timm model.")
                else:
                    convnext_timm = timm.create_model('convnext_tiny', pretrained=True)
                    # print("Loaded official ConvNeXt-Tiny pretrained weights (via timm download).")

                timm_state_dict = convnext_timm.state_dict()
                my_model_state_dict = self.state_dict()

                def map_timm_key_to_my_key(timm_key):
                    if timm_key.startswith('stem.0.'):
                        return 'st1_down.0.' + timm_key[len('stem.0.'):]
                    if '.downsample.' in timm_key:
                        parts = timm_key.split('.')
                        stage_idx = int(parts[1])  # 1, 2, 3
                        module_idx = parts[3]  # 0 for LayerNorm, 1 for Conv2d
                        param_name = parts[4]  # weight or bias

                        my_stage_prefix = f'st{stage_idx + 1}_down.'  # stages.1 -> st2_down

                        if module_idx == '0':  # LayerNorm
                            return f'{my_stage_prefix}norm.{param_name}'
                        elif module_idx == '1':  # Conv2d
                            return f'{my_stage_prefix}conv.{param_name}'
                    if '.blocks.' in timm_key:
                        parts = timm_key.split('.')
                        stage_idx = int(parts[1])  # 0, 1, 2, 3
                        block_idx = int(parts[3])  # Y
                        block_prefix = f'st{stage_idx + 1}_blocks.{block_idx}.'

                        if 'conv_dw' in timm_key:  # Depthwise Conv
                            return block_prefix + 'dwconv.' + parts[-1]
                        elif 'norm' in timm_key and 'mlp' not in timm_key:
                            return block_prefix + 'norm.' + parts[-1]
                        elif 'mlp.fc1' in timm_key:
                            return block_prefix + 'mlp.0.' + parts[-1]
                        elif 'mlp.fc2' in timm_key:
                            return block_prefix + 'mlp.2.' + parts[-1]

                    return None

                new_state_dict = {}
                for timm_key, timm_value in timm_state_dict.items():
                    my_key = map_timm_key_to_my_key(timm_key)
                    if my_key and my_key in my_model_state_dict:
                        if my_model_state_dict[my_key].shape == timm_value.shape:
                            new_state_dict[my_key] = timm_value
                        else:
                            print(
                                f"Warning: Shape mismatch for {my_key} (My: {my_model_state_dict[my_key].shape}, Timm: {timm_value.shape}), skipping this specific parameter."
                            )

                self.load_state_dict(new_state_dict, strict=False)
                print("Custom backbone loaded relevant pretrained weights with mapping.")

            except Exception as e:
                print(f"Warning: Failed to load pretrained weights. Error: {e}")

    def forward(self, x):
        # x: [B, 3, 512, 512]
        x = self.st1_down(x)  # → [B,  96, 128, 128]
        x = self.st1_blocks(x)  # → [B,  96, 128, 128]

        x = self.st2_down(x)  # → [B, 192,  64,  64]
        x = self.st2_blocks(x)  # → [B, 192,  64,  64]

        x = self.st3_down(x)  # → [B, 384,  32,  32]
        x = self.st3_blocks(x)  # → [B, 384,  32,  32]

        x = self.st4_down(x)  # → [B, 768,  16,  16]
        x = self.st4_blocks(x)  # → [B, 768,  16,  16]

        x = self.st5_proj(x)  # → [B, 384, 16, 16]
        x = self.st5_blocks(x)  # → [B, 384, 16, 16]

        x = self.attention_neck(x)
        x = self.und_block(x)

        return x


class PLDNet(nn.Module):
    """
    PLDNet (fixed output 16×16 grid):
      1) Backbone output [B,384,16,16]
      2) Classification Head: 1×1 Convolution → [B, 2N,16,16] → Reshape [B,16,16,N,2]
      3) Regression Head: 1×1 Convolution → [B, 3N,16,16] → Sigmoid → Reshape [B,16,16,N,3]
    Return:
      pred_logits: [B,16,16,N,2]
      pred_lines : [B,16,16,N,3]
    """

    def __init__(self,
                 num_lines: int = 10,
                 backbone_type: str = "convnext-mini",
                 pretrained_backbone: bool = True,
                 pretrained_path: str = None):
        super().__init__()
        self.num_lines = num_lines
        self.backbone_type = backbone_type.lower()
        self.backbone = PLDNet_Backbone(pretrained_backbone, pretrained_path)

        # 3) cls Head: [B,384,16,16] → [B,2N,16,16]
        self.cls_head = nn.Conv2d(
            in_channels=384,
            out_channels=2 * num_lines,
            kernel_size=1,
            bias=True
        )
        # 4) reg Head: [B,384,16,16] → [B,3N,16,16]
        self.reg_head = nn.Conv2d(
            in_channels=384,
            out_channels=3 * num_lines,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):
        """
        :param x: [B,3,512,512]
        :return: {
            'pred_logits': [B,16,16,N,2],
            'pred_lines' : [B,16,16,N,3]
        }
        """
        B = x.shape[0]

        # 1) Backbone → [B,384,16,16]
        feat = self.backbone(x)

        G = 16

        # 3) cls Head: [B,384,16,16] → [B,2N,16,16]
        logits = self.cls_head(feat)  # [B, 2N,16,16]
        logits = logits.view(B, 2 * self.num_lines, G, G)
        logits = logits.permute(0, 2, 3, 1)  # → [B,16,16,2N]
        logits = logits.view(B, G, G, self.num_lines, 2)

        # 4) reg Head: [B,384,16,16] → [B,3N,16,16]
        regs = self.reg_head(feat)  # [B, 3N,16,16]
        regs = regs.view(B, 3 * self.num_lines, G, G)
        regs = regs.permute(0, 2, 3, 1).contiguous()
        regs = regs.view(B, G, G, self.num_lines, 3)

        pred_d = torch.sigmoid(regs[..., 0:1])
        pred_ang = regs[..., 1:]
        regs = torch.cat([pred_d, pred_ang], dim=-1)

        return {
            'pred_logits': logits,  # [B,16,16,N,2]
            'pred_lines': regs,  # [B,16,16,N,3]
        }
