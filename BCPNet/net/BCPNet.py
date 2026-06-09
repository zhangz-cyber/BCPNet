# Auto-generated BCPNet.py
# Renamed modules:
# BPG = BoundaryPriorGeneration
# BGCA = BoundaryGroundedContrastiveAlignment
# PSGD = ProgressiveSemanticGuidedDecoder
# DSPR = DynamicSemanticPromptRefinement

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Res2Net import res2net50_v1b_26w_4s
from .swin_backbone import SwinBackbone
from .pvt_v2_eff import pvt_v2_eff_b2, pvt_v2_eff_b3, pvt_v2_eff_b4, pvt_v2_eff_b5


# ============================================================
# Basic blocks
# ============================================================
class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MPD(nn.Module):
    """Modified Partial Decoder: align two adjacent levels and fuse them."""
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.fuse = nn.Sequential(
            ConvBNR(in_ch * 2, out_ch, 3),
            ConvBNR(out_ch, out_ch, 3),
        )

    def forward(self, low, high):
        if high.shape[2:] != low.shape[2:]:
            high = F.interpolate(
                high,
                size=low.shape[2:],
                mode='bilinear',
                align_corners=False,
            )

        return self.fuse(torch.cat([low, high], dim=1))


class BCS(nn.Module):
    """BConv + Conv + Sigmoid, used to predict a 1-channel prior map."""
    def __init__(self, in_ch):
        super().__init__()

        self.net = nn.Sequential(
            ConvBNR(in_ch, in_ch, 3),
            nn.Conv2d(in_ch, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Backbone wrappers
# ============================================================
class Res2NetBackbone(nn.Module):
    """Return 4 feature maps: 1/4, 1/8, 1/16, 1/32."""
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = res2net50_v1b_26w_4s(pretrained=pretrained)
        self.out_channels = (256, 512, 1024, 2048)

    def forward(self, x):
        e = self.encoder

        x = e.conv1(x)
        x = e.bn1(x)
        x = e.relu(x)
        x = e.maxpool(x)

        x1 = e.layer1(x)
        x2 = e.layer2(x1)
        x3 = e.layer3(x2)
        x4 = e.layer4(x3)

        return x1, x2, x3, x4


class PVTBackbone(nn.Module):
    def __init__(self, variant='pvt_v2_b2', pretrained=True):
        super().__init__()

        self.out_channels = (64, 128, 320, 512)

        if variant == 'pvt_v2_b2':
            self.encoder = pvt_v2_eff_b2(pretrained=pretrained)
        elif variant == 'pvt_v2_b3':
            self.encoder = pvt_v2_eff_b3(pretrained=pretrained)
        elif variant == 'pvt_v2_b4':
            self.encoder = pvt_v2_eff_b4(pretrained=pretrained)
        elif variant == 'pvt_v2_b5':
            self.encoder = pvt_v2_eff_b5(pretrained=pretrained)
        else:
            raise ValueError(f'Unsupported PVT variant: {variant}')

    def forward(self, x):
        endpoints = self.encoder(x)

        return (
            endpoints['reduction_2'],
            endpoints['reduction_3'],
            endpoints['reduction_4'],
            endpoints['reduction_5'],
        )


class DINOv2Backbone(nn.Module):
    """
    DINOv2 ViT backbone wrapper.

    Input:
        image: [B, 3, H, W]

    Output:
        x1: [B, 128, H/4,  W/4]
        x2: [B, 256, H/8,  W/8]
        x3: [B, 512, H/16, W/16]
        x4: [B, 768, H/32, W/32]

    DINOv2 ViT-B/14 在 448x448 输入下原始 patch feature 是 [B, 768, 32, 32]。
    这里取 4 个中间层 token feature，再插值成 4 个尺度，兼容 DSM。
    """
    def __init__(
        self,
        variant='dinov2_vitb14',
        ckpt_path='./models/dinov2/dinov2_vitb14_pretrain.pth',
        repo_or_dir='./external/dinov2',
        source='local',
        freeze=True,
        out_channels=(128, 256, 512, 768),
        out_indices=(2, 5, 8, 11),
    ):
        super().__init__()

        self.variant = variant
        self.ckpt_path = ckpt_path
        self.repo_or_dir = repo_or_dir
        self.source = source
        self.freeze = freeze
        self.out_indices = out_indices
        self.out_channels = out_channels

        if source == 'local':
            self.encoder = torch.hub.load(
                repo_or_dir,
                variant,
                source='local',
                pretrained=False
            )
        else:
            self.encoder = torch.hub.load(
                repo_or_dir,
                variant,
                pretrained=False
            )

        if ckpt_path is not None and len(str(ckpt_path)) > 0:
            state = torch.load(ckpt_path, map_location='cpu')

            if isinstance(state, dict):
                if 'teacher' in state:
                    state = state['teacher']
                elif 'model' in state:
                    state = state['model']
                elif 'state_dict' in state:
                    state = state['state_dict']

            new_state = {}
            for k, v in state.items():
                nk = k
                for prefix in ['module.', 'backbone.', 'encoder.']:
                    if nk.startswith(prefix):
                        nk = nk[len(prefix):]
                new_state[nk] = v

            msg = self.encoder.load_state_dict(new_state, strict=False)
            print(f'[DINOv2] Loaded checkpoint from: {ckpt_path}')
            print(f'[DINOv2] load_state_dict: {msg}')

        if 'vits14' in variant:
            self.embed_dim = 384
        elif 'vitb14' in variant:
            self.embed_dim = 768
        elif 'vitl14' in variant:
            self.embed_dim = 1024
        elif 'vitg14' in variant:
            self.embed_dim = 1536
        else:
            raise ValueError(f'Unsupported DINOv2 variant: {variant}')

        self.proj1 = Conv1x1(self.embed_dim, out_channels[0])
        self.proj2 = Conv1x1(self.embed_dim, out_channels[1])
        self.proj3 = Conv1x1(self.embed_dim, out_channels[2])
        self.proj4 = Conv1x1(self.embed_dim, out_channels[3])

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)

        if self.freeze:
            self.encoder.eval()

        return self

    def forward(self, x):
        b, _, h, w = x.shape

        if self.freeze:
            with torch.no_grad():
                feats = self.encoder.get_intermediate_layers(
                    x,
                    n=list(self.out_indices),
                    reshape=True,
                    return_class_token=False,
                    norm=True
                )
        else:
            feats = self.encoder.get_intermediate_layers(
                x,
                n=list(self.out_indices),
                reshape=True,
                return_class_token=False,
                norm=True
            )

        f1, f2, f3, f4 = list(feats)

        f1 = self.proj1(f1)
        f2 = self.proj2(f2)
        f3 = self.proj3(f3)
        f4 = self.proj4(f4)

        s1 = (max(1, h // 4),  max(1, w // 4))
        s2 = (max(1, h // 8),  max(1, w // 8))
        s3 = (max(1, h // 16), max(1, w // 16))
        s4 = (max(1, h // 32), max(1, w // 32))

        f1 = F.interpolate(f1, size=s1, mode='bilinear', align_corners=False)
        f2 = F.interpolate(f2, size=s2, mode='bilinear', align_corners=False)
        f3 = F.interpolate(f3, size=s3, mode='bilinear', align_corners=False)
        f4 = F.interpolate(f4, size=s4, mode='bilinear', align_corners=False)

        return f1, f2, f3, f4


# ============================================================
# Token-level text modules
# ============================================================
class TextProjector(nn.Module):
    """
    支持两种输入：

    1. 旧格式:
       [D] 或 [B, D]

    2. 新 token 格式:
       [L, D] 或 [B, L, D]

    输出:
       token_feat:  [B, L, H]
       global_feat: [B, H]
       token_mask:  [B, L]
    """
    def __init__(self, text_dim=512, hidden_dim=256, dropout=0.1, context_length=77):
        super().__init__()

        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length

        self.net = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, text_feat):
        text_feat = text_feat.float()

        if text_feat.dim() == 1:
            text_feat = text_feat.unsqueeze(0)

        if (
            text_feat.dim() == 2
            and text_feat.shape[-1] == self.text_dim
            and text_feat.shape[0] == self.context_length
        ):
            text_feat = text_feat.unsqueeze(0)

        # old global format: [B, D]
        if text_feat.dim() == 2:
            global_feat = F.normalize(text_feat, dim=-1)
            global_feat = self.net(global_feat)
            global_feat = F.normalize(global_feat, dim=-1)

            token_feat = global_feat.unsqueeze(1)
            token_mask = torch.ones(
                token_feat.shape[:2],
                dtype=torch.bool,
                device=token_feat.device
            )

            return token_feat, global_feat, token_mask

        # new token format: [B, L, D]
        if text_feat.dim() == 3:
            token_mask = text_feat.abs().sum(dim=-1) > 0

            token_feat = F.normalize(text_feat, dim=-1)
            token_feat = self.net(token_feat)
            token_feat = F.normalize(token_feat, dim=-1)

            token_feat = token_feat * token_mask.unsqueeze(-1).float()

            denom = token_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
            global_feat = token_feat.sum(dim=1) / denom
            global_feat = F.normalize(global_feat, dim=-1)

            return token_feat, global_feat, token_mask

        raise RuntimeError(f"Unsupported text feature shape: {tuple(text_feat.shape)}")


class TextFiLM(nn.Module):
    def __init__(self, text_hidden_dim, feat_channels, scale=0.1):
        super().__init__()

        self.gamma = nn.Linear(text_hidden_dim, feat_channels)
        self.beta = nn.Linear(text_hidden_dim, feat_channels)
        self.scale = scale

        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, t):
        gamma = torch.tanh(self.gamma(t)).unsqueeze(-1).unsqueeze(-1)
        beta = torch.tanh(self.beta(t)).unsqueeze(-1).unsqueeze(-1)

        return x * (1.0 + self.scale * gamma) + self.scale * beta


class TokenCrossAttention(nn.Module):
    """
    visual feature: [B, C, H, W]
    text tokens:    [B, L, T]

    使用 bottleneck attention，避免高通道 attention 显存过大。
    """
    def __init__(self, visual_dim, text_dim=256, attn_dim=64, num_heads=4, dropout=0.0):
        super().__init__()

        attn_dim = min(attn_dim, visual_dim)
        if attn_dim % num_heads != 0:
            num_heads = 1

        self.q_proj = nn.Linear(visual_dim, attn_dim)
        self.k_proj = nn.Linear(text_dim, attn_dim)
        self.v_proj = nn.Linear(text_dim, attn_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.out_proj = nn.Linear(attn_dim, visual_dim)

        self.norm1 = nn.LayerNorm(visual_dim)
        self.norm2 = nn.LayerNorm(visual_dim)

        self.ffn = nn.Sequential(
            nn.Linear(visual_dim, visual_dim * 2),
            nn.GELU(),
            nn.Linear(visual_dim * 2, visual_dim),
        )

    def forward(self, visual_feat, text_tokens=None, text_mask=None):
        if text_tokens is None:
            return visual_feat

        b, c, h, w = visual_feat.shape

        visual_tokens = visual_feat.flatten(2).transpose(1, 2)

        q = self.q_proj(visual_tokens)
        k = self.k_proj(text_tokens)
        v = self.v_proj(text_tokens)

        key_padding_mask = None
        if text_mask is not None:
            key_padding_mask = ~text_mask.bool()

        attn_out, _ = self.attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )

        x = self.norm1(visual_tokens + self.out_proj(attn_out))
        x = self.norm2(x + self.ffn(x))

        x = x.transpose(1, 2).reshape(b, c, h, w)

        return x


class SpatialCrossAttention2D(nn.Module):
    """
    query_feat:   [B, C, H, W]
    context_feat: [B, C, H2, W2]
    """
    def __init__(self, channels=64, attn_dim=64, num_heads=4, dropout=0.0):
        super().__init__()

        attn_dim = min(attn_dim, channels)
        if attn_dim % num_heads != 0:
            num_heads = 1

        self.q_proj = nn.Linear(channels, attn_dim)
        self.k_proj = nn.Linear(channels, attn_dim)
        self.v_proj = nn.Linear(channels, attn_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.out_proj = nn.Linear(attn_dim, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, query_feat, context_feat):
        if context_feat.shape[2:] != query_feat.shape[2:]:
            context_feat = F.interpolate(
                context_feat,
                size=query_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        b, c, h, w = query_feat.shape

        q_tokens = query_feat.flatten(2).transpose(1, 2)
        kv_tokens = context_feat.flatten(2).transpose(1, 2)

        q = self.q_proj(q_tokens)
        k = self.k_proj(kv_tokens)
        v = self.v_proj(kv_tokens)

        out, _ = self.attn(
            query=q,
            key=k,
            value=v,
            need_weights=False
        )

        out = self.norm(q_tokens + self.out_proj(out))
        out = out.transpose(1, 2).reshape(b, c, h, w)

        return out


class TextGuidedFeatureExtraction(nn.Module):
    """
    文本指导的特征提取 / 矫正模块。

    作用 1:
    视觉编码器对伪装特征提取不足时，用 token 级文本描述
    对 backbone 的多尺度视觉特征做 cross-attention 残差矫正。
    """
    def __init__(self, in_channels, text_dim=256, attn_dim=64, num_heads=4, init_scale=0.1):
        super().__init__()

        self.cross_attns = nn.ModuleList([
            TokenCrossAttention(
                visual_dim=c,
                text_dim=text_dim,
                attn_dim=attn_dim,
                num_heads=num_heads
            )
            for c in in_channels
        ])

        self.alpha = nn.Parameter(torch.ones(len(in_channels)) * init_scale)

    def forward(self, feats, text_tokens=None, text_mask=None):
        if text_tokens is None:
            return feats

        outs = []

        for i, feat in enumerate(feats):
            text_aware = self.cross_attns[i](
                visual_feat=feat,
                text_tokens=text_tokens,
                text_mask=text_mask
            )

            feat = feat + self.alpha[i] * (text_aware - feat)
            outs.append(feat)

        return outs


# ============================================================
class BoundaryPriorGeneration(nn.Module):
    def __init__(self, in_channels, mid_ch=64):
        super().__init__()

        self.reduce = nn.ModuleList([Conv1x1(c, mid_ch) for c in in_channels])

        self.high_mpd = MPD(mid_ch, mid_ch)
        self.low_mpd = MPD(mid_ch, mid_ch)

        self.region_head = BCS(mid_ch)
        self.boundary_head = BCS(mid_ch)

        self.pfg = nn.Sequential(
            ConvBNR(2, mid_ch, 3),
            nn.Conv2d(mid_ch, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feats):
        f1, f2, f3, f4 = [r(x) for r, x in zip(self.reduce, feats)]

        fch = self.high_mpd(f3, f4)
        mr_deep = self.region_head(fch)

        mr_f1 = F.interpolate(
            mr_deep,
            size=f1.shape[2:],
            mode='bilinear',
            align_corners=False,
        )
        mr_f2 = F.interpolate(
            mr_deep,
            size=f2.shape[2:],
            mode='bilinear',
            align_corners=False,
        )

        fcl = self.low_mpd(
            f1 * (1.0 + mr_f1),
            f2 * (1.0 + mr_f2),
        )

        mb = self.boundary_head(
            torch.abs(
                fcl - F.interpolate(
                    fch,
                    size=fcl.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )
            )
        )

        mr = F.interpolate(
            mr_deep,
            size=mb.shape[2:],
            mode='bilinear',
            align_corners=False,
        )

        alpha = self.pfg(torch.cat([mb, mr], dim=1))
        matt = alpha * mb + (1.0 - alpha) * mr

        return (f1, f2, f3, f4), matt, mb, mr



# ============================================================
class CrossModalMultiHeadAttention(nn.Module):
    """
    BCPNet CPG 中的 Cross-modal Multi-head Attention 适配版。

    用深层视觉特征作为 query，token 级文本作为 key/value，
    生成低层多模态特征 Fc。
    """
    def __init__(self, channels=64, text_dim=256, attn_dim=64, num_heads=4):
        super().__init__()

        self.cross = TokenCrossAttention(
            visual_dim=channels,
            text_dim=text_dim,
            attn_dim=attn_dim,
            num_heads=num_heads
        )

        self.refine = ConvBNR(channels, channels, 3)

    def forward(self, visual_feat, text_tokens=None, text_mask=None):
        x = self.cross(
            visual_feat=visual_feat,
            text_tokens=text_tokens,
            text_mask=text_mask
        )

        return self.refine(x)


class MultiLevelVisualCollaborationModule(nn.Module):
    """
    BCPNet CPG 中的 Multi-level Visual Collaboration Module 适配版。
    """
    def __init__(self, channels=64, attn_dim=64, num_heads=4):
        super().__init__()

        self.align2 = SpatialCrossAttention2D(channels, attn_dim, num_heads)
        self.align3 = SpatialCrossAttention2D(channels, attn_dim, num_heads)

        self.conv1a = ConvBNR(channels, channels, 3)
        self.conv1b = ConvBNR(channels, channels, 3)

        self.conv2a = ConvBNR(channels, channels, 3)
        self.conv2b = ConvBNR(channels, channels, 3)

        self.conv3a = ConvBNR(channels, channels, 3)
        self.conv3b = ConvBNR(channels, channels, 3)

        self.fuse = ConvBNR(channels * 3, channels, 3)

    def forward(self, fc, f2, f3):
        fn = self.align2(fc, f2) + self.align3(fc, f3)

        f1n = self.conv1b(self.conv1a(fn) + fn)

        x2 = fn * f1n
        f2n = self.conv2b(self.conv2a(x2) + x2)

        x3 = fn * f2n
        f3n = self.conv3b(self.conv3a(x3) + x3)

        fv = self.fuse(torch.cat([f1n, f2n, f3n], dim=1))

        return fv


class BoundaryGroundedContrastiveAlignment(nn.Module):
    """
    由 token 文本 + 多层视觉协作生成 Gc。

    对应红框里的两个模块：
        1. Cross-modal Multi-head Attention
        2. Multi-level Visual Collaboration Module
    """
    def __init__(self, channels=64, text_dim=256, attn_dim=64, num_heads=4):
        super().__init__()

        self.cma = CrossModalMultiHeadAttention(
            channels=channels,
            text_dim=text_dim,
            attn_dim=attn_dim,
            num_heads=num_heads
        )

        self.mvcm = MultiLevelVisualCollaborationModule(
            channels=channels,
            attn_dim=attn_dim,
            num_heads=num_heads
        )

        self.prompt_fuse = nn.Sequential(
            ConvBNR(channels * 2 + 1, channels, 3),
            ConvBNR(channels, channels, 3),
        )

    def forward(self, x2, x3, x4, matt, text_tokens=None, text_mask=None):
        m4 = F.interpolate(
            matt,
            size=x4.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        x4_guided = x4 * (1.0 + m4)

        fc = self.cma(
            visual_feat=x4_guided,
            text_tokens=text_tokens,
            text_mask=text_mask
        )

        fv = self.mvcm(fc, x2, x3)

        gc = self.prompt_fuse(torch.cat([fc, fv, m4], dim=1))

        return gc


class GroupGate(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class ProgressiveSemanticGuidedDecoder(nn.Module):
    """BCPNet SCM channel group refinement."""
    def __init__(self, channels):
        super().__init__()

        assert channels % 4 == 0, 'channels must be divisible by 4'

        group_ch = channels // 4

        self.convs = nn.ModuleList([
            ConvBNR(group_ch, group_ch, kernel_size=1),
            ConvBNR(group_ch, group_ch, kernel_size=3, dilation=2),
            ConvBNR(group_ch, group_ch, kernel_size=3, dilation=5),
            ConvBNR(group_ch, group_ch, kernel_size=3, dilation=7),
        ])

        self.gates = nn.ModuleList([
            GroupGate(group_ch)
            for _ in range(4)
        ])

        self.fuse = ConvBNR(channels, channels, 3)

    def forward(self, x):
        chunks = torch.chunk(x, 4, dim=1)

        outs = []
        for chunk, conv, gate in zip(chunks, self.convs, self.gates):
            outs.append(conv(chunk) * gate(chunk))

        return self.fuse(torch.cat(outs, dim=1))


class SemanticsConsistencyModule(nn.Module):
    """

    Input:
        xi, xj: adjacent visual features
        gc: class-semantic prompt feature

    Output:
        fs: refined feature
        gc_new: updated prompt
    """
    def __init__(self, channels=64):
        super().__init__()

        self.spatial_conv = nn.Conv2d(channels, 1, 3, padding=1)
        self.prompt_fc = Conv1x1(channels * 2, channels)

        self.xcot_conv = ConvBNR(channels * 2, channels, 3)
        self.psgd = ProgressiveSemanticGuidedDecoder(channels)
        self.out = ConvBNR(channels, channels, 3)

        self.prompt_update = ConvBNR(channels * 2, channels, 3)

    def forward(self, xi, xj, gc):
        if xj.shape[2:] != xi.shape[2:]:
            xj = F.interpolate(
                xj,
                size=xi.shape[2:],
                mode='bilinear',
                align_corners=False,
            )

        if gc.shape[2:] != xi.shape[2:]:
            gc = F.interpolate(
                gc,
                size=xi.shape[2:],
                mode='bilinear',
                align_corners=False,
            )

        r = self.spatial_conv((xi + xj) * gc)
        b, _, h, w = r.shape
        r = F.softmax(r.flatten(2), dim=-1).view(b, 1, h, w)

        xi_hat = xi * r
        xj_hat = xj * r

        g_prime = self.prompt_fc(torch.cat([xi_hat, xj_hat], dim=1))

        xi_tilde = xi_hat + g_prime
        xj_tilde = xj_hat + g_prime

        xcot = self.xcot_conv(torch.cat([xi_tilde, xj_tilde], dim=1))
        fs = self.out(self.psgd(xcot) + xcot)

        gc_new = self.prompt_update(torch.cat([fs, g_prime], dim=1))

        return fs, gc_new


class DynamicSemanticPromptRefinement(nn.Module):
    """
    DSM 后模块替换为 BCPNet-like:

        CPG red-box modules
          -> Cross-modal Multi-head Attention
          -> MVCM
          -> Gc
          -> SCM decoder
    """
    def __init__(self, channels=64, text_dim=256, attn_dim=64, num_heads=4):
        super().__init__()

        self.bgca = BoundaryGroundedContrastiveAlignment(
            channels=channels,
            text_dim=text_dim,
            attn_dim=attn_dim,
            num_heads=num_heads
        )

        self.scm3 = SemanticsConsistencyModule(channels)
        self.scm2 = SemanticsConsistencyModule(channels)
        self.scm1 = SemanticsConsistencyModule(channels)

        self.refine = nn.ModuleList([
            nn.Sequential(
                ConvBNR(channels, channels, 3),
                nn.Conv2d(channels, channels, 3, padding=1),
            )
            for _ in range(4)
        ])

    def forward(self, feats, matt, text_tokens=None, text_mask=None):
        x1, x2, x3, x4 = feats

        gc = self.bgca(
            x2=x2,
            x3=x3,
            x4=x4,
            matt=matt,
            text_tokens=text_tokens,
            text_mask=text_mask
        )

        x3, gc = self.scm3(x3, x4, gc)
        x2, gc = self.scm2(x2, x3, gc)
        x1, gc = self.scm1(x1, x2, gc)

        f4 = self.refine[0](
            x4 + F.interpolate(
                gc,
                size=x4.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
        )

        f3 = self.refine[1](
            x3 + F.interpolate(
                f4,
                size=x3.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
        )

        f2 = self.refine[2](
            x2 + F.interpolate(
                f3,
                size=x2.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
        )

        f1 = self.refine[3](
            x1 + F.interpolate(
                f2,
                size=x1.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
        )

        return f1, f2, f3, f4


# ============================================================
# Main Net
# ============================================================
class Net(nn.Module):
    def __init__(
        self,
        backbone='dinov2',
        swin_variant='swin_b_384_22k',
        swin_ckpt='./models/swin_base_patch4_window12_384_22k.pth',
        dinov2_variant='dinov2_vitb14',
        dinov2_ckpt='./models/dinov2/dinov2_vitb14_pretrain.pth',
        dinov2_repo='./external/dinov2',
        dinov2_source='local',
        use_text=False,
        text_dim=512,
        text_hidden_dim=256,
        text_dropout=0.1,
        text_scale=0.1,
        decoder_ch=64,
        return_aux=False,
    ):
        super().__init__()

        self.use_text = use_text
        self.return_aux = return_aux
        self.decoder_ch = decoder_ch

        if backbone == 'swin':
            self.backbone = SwinBackbone(
                variant=swin_variant,
                ckpt_path=swin_ckpt,
                pretrained=True,
            )
        elif backbone == 'pvt':
            pvt_variant = swin_variant if swin_variant.startswith('pvt_') else 'pvt_v2_b2'
            self.backbone = PVTBackbone(pvt_variant, pretrained=True)
        elif backbone == 'res2net':
            self.backbone = Res2NetBackbone(pretrained=True)
        elif backbone == 'dinov2':
            self.backbone = DINOv2Backbone(
                variant=dinov2_variant,
                ckpt_path=dinov2_ckpt,
                repo_or_dir=dinov2_repo,
                source=dinov2_source,
                freeze=True,
                out_channels=(128, 256, 512, 768),
                out_indices=(2, 5, 8, 11),
            )
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        C = self.backbone.out_channels

        self.bpg = BoundaryPriorGeneration(C, mid_ch=decoder_ch)

        if use_text:
            self.text_proj = TextProjector(
                text_dim=text_dim,
                hidden_dim=text_hidden_dim,
                dropout=text_dropout,
                context_length=77,
            )

            self.film = nn.ModuleList([
                TextFiLM(text_hidden_dim, c, scale=text_scale)
                for c in C
            ])

            self.text_guided_extraction = TextGuidedFeatureExtraction(
                in_channels=C,
                text_dim=text_hidden_dim,
                attn_dim=64,
                num_heads=4,
                init_scale=0.1
            )

            self.text_to_prompt = nn.Linear(text_hidden_dim, decoder_ch)

            self.visual_align = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(decoder_ch, text_hidden_dim),
                nn.LayerNorm(text_hidden_dim),
            )
        else:
            self.text_proj = None
            self.film = None
            self.text_guided_extraction = None
            self.text_to_prompt = None
            self.visual_align = None

        self.dspr = DynamicSemanticPromptRefinement(
            channels=decoder_ch,
            text_dim=text_hidden_dim,
            attn_dim=64,
            num_heads=4
        )

        self.pred1 = nn.Conv2d(decoder_ch, 1, 1)
        self.pred2 = nn.Conv2d(decoder_ch, 1, 1)
        self.pred3 = nn.Conv2d(decoder_ch, 1, 1)
        self.pred4 = nn.Conv2d(decoder_ch, 1, 1)

    def forward(self, x, text=None):
        size = x.shape[2:]

        feats_raw = list(self.backbone(x))

        text_tokens = None
        text_mask = None
        text_feat_proj = None

        if self.use_text and text is not None:
            text_tokens, text_feat_proj, text_mask = self.text_proj(text)

            feats_raw = [
                film(f, text_feat_proj)
                for film, f in zip(self.film, feats_raw)
            ]

            feats_raw = self.text_guided_extraction(
                feats_raw,
                text_tokens=text_tokens,
                text_mask=text_mask
            )

        feats, matt, mb, mr = self.bpg(feats_raw)

        if self.use_text and text_feat_proj is not None:
            tp = self.text_to_prompt(text_feat_proj).unsqueeze(-1).unsqueeze(-1)
            feats = tuple(f + tp for f in feats)

        d1, d2, d3, d4 = self.dspr(
            feats,
            matt,
            text_tokens=text_tokens,
            text_mask=text_mask
        )

        o1 = F.interpolate(
            self.pred1(d1),
            size=size,
            mode='bilinear',
            align_corners=False,
        )

        if not self.return_aux:
            if self.use_text and text_feat_proj is not None:
                visual_feat = F.normalize(self.visual_align(d1), dim=-1)
                return o1, visual_feat, text_feat_proj

            return o1

        o2 = F.interpolate(
            self.pred2(d2),
            size=size,
            mode='bilinear',
            align_corners=False,
        )

        o3 = F.interpolate(
            self.pred3(d3),
            size=size,
            mode='bilinear',
            align_corners=False,
        )

        o4 = F.interpolate(
            self.pred4(d4),
            size=size,
            mode='bilinear',
            align_corners=False,
        )

        mb = F.interpolate(
            mb,
            size=size,
            mode='bilinear',
            align_corners=False,
        )

        mr = F.interpolate(
            mr,
            size=size,
            mode='bilinear',
            align_corners=False,
        )

        matt = F.interpolate(
            matt,
            size=size,
            mode='bilinear',
            align_corners=False,
        )

        if self.use_text and text_feat_proj is not None:
            visual_feat = F.normalize(self.visual_align(d1), dim=-1)
            return o1, visual_feat, text_feat_proj

        return o1, o2, o3, o4, mb, mr, matt