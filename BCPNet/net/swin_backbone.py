import torch
import torch.nn as nn
from net.swin_encoder import SwinTransformer  # 确保此模块存在

class SwinBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(SwinBackbone, self).__init__()

        # ✅ 创建与 swin_base_patch4_window12_384_22k.pth 匹配的结构
        self.encoder = SwinTransformer(
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12
        )

        if pretrained:
            ckpt = torch.load('./models/swin_base_patch4_window12_384_22k.pth', map_location='cpu')
            if "model" in ckpt:
                ckpt = ckpt["model"]
            ckpt = {k: v for k, v in ckpt.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(ckpt, strict=False)
            print("[SwinBackbone] Loaded pretrained weights from 22k model.")

    def forward(self, x):
        # encoder 返回: [_, x4, x3, x2, x1]
        _, x4, x3, x2, x1 = self.encoder(x)

        # 返回顺序和 BDNet 一致: [x1, x2, x3, x4]
        return x1, x2, x3, x4
