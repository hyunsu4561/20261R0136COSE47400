"""
MAU-Net Generator for Semantic-aware CVD Recoloring
====================================================

Implementation of the Modified Multi-Attention U-Net (MAU-Net) generator from:

    Nathanael, O. T., & Prasetyo, S. Y. (2024).
    "Color and Attention for U: Modified Multi Attention U-Net for a
    Better Image Colorization." JOIV, 8(3), 1453-1459.

ADAPTED for the project task (Semantic-aware Recoloring for Colorblindness):
the original paper does grayscale->color colorization in Lab space
(3ch L-ish input -> 2ch ab output). This version is adapted to the project's
Stage-2 spec:

    Input : [B, 4, H, W]  = RGB (3 channels) + priority map (1 channel)
    Output: [B, 3, H, W]  = recolored RGB in [-1, 1] (tanh)

Components that follow the paper directly:
  * U-Net backbone: (Conv3x3 + BN + LeakyReLU) x2 blocks, DownConv 4x4,
    TransposeConv 4x4 (paper Fig. 3/5 legend).
  * MobileNetV3 embeddings injected at the bottleneck. The paper extracts
    two embeddings (avg-pool: 576-dim, classifier output: 1000-dim),
    concatenates them to the deepest latent, then a 1x1 conv with 512
    filters reduces channels (paper section D).
  * CAM (Channel Attention Module), used in the low-resolution decoder
    stages (paper Fig. 6 / section E).
  * SAM (Spatial Attention Module), used near the end / high-resolution
    decoder stages (paper Fig. 7 / section E).

Notes / assumptions where the paper is under-specified (flagged in comments):
  * Exact decoder levels for CAM vs SAM: paper says CAM at low-res,
    SAM "near the end". We place CAM on the two deepest decoder stages and
    SAM on the two shallowest. Easy to change via the `attn_per_stage` list.
  * CAM linear reduction ratio: not stated; we use 16 (CBAM convention).
  * Embedding broadcast: MobileNetV3 1D embeddings are expanded spatially
    to the bottleneck H'xW' before concatenation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ---------------------------------------------------------------------------
# Basic building blocks (paper Fig. 3 / Fig. 5 legend)
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> LeakyReLU) x2 -- the red arrow in the paper figures."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownConv(nn.Module):
    """DownConv 4x4 + BN + LeakyReLU -- the orange arrow (stride-2 downsample)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpConv(nn.Module):
    """TransposeConv 4x4 + BN + LeakyReLU -- the green arrow (stride-2 upsample)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# Channel Attention Module (paper Fig. 6, section E)
#   Conv3x3 -> {MaxPool, AvgPool over spatial} -> shared Linear -> sum
#   -> Sigmoid -> multiply back. Used at low-res, high-channel stages.
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Paper: "applying a convolution layer with a 3x3 kernel" first.
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Shared MLP applied to both pooled vectors ("transformed linearly").
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        b, c, _, _ = feat.shape
        max_vec = self.mlp(self.max_pool(feat).view(b, c))
        avg_vec = self.mlp(self.avg_pool(feat).view(b, c))
        attn = self.sigmoid(max_vec + avg_vec).view(b, c, 1, 1)
        return feat * attn


# ---------------------------------------------------------------------------
# Spatial Attention Module (paper Fig. 7, section E)
#   Conv3x3 -> {MaxPool, AvgPool over channel} -> concat (2ch)
#   -> Conv4x4 -> Sigmoid -> multiply back. Used near the end (high-res).
# ---------------------------------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        # 4x4 conv with padding=1 + 1px pad on one side keeps H,W via F.pad below;
        # we use padding to preserve spatial size for a 4x4 kernel.
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=4, padding=2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        max_map = torch.max(feat, dim=1, keepdim=True)[0]   # [B,1,H,W]
        avg_map = torch.mean(feat, dim=1, keepdim=True)      # [B,1,H,W]
        concat = torch.cat([max_map, avg_map], dim=1)        # [B,2,H,W]
        attn = self.spatial_conv(concat)
        # padding=2 on a 4x4 kernel grows size by 1; crop back to match.
        attn = attn[:, :, : feat.shape[2], : feat.shape[3]]
        attn = self.sigmoid(attn)                            # [B,1,H,W]
        return feat * attn


# ---------------------------------------------------------------------------
# MobileNetV3 embedding extractor (paper section D)
#   Two embeddings: avg-pool (576-dim) and classifier output (1000-dim).
# ---------------------------------------------------------------------------
class MobileNetV3Embedding(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        weights = (
            torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        net = torchvision.models.mobilenet_v3_large(weights=weights)
        self.features = net.features          # conv backbone
        self.avgpool = net.avgpool            # -> 960-dim spatially pooled
        self.classifier = net.classifier      # -> 1000-dim logits
        # Paper states 576 + 1000. mobilenet_v3_large's pooled feature is 960,
        # so the "576" in the paper corresponds to mobilenet_v3_small.
        # We expose the actual pooled dim so the bottleneck adapts automatically.
        self.pooled_dim = 960
        self.logit_dim = 1000
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def forward(self, rgb):
        # rgb: [B,3,H,W] in roughly ImageNet range. The paper feeds the (gray)
        # image stacked to 3 channels; here we feed the RGB part directly.
        x = self.features(rgb)
        pooled = self.avgpool(x)                 # [B,960,1,1]
        pooled_vec = torch.flatten(pooled, 1)     # [B,960]
        logits = self.classifier(pooled_vec)      # [B,1000]
        return pooled_vec, logits


# ---------------------------------------------------------------------------
# MAU-Net Generator
# ---------------------------------------------------------------------------
class MAUNet(nn.Module):
    """
    Modified Multi-Attention U-Net generator adapted for CVD recoloring.

    Args:
        in_ch:  input channels (4 = RGB + priority map for the project)
        out_ch: output channels (3 = recolored RGB)
        base:   base channel width (64 -> 128 -> 256 -> 512, bottleneck 1024)
        use_embedding: inject MobileNetV3 bottleneck embeddings (paper section D)
        attn_per_stage: which attention to use per decoder stage, deepest first.
                        'cam' = Channel Attention, 'sam' = Spatial Attention,
                        None  = plain upsample (paper keeps some stages plain).
    """

    def __init__(
        self,
        in_ch=4,
        out_ch=3,
        base=64,
        use_embedding=True,
        attn_per_stage=("cam", "cam", "sam", "sam"),
    ):
        super().__init__()
        self.use_embedding = use_embedding
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8       # 64,128,256,512
        c_bottle = base * 16                                       # 1024

        # ---- Encoder (4 down stages) ----
        self.enc1 = DoubleConv(in_ch, c1)
        self.down1 = DownConv(c1, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.down2 = DownConv(c2, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.down3 = DownConv(c3, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.down4 = DownConv(c4, c4)

        # ---- Bottleneck ----
        self.bottleneck = DoubleConv(c4, c_bottle)                 # 1024

        # MobileNetV3 embedding injection (paper section D):
        # concat [bottleneck, broadcast(pooled), broadcast(logits)] then 1x1 conv.
        if use_embedding:
            self.embedder = MobileNetV3Embedding(pretrained=True, freeze=True)
            emb_dim = self.embedder.pooled_dim + self.embedder.logit_dim
            self.embed_reduce = nn.Sequential(
                nn.Conv2d(c_bottle + emb_dim, base * 8, kernel_size=1, bias=False),
                nn.BatchNorm2d(base * 8),
                nn.LeakyReLU(0.2, inplace=True),
            )
            bottle_out = base * 8   # 512  (paper: "1x1 conv with 512 filters")
        else:
            bottle_out = c_bottle

        # ---- Decoder (4 up stages) with skip connections ----
        # Stage layout (deepest -> shallowest): channels mirror the encoder.
        self.up4 = UpConv(bottle_out, c4)                          # ->512, H/8
        self.dec4 = DoubleConv(c4 + c4, c4)
        self.attn4 = self._make_attn(attn_per_stage[0], c4)

        self.up3 = UpConv(c4, c3)                                  # ->256, H/4
        self.dec3 = DoubleConv(c3 + c3, c3)
        self.attn3 = self._make_attn(attn_per_stage[1], c3)

        self.up2 = UpConv(c3, c2)                                  # ->128, H/2
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.attn2 = self._make_attn(attn_per_stage[2], c2)

        self.up1 = UpConv(c2, c1)                                  # ->64,  H
        self.dec1 = DoubleConv(c1 + c1, c1)
        self.attn1 = self._make_attn(attn_per_stage[3], c1)

        # ---- Output head (project-specific: RGB recoloring) ----
        self.out_conv = nn.Conv2d(c1, out_ch, kernel_size=1)
        self.out_act = nn.Tanh()   # recolored RGB in [-1,1]

    @staticmethod
    def _make_attn(kind, channels):
        if kind == "cam":
            return ChannelAttention(channels)
        if kind == "sam":
            return SpatialAttention(channels)
        return nn.Identity()

    def forward(self, x):
        # x: [B, in_ch, H, W]; first 3 channels are RGB for the embedder.
        rgb = x[:, :3, :, :]

        # Encoder
        e1 = self.enc1(x);  p1 = self.down1(e1)
        e2 = self.enc2(p1); p2 = self.down2(e2)
        e3 = self.enc3(p2); p3 = self.down3(e3)
        e4 = self.enc4(p3); p4 = self.down4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        if self.use_embedding:
            # MobileNetV3 wants a reasonable input size; resize RGB to 224.
            rgb_resized = F.interpolate(
                rgb, size=(224, 224), mode="bilinear", align_corners=False
            )
            pooled, logits = self.embedder(rgb_resized)            # [B,960],[B,1000]
            emb = torch.cat([pooled, logits], dim=1)               # [B,1960]
            emb = emb[:, :, None, None].expand(-1, -1, b.shape[2], b.shape[3])
            b = torch.cat([b, emb], dim=1)
            b = self.embed_reduce(b)                               # [B,512,H/16,W/16]

        # Decoder with skips + attention
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d4 = self.attn4(d4)

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d3 = self.attn3(d3)

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d2 = self.attn2(d2)

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d1 = self.attn1(d1)

        out = self.out_act(self.out_conv(d1))
        return out


if __name__ == "__main__":
    # Quick shape / sanity check.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MAUNet(in_ch=4, out_ch=3, use_embedding=True).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {n_params:,}")
    print(f"Trainable params: {n_train:,}  (MobileNetV3 frozen)")

    x = torch.randn(2, 4, 256, 256, device=device)  # B=2, RGB+priority, 256x256
    y = model(x)
    print("Input :", tuple(x.shape))
    print("Output:", tuple(y.shape))
    print("Output range:", float(y.min()), "to", float(y.max()))
    assert y.shape == (2, 3, 256, 256), "Output shape mismatch!"
    print("\nForward pass OK.")
