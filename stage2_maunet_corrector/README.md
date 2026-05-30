# Stage 2 — MAU-Net Corrector

Second stage of the **Semantic-aware Recoloring for Color-Blindness Users** pipeline (Group 14, COSE474).

This stage takes the class-aware priority map produced by the [Stage 1 U-Net](../stage1_priority_map_unet/) and learns to recolor the image — applying **stronger correction to safety-critical regions** (traffic lights, signs) while leaving low-priority regions (sky, buildings) natural.

---

## Role in the Pipeline

```
[STAGE 1] Priority Map U-Net  ──► Priority Map [1, 256, 256]
                                        │
              Original RGB [3,256,256]  │
                          └──── concat ─┤
                                        ▼
                          [R, G, B, Priority]  =  [4, 256, 256]
                                        │
                                        ▼
[STAGE 2] MAU-Net Corrector  ──►  Recolored RGB [3, 256, 256]
                                        │
                                        ▼
                          CVD Simulation (fixed)
                                        │
                                        ▼
                          L1 / Perceptual Loss ──► Backprop (MAU-Net only)
```

Stage 1 stays **frozen**; only the MAU-Net trains in Stage 2.

---

## Architecture (`mau_net.py`)

Modified Multi-Attention U-Net, adapted from Nathanael & Prasetyo (2024) for the recoloring task.

| Component | Detail |
|---|---|
| **Input** | `[B, 4, 256, 256]` — RGB + priority map |
| **Output** | `[B, 3, 256, 256]` — recolored RGB, `tanh` ∈ [-1, 1] |
| **Backbone** | U-Net encoder/decoder, channels 64→128→256→512, bottleneck 1024 |
| **Bottleneck embedding** | Frozen MobileNetV3 features injected at the bottleneck (paper §D) |
| **Channel Attention (CAM)** | Used in the deep, low-resolution decoder stages (paper Fig. 6) |
| **Spatial Attention (SAM)** | Used in the shallow, high-resolution decoder stages (paper Fig. 7) |

The input stem (4-channel) and output head (3-channel RGB) differ from the original paper, which did Lab-space colorization. Everything else follows the paper.

---

## Usage

```python
from mau_net import MAUNet

model = MAUNet(in_ch=4, out_ch=3, use_embedding=True)

# priority_map: [B, 1, H, W] from the frozen Stage 1 U-Net (in [0, 1])
conditioned_input = torch.cat([rgb, priority_map], dim=1)   # [B, 4, H, W]
recolored = model(conditioned_input)                         # [B, 3, H, W] in [-1, 1]
```

Verified end-to-end compatible with the Stage 1 U-Net output (`[B, 1, 256, 256]` sigmoid).

---

## Status

- [x] MAU-Net generator implemented and shape-verified
- [x] Interface compatibility with Stage 1 U-Net confirmed
- [ ] PatchGAN discriminator
- [ ] Priority-weighted L1 + Perceptual + Adversarial loss
- [ ] CVD simulation module (fixed, in the loss path)
- [ ] Stage 2 training loop (U-Net frozen)
- [ ] Evaluation vs. Deep Correct baseline

---

## Reference

Nathanael, O. T., & Prasetyo, S. Y. (2024). *Color and Attention for U: Modified Multi Attention U-Net for a Better Image Colorization.* JOIV, 8(3), 1453–1459.
