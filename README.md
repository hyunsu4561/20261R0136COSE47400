[README.md](https://github.com/user-attachments/files/28267936/README.md)
# Semantic-aware Recoloring for Color-Blindness Users

**Course:** COSE474 — Deep Learning (2026 Spring)  
**Team:** Group 14  
**Members:** 인탄 누르나지하 (2023320129) · 파티마 (2024320175) · 신현수

---

## Project Overview

Color Vision Deficiency (CVD) affects approximately 350 million people worldwide. Existing recoloring tools apply the same color transformation to every pixel, making safety-critical objects like traffic lights indistinguishable from other red objects (billboards, brake lights).

This project proposes a **two-stage semantic-aware recoloring pipeline** that:
1. Uses a U-Net to produce a **class-aware priority map** — assigning each pixel a recoloring weight based on its semantic class.
2. Feeds the priority map + original RGB image into a **MAU-Net corrector** that applies stronger recoloring to safety-critical regions (traffic lights = 1.0) and leaves low-priority regions (sky, buildings = 0.1) natural.

---

## Pipeline Architecture

```
Input Image
    │
    ├──────────────────────────────────────┐
    ▼                                      │
[STAGE 1] Priority Map U-Net              │
  (trained from scratch)                  │
    │                                      │
    │  Priority Map (conditioning)         │
    ▼                                      ▼
[STAGE 2] MAU-Net Corrector  ◄── [R, G, B, Priority]
    │
    ▼
Recolored Image ──► CVD Simulation (fixed) ──► L1 / Perceptual Loss
                                                      │
                                              Backprop (MAU-Net only)
```

### Priority Class Weights

| Class | Priority Value |
|---|---|
| Background / Sky / Buildings | 0.1 |
| Vegetation | 0.1 |
| Person / Cyclist | 0.4 |
| Vehicle | 0.7 |
| Traffic Light / Sign | 1.0 |

---

## Repository Structure

```
20261R0136COSE47400/
├── README.md
├── stage1_priority_map_unet/
│   ├── README.md
│   └── Traffic_Light_UNet_PriorityMap_ClassAware.ipynb
└── stage2_maunet_corrector/          ← (planned, Finals)
    └── ...
```

---

## Stage 1 — Priority Map U-Net (Complete)

See [`stage1_priority_map_unet/`](./stage1_priority_map_unet/)

**U-Net Architecture:**
- Input: RGB image `[3, 256, 256]`
- Encoder: 4× (Conv3×3 → BN → ReLU ×2, MaxPool) — channels 64→128→256→512
- Bottleneck: 1024 channels
- Decoder: 4× (TransposeConv + skip concat + Conv3×3 → BN → ReLU ×2)
- Output: `[1, 256, 256]` sigmoid priority map

**Training Setup:**
- Dataset: Road-Traffic Dataset (HuggingFace)
- Ground truth: generated on-the-fly via DeepLabV3 + HSV hybrid
- Image size: 256×256, Batch: 4, Epochs: 25, LR: 1e-4
- Optimizer: Adam with weight decay; Scheduler: ReduceLROnPlateau
- Loss: MSE + weighted BCE

**Stage 1 Results (Best ~Epoch 25):**

| Metric | Epoch 1 | Best |
|---|---|---|
| Train Loss | ~0.85 | ~0.02 |
| Val Loss | ~0.28 | ~0.02 |
| Val MAE | ~0.055 | ~0.020 |
| Traffic Light Hue IoU | ~0.15 | ~0.50 |

---

## Stage 2 — MAU-Net Corrector (Planned — Finals)

- Input: `[R, G, B, Priority]` 4-channel tensor
- Architecture: Modified Multi-Attention U-Net (MAU-Net) from Nathanael & Prasetyo (2024)
- Loss: L1 + Perceptual loss after CVD simulation
- Only MAU-Net trains; U-Net stays frozen

---

## References

1. Wang et al., "A Review of Gray-scale Image Recoloring Methods With Neural Network Based Model," ICIVC 2022.
2. Petrovic & Fujita, "Deep Correct: Deep Learning Color Correction for Color Blindness," 2017.
3. Nathanael & Prasetyo, "Color and Attention for U: Modified Multi Attention U-Net for a Better Image Colorization," JOIV 2024.
