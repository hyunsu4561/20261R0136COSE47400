"""
CVD Simulation — Differentiable Deuteranomaly (Machado et al. 2009)
===================================================================

Simulates how an image looks to a person with deuteranomaly (red-green
color vision deficiency, the most common type). Used in the Stage 2 training
loop: the MAU-Net's recolored output is passed through this "colorblind
goggle" before the loss is computed, so the model learns to make
safety-critical regions distinguishable *as seen by a CVD viewer*.

Method
------
Machado, Oliveira & Fernandes (2009), "A Physiologically-based Model for
Simulation of Color Vision Deficiency", IEEE TVCG 15(6):1291-1298.

The simulation is a single 3x3 matrix multiplication on linear-RGB pixels
(paper Eq. 1), which makes it fully differentiable — gradients flow back
through it to the MAU-Net. Pre-computed deuteranomaly matrices are provided
for severities 0.0 (no CVD) to 1.0 (dichromacy / deuteranopia) in steps of
0.1, taken verbatim from the paper's Table 1. Intermediate severities are
linearly interpolated between the two nearest matrices, as recommended.

Severity guide:
    0.0  = normal color vision (identity matrix)
    0.5  = moderate deuteranomaly
    1.0  = deuteranopia (full dichromacy, most severe) -- default

Note on color range: these matrices apply to *linear* RGB in [0, 1]. This
module accepts images in [0, 1] (default) or [-1, 1] (set in01=False to
match MAU-Net's tanh output) and handles the conversion internally.
"""

import torch
import torch.nn as nn


# Deuteranomaly simulation matrices from Machado et al. (2009), Table 1.
# Index = severity * 10 (so DEUTERANOMALY[10] is severity 1.0). Rows are the
# matrix rows applied as  [R_s, G_s, B_s]^T = M @ [R, G, B]^T.
_DEUTERANOMALY = {
    0.0: [[1.000000,  0.000000, -0.000000],
          [0.000000,  1.000000,  0.000000],
          [-0.000000, -0.000000, 1.000000]],
    0.1: [[0.866435,  0.177704, -0.044139],
          [0.049567,  0.939063,  0.011370],
          [-0.003453, 0.007233,  0.996220]],
    0.2: [[0.760729,  0.319078, -0.079807],
          [0.090568,  0.889315,  0.020117],
          [-0.006027, 0.013325,  0.992702]],
    0.3: [[0.675425,  0.433850, -0.109275],
          [0.125303,  0.847755,  0.026942],
          [-0.007950, 0.018572,  0.989378]],
    0.4: [[0.605511,  0.528560, -0.134071],
          [0.155318,  0.812366,  0.032316],
          [-0.009376, 0.023176,  0.986200]],
    0.5: [[0.547494,  0.607765, -0.155259],
          [0.181692,  0.781742,  0.036566],
          [-0.010410, 0.027275,  0.983136]],
    0.6: [[0.498864,  0.674741, -0.173604],
          [0.205199,  0.754872,  0.039929],
          [-0.011131, 0.030969,  0.980162]],
    0.7: [[0.457771,  0.731899, -0.189670],
          [0.226409,  0.731012,  0.042579],
          [-0.011595, 0.034333,  0.977261]],
    0.8: [[0.422823,  0.781057, -0.203881],
          [0.245752,  0.709602,  0.044646],
          [-0.011843, 0.037423,  0.974421]],
    0.9: [[0.392952,  0.823610, -0.216562],
          [0.263559,  0.690210,  0.046232],
          [-0.011910, 0.040281,  0.971630]],
    1.0: [[0.367322,  0.860646, -0.227968],
          [0.280085,  0.672501,  0.047413],
          [-0.011820, 0.042940,  0.968881]],
}


def deuteranomaly_matrix(severity: float) -> torch.Tensor:
    """Return the 3x3 deuteranomaly matrix for an arbitrary severity in [0, 1].

    Uses the exact Machado matrix when severity lands on a 0.1 step, and
    linearly interpolates between the two nearest steps otherwise.
    """
    severity = float(max(0.0, min(1.0, severity)))
    lower = int(severity * 10) / 10.0          # nearest 0.1 step below
    upper = min(lower + 0.1, 1.0)
    lower = round(lower, 1)
    upper = round(upper, 1)

    m_low = torch.tensor(_DEUTERANOMALY[lower], dtype=torch.float32)
    if upper == lower:
        return m_low
    m_high = torch.tensor(_DEUTERANOMALY[upper], dtype=torch.float32)
    w = (severity - lower) / (upper - lower)   # interpolation weight
    return (1.0 - w) * m_low + w * m_high


class CVDSimulation(nn.Module):
    """Differentiable deuteranomaly simulation as an nn.Module.

    Args:
        severity: 0.0 (normal) to 1.0 (deuteranopia). Default 1.0.
        in01:     True  -> input/output images are in [0, 1] (default).
                  False -> input/output images are in [-1, 1] (MAU-Net tanh).
        linear_rgb: True (default) -> convert sRGB->linear before the matrix
                  multiply and back to sRGB after, as the Machado et al. (2009)
                  model intends (the matrices act on LINEAR RGB). False ->
                  apply the matrix directly to gamma-encoded sRGB (the previous
                  behavior; slightly less physically accurate but cheaper).

    Input  : image tensor [B, 3, H, W]
    Output : CVD-simulated image tensor, same shape and same range as input.

    The matrix is registered as a buffer (not a parameter) so it moves with
    .to(device) but never receives gradients — this is a fixed transform.
    """

    def __init__(self, severity: float = 1.0, in01: bool = True,
                 linear_rgb: bool = True):
        super().__init__()
        self.severity = severity
        self.in01 = in01
        self.linear_rgb = linear_rgb
        self.register_buffer("matrix", deuteranomaly_matrix(severity))

    @staticmethod
    def _srgb_to_linear(c):
        # Standard sRGB EOTF (gamma decode). Differentiable.
        return torch.where(c <= 0.04045, c / 12.92,
                           ((c + 0.055) / 1.055).clamp(min=0.0) ** 2.4)

    @staticmethod
    def _linear_to_srgb(c):
        c = c.clamp(0.0, 1.0)
        return torch.where(c <= 0.0031308, c * 12.92,
                           1.055 * (c ** (1.0 / 2.4)) - 0.055)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() != 4 or img.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W], got {tuple(img.shape)}")

        # Bring image to [0,1] sRGB for the matrix multiply if needed.
        x = (img + 1.0) * 0.5 if not self.in01 else img
        x = x.clamp(0.0, 1.0)

        # The Machado matrices are defined on LINEAR RGB. Decode gamma first.
        if self.linear_rgb:
            x = self._srgb_to_linear(x)

        b, c, h, w = x.shape
        flat = x.reshape(b, c, h * w)                  # [B, 3, H*W]
        # matrix [3,3] applied per-pixel: out_c = sum_c' M[c,c'] * in_c'
        simulated = torch.einsum("ij,bjn->bin", self.matrix, flat)
        simulated = simulated.reshape(b, c, h, w).clamp(0.0, 1.0)

        # Re-encode gamma to get back to sRGB [0,1].
        if self.linear_rgb:
            simulated = self._linear_to_srgb(simulated)

        # Convert back to the input's range.
        if not self.in01:
            simulated = simulated * 2.0 - 1.0
        return simulated


if __name__ == "__main__":
    # Sanity checks.
    print("== severity 0.0 should be identity ==")
    sim0 = CVDSimulation(severity=0.0, in01=True)
    img = torch.rand(2, 3, 64, 64)
    out0 = sim0(img)
    print("max diff from input:", float((out0 - img).abs().max()))  # ~0

    print("\n== severity 1.0 (deuteranopia) changes the image ==")
    sim1 = CVDSimulation(severity=1.0, in01=True)
    out1 = sim1(img)
    print("mean abs change:", float((out1 - img).abs().mean()))

    print("\n== differentiable: gradient flows back to input ==")
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    y = CVDSimulation(severity=1.0)(x)
    y.mean().backward()
    print("input has grad:", x.grad is not None,
          "| grad nonzero:", float(x.grad.abs().sum()) > 0)

    print("\n== works on tanh-range [-1,1] input (MAU-Net output) ==")
    sim_t = CVDSimulation(severity=1.0, in01=False)
    xt = torch.rand(1, 3, 32, 32) * 2 - 1
    yt = sim_t(xt)
    print("output range: [%.3f, %.3f]" % (float(yt.min()), float(yt.max())))

    print("\n== interpolation: severity 0.55 is between 0.5 and 0.6 ==")
    m5, m55, m6 = (deuteranomaly_matrix(s) for s in (0.5, 0.55, 0.6))
    print("0.55 between 0.5 and 0.6:",
          bool(((m55 >= torch.min(m5, m6)) & (m55 <= torch.max(m5, m6))).all()))

    print("\nAll CVD simulation checks passed.")
