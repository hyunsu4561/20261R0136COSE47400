"""
Stage 2 Training Loop — MAU-Net Corrector (production version)
==============================================================

Assembles every Stage 2 component into one training script:

    frozen U-Net (priority map)
        -> MAU-Net (recolor)         [TRAINED]
        -> CVD simulation (fixed goggle)
        -> PatchGAN discriminator    [TRAINED, adversarial]
        -> priority-weighted loss
        -> backprop updates MAU-Net (and the discriminator)

The Stage 1 U-Net stays FROZEN: priority maps are produced under torch.no_grad()
and its parameters never receive gradients.

This version adds everything needed for real training runs:
  * saves model checkpoints (so trained weights survive after the run)
  * logs loss history to a JSON file and (optionally) plots curves
  * saves sample recolored images periodically (to eyeball progress)
  * resume-from-checkpoint support
  * a clear place to plug in the real U-Net and the real dataset

================ WHAT FATIMA NEEDS TO PLUG IN (two spots) =================
  [PLUG-IN 1] build_unet(): load the trained Stage 1 U-Net. Search "PLUG-IN 1".
  [PLUG-IN 2] get_dataloader(): return the road-traffic DataLoader.
              Search "PLUG-IN 2".
The training loop itself does not need to change.
==========================================================================
"""

import os
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mau_net import MAUNet
from cvd_simulation import CVDSimulation
from discriminator import PatchGANDiscriminator
from priority_loss import PriorityWeightedLoss


# ===========================================================================
# [PLUG-IN 1] Stage 1 U-Net
# ===========================================================================
class DummyUNet(nn.Module):
    """Placeholder so the pipeline runs without the real checkpoint.
    Produces a [B,1,H,W] map in [0,1] from [B,3,H,W] RGB. NOT meaningful."""
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x):
        return torch.sigmoid(self.body(x))


def build_unet(device, checkpoint_path=None):
    """Return the FROZEN Stage 1 U-Net that produces priority maps.

    >>> PLUG-IN 1 - Fatima: replace the DummyUNet branch with the real U-Net. <<<

    Example (using the U-Net class from the Stage 1 notebook):

        from unet_priority_map import UNetPriorityMap   # your U-Net class
        net = UNetPriorityMap(in_channels=3)
        ckpt = torch.load(checkpoint_path, map_location=device)
        # adjust the key below to match how you saved it:
        net.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)

    The net must take [B,3,H,W] RGB and return [B,1,H,W] priority in [0,1].
    """
    if checkpoint_path is None:
        print("[build_unet] WARNING: using DummyUNet placeholder (no real weights).")
        net = DummyUNet()
    else:
        # >>> PLUG-IN 1 goes here <<<
        raise NotImplementedError(
            "Load the real U-Net here (see the docstring above)."
        )

    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad = False          # FROZEN - never trained in Stage 2
    return net


# ===========================================================================
# [PLUG-IN 2] Dataset
# ===========================================================================
def get_dataloader(batch_size=4, image_size=256, fake=True):
    """Return a DataLoader yielding RGB image batches in [-1, 1].

    >>> PLUG-IN 2 - Fatima: replace this with the road-traffic loader from the
        Stage 1 notebook. It must yield (or have as element [0]) an RGB tensor
        of shape [B,3,H,W] scaled to [-1, 1]. <<<

    The fake=True path returns random noise just so the script is runnable
    end-to-end for testing.
    """
    if fake:
        print("[get_dataloader] WARNING: using random fake data (testing only).")
        imgs = torch.rand(16, 3, image_size, image_size) * 2 - 1
        return DataLoader(TensorDataset(imgs), batch_size=batch_size, shuffle=True)
    # >>> PLUG-IN 2 goes here <<<
    raise NotImplementedError("Return the real road-traffic DataLoader here.")


# ===========================================================================
# Utilities: save sample images and checkpoints
# ===========================================================================
def save_samples(rgb, recolored, priority, path):
    """Save a side-by-side grid (original | priority | recolored) for eyeballing."""
    try:
        import torchvision
        ori = (rgb + 1) / 2
        rec = (recolored + 1) / 2
        pri = priority.repeat(1, 3, 1, 1)
        n = min(4, rgb.shape[0])
        grid = torch.cat([ori[:n], pri[:n], rec[:n]], dim=0).clamp(0, 1)
        torchvision.utils.save_image(grid, path, nrow=n)
    except Exception as e:
        print(f"[save_samples] skipped ({e})")


def save_checkpoint(path, mau, disc, opt_g, opt_d, epoch, step):
    torch.save({
        "mau": mau.state_dict(),
        "disc": disc.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "epoch": epoch,
        "step": step,
    }, path)
    print(f"[checkpoint] saved -> {path}")


# ===========================================================================
# Training
# ===========================================================================
def train_stage2(
    train_loader,
    device="cpu",
    epochs=1,
    lr=1e-4,
    lambda_distinct=1.0,
    lambda_natural=1.0,
    lambda_adv=0.1,
    cvd_severity=1.0,
    unet_checkpoint=None,
    out_dir="stage2_runs",
    log_every=10,
    sample_every=200,
    save_every_epochs=1,
    resume=None,
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    unet = build_unet(device, unet_checkpoint)
    mau = MAUNet(in_ch=4, out_ch=3, use_embedding=True).to(device)
    disc = PatchGANDiscriminator(in_ch=3).to(device)
    cvd = CVDSimulation(severity=cvd_severity, in01=False).to(device)
    gen_loss_fn = PriorityWeightedLoss(
        cvd, lambda_distinct=lambda_distinct,
        lambda_natural=lambda_natural, lambda_adv=lambda_adv, in01=False
    ).to(device)
    bce = nn.BCEWithLogitsLoss()

    opt_g = torch.optim.Adam(mau.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    start_epoch, step = 0, 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        mau.load_state_dict(ckpt["mau"]); disc.load_state_dict(ckpt["disc"])
        opt_g.load_state_dict(ckpt["opt_g"]); opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch, step = ckpt["epoch"], ckpt["step"]
        print(f"[resume] from {resume} (epoch {start_epoch}, step {step})")

    history = []
    for epoch in range(start_epoch, epochs):
        for rgb in train_loader:
            if isinstance(rgb, (list, tuple)):
                rgb = rgb[0]
            rgb = rgb.to(device)

            with torch.no_grad():
                priority = unet(rgb)
            conditioned = torch.cat([rgb, priority], dim=1)

            recolored = mau(conditioned)

            opt_d.zero_grad()
            d_real = disc(rgb)
            d_fake = disc(recolored.detach())
            loss_d = 0.5 * (bce(d_real, torch.ones_like(d_real))
                            + bce(d_fake, torch.zeros_like(d_fake)))
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad()
            d_fake_for_g = disc(recolored)
            adv_loss = bce(d_fake_for_g, torch.ones_like(d_fake_for_g))
            total_g, parts = gen_loss_fn(recolored, rgb, priority, adv_loss=adv_loss)
            total_g.backward()
            opt_g.step()

            if step % log_every == 0:
                parts["loss_d"] = float(loss_d.detach())
                parts["epoch"] = epoch
                parts["step"] = step
                history.append(parts)
                print(f"epoch {epoch} step {step} | "
                      f"G {parts['total']:.4f} (distinct {parts['distinct']:.4f}, "
                      f"natural {parts['natural']:.4f}, adv {parts['adv']:.4f}) | "
                      f"D {parts['loss_d']:.4f}")

            if sample_every and step % sample_every == 0:
                save_samples(rgb, recolored.detach(), priority,
                             os.path.join(out_dir, "samples", f"step_{step:06d}.png"))
            step += 1

        with open(os.path.join(out_dir, "loss_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        if (epoch + 1) % save_every_epochs == 0:
            save_checkpoint(
                os.path.join(out_dir, "checkpoints", f"mau_epoch{epoch:03d}.pth"),
                mau, disc, opt_g, opt_d, epoch + 1, step)

    save_checkpoint(os.path.join(out_dir, "checkpoints", "mau_final.pth"),
                    mau, disc, opt_g, opt_d, epochs, step)
    plot_curves(history, out_dir)
    return mau, disc, history


def plot_curves(history, out_dir):
    """Plot loss curves to a PNG. Skips if matplotlib unavailable."""
    if not history:
        return
    try:
        import matplotlib.pyplot as plt
        steps = [h["step"] for h in history]
        plt.figure(figsize=(9, 4))
        for key in ["total", "distinct", "natural", "adv", "loss_d"]:
            if key in history[0]:
                plt.plot(steps, [h[key] for h in history], label=key)
        plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.title("Stage 2 training")
        plt.tight_layout()
        p = os.path.join(out_dir, "loss_curves.png")
        plt.savefig(p, dpi=110); plt.close()
        print(f"[plot] saved -> {p}")
    except Exception as e:
        print(f"[plot] skipped ({e})")


def parse_args():
    ap = argparse.ArgumentParser(description="Stage 2 MAU-Net training")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_distinct", type=float, default=1.0)
    ap.add_argument("--lambda_natural", type=float, default=1.0)
    ap.add_argument("--lambda_adv", type=float, default=0.1)
    ap.add_argument("--cvd_severity", type=float, default=1.0)
    ap.add_argument("--unet_checkpoint", type=str, default=None,
                    help="path to trained Stage 1 U-Net (PLUG-IN 1)")
    ap.add_argument("--out_dir", type=str, default="stage2_runs")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--real_data", action="store_true",
                    help="use the real dataloader (PLUG-IN 2) instead of fake data")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    loader = get_dataloader(batch_size=args.batch_size, fake=not args.real_data)

    train_stage2(
        loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        lambda_distinct=args.lambda_distinct,
        lambda_natural=args.lambda_natural,
        lambda_adv=args.lambda_adv,
        cvd_severity=args.cvd_severity,
        unet_checkpoint=args.unet_checkpoint,
        out_dir=args.out_dir,
        resume=args.resume,
    )
    print("done.")
