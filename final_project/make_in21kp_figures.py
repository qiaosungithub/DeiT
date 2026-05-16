#!/usr/bin/env python3
"""Generate IN-21K-P training-trajectory and K-ablation figures.

Per-epoch numbers below are the 8-step iterative top-1 / top-5 (MBM heads) or
the single-pass softmax top-1 / top-5 (standard head) printed at every epoch
boundary by mbm/engine.py during training, parsed from SLURM stdout for:
  * Standard CE softmax: vit21k_805255.out (ep 0-2) + vit21k_806581.out (ep 3-28)
  * MBM K=80 + Mixup/CutMix (Run AB): vit21k_822746.out (ep 0-18) + vit21k_828980.out (ep 19-21)
  * MBM K=10 (no Mixup): vit21k_817685.out (ep 0-21)
  * MBM cross-attn (historical, K=10): vit21k_798148.out (ep 0-28)
The 'Step' axis used in the W&B preview corresponds to epoch number here.
"""

from pathlib import Path
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STANDARD = {
    "ep": list(range(29)),
    "top1": [0.3559, 0.3776, 0.3916,
             0.4110, 0.4243, 0.4296, 0.4392, 0.4448, 0.4498, 0.4545, 0.4586,
             0.4615, 0.4659, 0.4714, 0.4714, 0.4741, 0.4756, 0.4784, 0.4806,
             0.4812, 0.4826, 0.4835, 0.4853, 0.4851, 0.4865, 0.4869, 0.4870,
             0.4866, 0.4868],
    "top5": [0.6594, 0.6832, 0.6996,
             0.7187, 0.7301, 0.7377, 0.7462, 0.7515, 0.7555, 0.7598, 0.7640,
             0.7667, 0.7699, 0.7737, 0.7754, 0.7781, 0.7795, 0.7817, 0.7830,
             0.7842, 0.7854, 0.7869, 0.7876, 0.7886, 0.7891, 0.7890, 0.7894,
             0.7900, 0.7901],
}

MBM_RUNAB = {
    "ep": list(range(22)),
    "top1": [0.1684, 0.2320, 0.2551, 0.2834, 0.3020, 0.3147, 0.3248, 0.3325,
             0.3403, 0.3472, 0.3500, 0.3582, 0.3621, 0.3679, 0.3735, 0.3801,
             0.3843, 0.3909, 0.3959, 0.4016, 0.4065, 0.4095],
    "top5": [0.2982, 0.3536, 0.3738, 0.3984, 0.4173, 0.4269, 0.4374, 0.4463,
             0.4514, 0.4594, 0.4609, 0.4680, 0.4707, 0.4763, 0.4821, 0.4884,
             0.4927, 0.4988, 0.5032, 0.5077, 0.5116, 0.5145],
}

MBM_K10 = {
    "ep": list(range(22)),
    "top1": [0.1866, 0.2341, 0.2566, 0.2870, 0.3046, 0.3187, 0.3259, 0.3377,
             0.3449, 0.3521, 0.3571, 0.3624, 0.3690, 0.3730, 0.3799, 0.3861,
             0.3901, 0.3971, 0.4023, 0.4079, 0.4143, 0.4180],
    "top5": [0.3179, 0.3509, 0.3698, 0.3987, 0.4164, 0.4296, 0.4354, 0.4452,
             0.4543, 0.4615, 0.4658, 0.4703, 0.4765, 0.4798, 0.4865, 0.4911,
             0.4951, 0.5013, 0.5065, 0.5127, 0.5172, 0.5203],
}

MBM_CROSSATTN = {
    "ep": list(range(29)),
    "top1": [0.0792, 0.1679, 0.1980, 0.2256, 0.2490, 0.2676, 0.2838, 0.2954,
             0.3071, 0.3200, 0.3277, 0.3365, 0.3446, 0.3522, 0.3612, 0.3663,
             0.3747, 0.3805, 0.3883, 0.3921, 0.3979, 0.4031, 0.4063, 0.4099,
             0.4125, 0.4154, 0.4177, 0.4182, 0.4189],
    "top5": [0.1760, 0.2747, 0.2981, 0.3238, 0.3443, 0.3594, 0.3758, 0.3860,
             0.3970, 0.4095, 0.4181, 0.4247, 0.4324, 0.4418, 0.4488, 0.4547,
             0.4620, 0.4679, 0.4745, 0.4769, 0.4825, 0.4879, 0.4913, 0.4955,
             0.4974, 0.5001, 0.5024, 0.5030, 0.5035],
}

# Colors chosen to match the W&B preview supplied by the user.
COLORS = {
    "standard":  "#3CBEAD",
    "runab":     "#2A9DD8",
    "k10":       "#E8A23E",
    "crossattn": "#7B6BC9",
}


def pct(xs):
    return [100.0 * v for v in xs]


def make_training_panels():
    """Single-panel top-1 trajectory across all four runs."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(STANDARD["ep"], pct(STANDARD["top1"]),
            marker="o", linewidth=2.0, color=COLORS["standard"],
            label="Standard CE softmax")
    ax.plot(MBM_RUNAB["ep"], pct(MBM_RUNAB["top1"]),
            marker="o", linewidth=2.0, color=COLORS["runab"],
            label="MBM K=80 + Mixup/CutMix (Run AB)")
    ax.plot(MBM_K10["ep"], pct(MBM_K10["top1"]),
            marker="o", linewidth=2.0, color=COLORS["k10"],
            label="MBM K=10 (no Mixup)")
    ax.plot(MBM_CROSSATTN["ep"], pct(MBM_CROSSATTN["top1"]),
            marker="o", linewidth=2.0, color=COLORS["crossattn"],
            label="MBM cross-attn (historical)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation top-1 (%)")
    ax.grid(True, alpha=0.28)
    ax.set_xlim(-0.5, 29)
    ax.set_ylim(5, 55)
    ax.set_title("ImageNet-21K-P: validation top-1 trajectory")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "in21kp_training_curve.svg")
    plt.close(fig)


def make_k_ablation():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(MBM_K10["ep"], pct(MBM_K10["top1"]),
            marker="o", linewidth=2.1, color=COLORS["k10"],
            label="K = 10, no Mixup/CutMix")
    ax.plot(MBM_RUNAB["ep"], pct(MBM_RUNAB["top1"]),
            marker="o", linewidth=2.1, color=COLORS["runab"],
            label="K = 80 + Mixup/CutMix (Run AB)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("8-step iterative validation top-1 (%)")
    ax.set_title("Effect of mask samples per image forward (ImageNet-21K-P)")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(15, 45)
    ax.set_xlim(-0.5, 22)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "in21kp_mask_ablation.svg")
    plt.close(fig)


if __name__ == "__main__":
    make_training_panels()
    make_k_ablation()
    print(f"Wrote figures to {OUT_DIR}")
