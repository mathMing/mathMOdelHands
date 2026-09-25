"""重复随机起点缺失实验：报告均值、标准差和实际条件。"""
import os
import numpy as np
import pandas as pd
import torch

from dataset import get_dataloaders
from model import RobustMultimodalModel
from ablation_experiments import evaluate_with_custom_mask

ROOT = "/root/E_Problem" if os.path.exists("/root/E_Problem") else r"X:\\mathModelHands"
PKL = os.path.join(ROOT, "DATA", "Att2-Extracted-Features", "aligned_50.pkl")
MODEL = os.path.join(ROOT, "problem2_robust", "best_robust_model.pt")
OUT = os.path.join(ROOT, "problem2_robust", "repeated_missing_summary.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    ckpt = torch.load(MODEL, map_location=DEVICE, weights_only=False)
    models = []
    for sd in ckpt.get("ensemble_state_dicts", [ckpt["model_state_dict"]]):
        model = RobustMultimodalModel(d_model=ckpt.get("d_model", 256), num_heads=ckpt.get("num_heads", 4)).to(DEVICE)
        model.load_state_dict(sd)
        model.eval()
        models.append(model)
    _, valid_loader, _, _ = get_dataloaders(PKL, batch_size=64)
    rows = []
    for miss_type in ["a", "v", "t", "a_v"]:
        for ratio in [0.3, 0.5, 0.7]:
            miss_len = int(50 * ratio)
            starts = sorted(set([0, max(0, (50 - miss_len)//2), 50 - miss_len]))
            vals = []
            for start in starts:
                vals.append(evaluate_with_custom_mask(models, valid_loader, miss_type=miss_type, ratio=ratio, start_idx_override=start))
            arr = np.asarray(vals, dtype=float)
            rows.append({
                "missing_type": miss_type,
                "nominal_ratio": ratio,
                "starts": ",".join(map(str, starts)),
                "accuracy_mean": arr[:, 0].mean(), "accuracy_std": arr[:, 0].std(ddof=1),
                "macro_f1_mean": arr[:, 1].mean(), "macro_f1_std": arr[:, 1].std(ddof=1),
                "mae_mean": arr[:, 2].mean(), "mae_std": arr[:, 2].std(ddof=1),
                "pearson_mean": arr[:, 3].mean(), "pearson_std": arr[:, 3].std(ddof=1),
            })
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"重复缺失实验完成: {OUT}")

if __name__ == "__main__":
    main()
