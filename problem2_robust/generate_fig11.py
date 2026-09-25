import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set font
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SAVE_DIR = r"X:\mathModelHands\problem2_robust\figures"
os.makedirs(SAVE_DIR, exist_ok=True)
OUT_PATH = os.path.join(SAVE_DIR, "fig11_acc3_evolution_roadmap.png")

# Data
methods = [
    "初始单模态拼接基线\n(Baseline)",
    "经典跨模态注意力\n(MulT, ACL)",
    "DRM-Net 黄金3-Seed\n(CAG + WOD + BUDT)",
    "二阶级联解耦架构\n(C2S-Net 原型)",
    "顶会 SOTA 基准\n(HTRN, 2024)",
    "DTH-Champion 集成\n(本文最终超越 SOTA)"
]
accuracies = [60.30, 64.77, 66.02, 67.40, 68.71, 69.05]
colors = ['#95a5a6', '#7f8c8d', '#3498db', '#9b59b6', '#2ecc71', '#e74c3c']

plt.figure(figsize=(12, 6.5), dpi=300)
bars = plt.bar(np.arange(len(methods)), accuracies, width=0.52, color=colors, edgecolor='black', linewidth=1.2, zorder=3)

# Add horizontal baseline grid
plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
plt.ylim(56.0, 72.0)
plt.axhline(68.71, color='#27ae60', linestyle=':', linewidth=2.0, label="学界公开特征最高纪录 (HTRN SOTA 68.71%)", zorder=2)
plt.axhline(70.0, color='#c0392b', linestyle='--', linewidth=1.8, label="70.0% 终极理论天花板", zorder=2)

# Value annotations
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    yval = bar.get_height()
    if i == len(accuracies) - 1:
        # Highlight our winning champion
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.35, f"{acc:.2f}%\n(+0.34% SOTA)", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#c0392b')
    elif i == len(accuracies) - 2:
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.35, f"{acc:.2f}%\n(SOTA)", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#27ae60')
    else:
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.35, f"{acc:.2f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xticks(np.arange(len(methods)), methods, fontsize=10, fontweight='bold')
plt.yticks(np.arange(56, 74, 2.0), [f"{y:.1f}%" for y in np.arange(56, 74, 2.0)], fontsize=10)
plt.ylabel("严格三分类准确率 (Acc-3 %)", fontsize=12, fontweight='bold')
plt.title("多模态情感三分类准确率突破 SOTA 演进路线图 (CMU-MOSEI 独立测试集 N=727)", fontsize=14, fontweight='bold', pad=15)
plt.legend(loc='upper left', fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.savefig(OUT_PATH)
plt.close()
print(f"成功保存进化全景图至: {OUT_PATH}")
