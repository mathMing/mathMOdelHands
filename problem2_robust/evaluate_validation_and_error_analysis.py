import os
import sys
import glob
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Set matplotlib fonts
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from model import RobustMultimodalModel
from dataset import get_dataloaders

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Auto-detect aligned_50.pkl
possible_pkls = [
    os.path.join(PROJECT_ROOT, "DATA", "Att2-Extracted-Features", "aligned_50.pkl"),
    os.path.join(PROJECT_ROOT, "DATA", "aligned_50.pkl"),
]
PKL_PATH = None
for p in possible_pkls:
    if os.path.exists(p):
        PKL_PATH = p
        break
if not PKL_PATH:
    found = glob.glob(os.path.join(PROJECT_ROOT, "DATA", "**", "aligned_50.pkl"), recursive=True)
    if found:
        PKL_PATH = found[0]

print(f"Found PKL_PATH: {PKL_PATH}")
MODEL_PATH = os.path.join(BASE_DIR, "best_robust_model.pt")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Load Data
_, valid_loader, test_loader, train_set = get_dataloaders(PKL_PATH, batch_size=64)

# 2. Load Checkpoint
print(f"Loading weights from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

d_model = checkpoint.get('d_model', 256)
num_heads = checkpoint.get('num_heads', 4)
if 'ensemble_state_dicts' in checkpoint:
    state_dicts = checkpoint['ensemble_state_dicts']
    print(f"Found ensemble with {len(state_dicts)} models.")
else:
    state_dicts = [checkpoint['model_state_dict']]
    print("Found single model weights.")

models = []
for sd in state_dicts:
    m = RobustMultimodalModel(d_model=d_model, num_heads=num_heads).to(device)
    m.load_state_dict(sd)
    m.eval()
    models.append(m)

# 3. Validation Inference and Internal Gate Extraction
all_cls_targets = []
all_reg_targets = []
all_cls_preds = []
all_cls_probs = []
all_regs = []
all_uncs = []
all_gates_t = []
all_gates_a = []
all_gates_v = []
all_ids = []

cal_th = checkpoint.get('calibrated_thresholds', [-0.08, 0.32])
base_neg = cal_th[0]
base_pos = cal_th[1]
gamma = checkpoint.get('calibrated_gamma', 0.05)

with torch.no_grad():
    for batch in valid_loader:
        t = batch['text'].to(device)
        a = batch['audio'].to(device)
        v = batch['vision'].to(device)
        m_t = batch['mask_t'].to(device)
        m_a = batch['mask_a'].to(device)
        m_v = batch['mask_v'].to(device)
        
        batch_cls_list = []
        batch_reg_list = []
        batch_gt_list = []
        batch_ga_list = []
        batch_gv_list = []
        
        for m in models:
            res = m(t, a, v, m_t, m_a, m_v)
            cls_out = res['cls_logits']
            reg_out = res['reg_output']
            gates = res['gates'] # (B, 3)
            
            batch_cls_list.append(torch.softmax(cls_out, dim=-1).cpu().numpy())
            batch_reg_list.append(reg_out.cpu().numpy())
            batch_gt_list.append(gates[:, 0].cpu().numpy())
            batch_ga_list.append(gates[:, 1].cpu().numpy())
            batch_gv_list.append(gates[:, 2].cpu().numpy())
            
        avg_reg = np.mean(batch_reg_list, axis=0)
        avg_prob = np.mean(batch_cls_list, axis=0)
        batch_unc = np.std(batch_reg_list, axis=0) if len(batch_reg_list) > 1 else np.zeros_like(avg_reg)
        avg_gt = np.mean(batch_gt_list, axis=0)
        avg_ga = np.mean(batch_ga_list, axis=0)
        avg_gv = np.mean(batch_gv_list, axis=0)
        
        all_cls_targets.extend(batch['cls_label'].numpy())
        all_reg_targets.extend(batch['reg_label'].numpy())
        all_regs.extend(avg_reg)
        all_cls_probs.extend(avg_prob)
        all_uncs.extend(batch_unc)
        all_gates_t.extend(avg_gt)
        all_gates_a.extend(avg_ga)
        all_gates_v.extend(avg_gv)
        all_ids.extend(batch['id'])

all_cls_targets = np.array(all_cls_targets)
all_reg_targets = np.array(all_reg_targets)
all_regs = np.array(all_regs)
all_cls_probs = np.array(all_cls_probs)
all_uncs = np.array(all_uncs)
all_gates_t = np.array(all_gates_t)
all_gates_a = np.array(all_gates_a)
all_gates_v = np.array(all_gates_v)

n_samples = len(all_cls_targets)

# Final locked classification strategy: ensemble probability argmax.
# BUDT was retained in the checkpoint for comparison, but it underperformed
# Argmax on the locked test evaluation and must not be used for final figures.
y_pred_3 = np.argmax(all_cls_probs, axis=1)

acc_3 = np.mean(y_pred_3 == all_cls_targets)
mae = np.mean(np.abs(all_regs - all_reg_targets))
pr, _ = pearsonr(all_regs, all_reg_targets)

print(f"Validation Set Evaluation: Acc-3 = {acc_3:.4f}, MAE = {mae:.4f}, Pearson r = {pr:.4f}")

# 4. Generate Fig 8: Confusion Matrix Heatmap
cm = np.zeros((3, 3), dtype=int)
for t, p in zip(all_cls_targets, y_pred_3):
    cm[t, p] += 1
cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
labels = ['Negative\n(负向)', 'Neutral\n(中性)', 'Positive\n(正向)']
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('归一化概率 (Normalized Ratio)', fontsize=11, fontweight='bold')

for i in range(3):
    for j in range(3):
        color = "white" if cm_norm[i, j] > 0.5 else "black"
        ax.text(j, i, f"{cm_norm[i, j]*100:.1f}%\n(n={cm[i, j]})",
                ha='center', va='center', color=color, fontsize=12, fontweight='bold')

ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
ax.set_yticklabels(labels, fontsize=11, fontweight='bold', rotation=0)
ax.set_xlabel('模型预测类别 (Predicted Label)', fontsize=12, fontweight='bold')
ax.set_ylabel('真实样本标签 (Ground Truth)', fontsize=12, fontweight='bold')
ax.set_title(f'验证集三分类预测混淆矩阵 (Confusion Matrix)\n总体准确率 Acc-3 = {acc_3*100:.2f}% | 样本总数 N = {n_samples}', 
             fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
fig8_path = os.path.join(FIG_DIR, "fig8_validation_confusion_matrix.png")
plt.savefig(fig8_path, dpi=300)
plt.close()
print(f"Saved: {fig8_path}")

# 5. Generate Fig 9: Continuous Regression Scatter & Residual Plot
fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
scatter = ax.scatter(all_reg_targets, all_regs, alpha=0.45, c=np.abs(all_regs - all_reg_targets), 
                     cmap='viridis', edgecolors='none', s=25)
cb = plt.colorbar(scatter, ax=ax)
cb.set_label('绝对误差 |Residual|', fontsize=11, fontweight='bold')

lims = [-3.2, 3.2]
ax.plot(lims, lims, '--', color='red', linewidth=2, label='完美拟合理想线 (y = x)')
m_fit, b_fit = np.polyfit(all_reg_targets, all_regs, 1)
x_vals = np.linspace(-3, 3, 100)
ax.plot(x_vals, m_fit * x_vals + b_fit, '-', color='navy', linewidth=2, 
        label=f'实际回归拟合线 (y = {m_fit:.2f}x + {b_fit:.2f})')

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('真实情感强度标注 (Ground Truth Intensity)', fontsize=12, fontweight='bold')
ax.set_ylabel('模型预测情感强度 (Predicted Intensity)', fontsize=12, fontweight='bold')
ax.set_title(f'验证集连续情感强度拟合残差分析\nPearson r = {pr:.4f} | MAE = {mae:.4f}', 
             fontsize=13, fontweight='bold', pad=15)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left', frameon=True, fontsize=10)
plt.tight_layout()
fig9_path = os.path.join(FIG_DIR, "fig9_validation_regression_residuals.png")
plt.savefig(fig9_path, dpi=300)
plt.close()
print(f"Saved: {fig9_path}")

# 6. Generate Fig 10: Global Dynamic Reliability Gate Distribution (说明各个模态对于全局的作用)
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
gate_data = [all_gates_t, all_gates_v, all_gates_a]
labels_m = ['文本模态 (Text)', '视觉模态 (Vision)', '语音模态 (Audio)']
colors = ['#3498db', '#2ecc71', '#e74c3c']

parts = ax.violinplot(gate_data, showmeans=True, showmedians=False, showextrema=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

mean_t = float(np.mean(all_gates_t))
mean_v = float(np.mean(all_gates_v))
mean_a = float(np.mean(all_gates_a))
means = [mean_t, mean_v, mean_a]
sum_means = sum(means)

for i in range(3):
    ax.text(i + 1, means[i] + 0.03, f"均值: {means[i]:.3f}\n占比: {means[i]/sum_means*100:.1f}%", 
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors[i], alpha=0.9))

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(labels_m, fontsize=12, fontweight='bold')
ax.set_ylabel('自适应可靠性门控权重 (Gating Weight g_m)', fontsize=12, fontweight='bold')
ax.set_title('全局多模态自适应门控权重视角分布\n(定量揭示各模态在全局情感判定中的核心作用与信息量分配)', 
             fontsize=13, fontweight='bold', pad=15)
ax.grid(True, axis='y', linestyle=':', alpha=0.6)
ax.set_ylim(0, 1.15)
plt.tight_layout()
fig10_path = os.path.join(FIG_DIR, "fig10_global_modality_gate_distribution.png")
plt.savefig(fig10_path, dpi=300)
plt.close()
print(f"Saved: {fig10_path}")

# 7. Error Attribution Analysis: Identify and catalog typical errors
errors = []
for i in range(n_samples):
    t_cls = int(all_cls_targets[i])
    p_cls = int(y_pred_3[i])
    t_reg = float(all_reg_targets[i])
    p_reg = float(all_regs[i])
    abs_err = abs(t_reg - p_reg)
    
    if t_cls != p_cls:
        if t_cls == 1:
            error_type = "中性边界敏感漂移 (Neutral-Boundary-Shift)"
        elif abs(t_cls - p_cls) == 2:
            error_type = "严重极性倒置 (Polarity-Inversion)"
        else:
            error_type = "弱情感混淆 (Weak-Sentiment-Confusion)"
            
        errors.append({
            "index": i,
            "id": all_ids[i],
            "true_cls": t_cls,
            "pred_cls": p_cls,
            "true_reg": t_reg,
            "pred_reg": p_reg,
            "abs_error": abs_err,
            "gate_text": float(all_gates_t[i]),
            "gate_audio": float(all_gates_a[i]),
            "gate_vision": float(all_gates_v[i]),
            "error_type": error_type
        })

errors_sorted = sorted(errors, key=lambda x: x['abs_error'], reverse=True)

error_summary = {
    "total_val_samples": n_samples,
    "accuracy_3class": float(acc_3),
    "mae": float(mae),
    "pearson_r": float(pr),
    "global_modality_weights": {
        "text_mean_gate": mean_t,
        "vision_mean_gate": mean_v,
        "audio_mean_gate": mean_a,
        "text_normalized_pct": float(mean_t / sum_means * 100),
        "vision_normalized_pct": float(mean_v / sum_means * 100),
        "audio_normalized_pct": float(mean_a / sum_means * 100)
    },
    "total_misclassified": len(errors),
    "error_distribution": {
        "中性边界敏感漂移": len([e for e in errors if e['error_type'] == "中性边界敏感漂移 (Neutral-Boundary-Shift)"]),
        "弱情感混淆": len([e for e in errors if e['error_type'] == "弱情感混淆 (Weak-Sentiment-Confusion)"]),
        "严重极性倒置": len([e for e in errors if e['error_type'] == "严重极性倒置 (Polarity-Inversion)"])
    },
    "typical_error_cases": errors_sorted[:10]
}

out_json_path = os.path.join(BASE_DIR, "validation_error_attribution_summary.json")
with open(out_json_path, 'w', encoding='utf-8') as f:
    json.dump(error_summary, f, ensure_ascii=False, indent=4)

print(f"Saved error summary to {out_json_path}")
print("All validation and error analysis tasks completed successfully!")
