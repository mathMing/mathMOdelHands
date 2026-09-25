import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score

from dataset import get_dataloaders
from model import RobustMultimodalModel

# ================= 配置区域 =================
ROOT_DIR = "/root/E_Problem" if os.path.exists("/root/E_Problem") else r"X:\mathModelHands"
PKL_PATH = os.path.join(ROOT_DIR, "DATA", "Att2-Extracted-Features", "aligned_50.pkl")
MODEL_PATH = os.path.join(ROOT_DIR, "problem2_robust", "best_robust_model.pt")
OUTPUT_DIR = os.path.join(ROOT_DIR, "problem2_robust")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def compute_metrics(cls_preds, cls_targets, reg_preds, reg_targets):
    acc = accuracy_score(cls_targets, cls_preds)
    f1 = f1_score(cls_targets, cls_preds, average='macro', zero_division=0)
    mae = np.mean(np.abs(reg_preds - reg_targets))
    try:
        r, _ = pearsonr(reg_preds, reg_targets)
        if np.isnan(r): r = 0.0
    except:
        r = 0.0
    return acc, f1, mae, r

def evaluate_with_custom_mask(models, loader, miss_type='none', pos='middle', ratio=0.3, start_idx_override=None):
    """
    按指定控制变量模拟缺失并评估:
    miss_type: 'none', 't', 'a', 'v', 'a_v', 't_a', 't_v'
    pos: 'early' (0-30%), 'middle' (35-65%), 'late' (70-100%)
    ratio: 缺失时长比例 (0.0 ~ 0.7)
    """
    for model in models:
        model.eval()
    all_cls_preds, all_cls_targets = [], []
    all_reg_preds, all_reg_targets = [], []

    seq_len = 50
    miss_len = int(seq_len * ratio)

    if start_idx_override is not None:
        start_idx = int(start_idx_override)
    elif pos == 'early':
        start_idx = 0
    elif pos == 'middle':
        start_idx = (seq_len - miss_len) // 2
    else: # late
        start_idx = seq_len - miss_len
    end_idx = min(seq_len, start_idx + miss_len)

    with torch.no_grad():
        for batch in loader:
            t = batch['text'].clone().to(DEVICE)
            a = batch['audio'].clone().to(DEVICE)
            v = batch['vision'].clone().to(DEVICE)
            mt = batch['mask_t'].clone().to(DEVICE)
            ma = batch['mask_a'].clone().to(DEVICE)
            mv = batch['mask_v'].clone().to(DEVICE)

            original_a, original_ma = a.clone(), ma.clone()
            original_v, original_mv = v.clone(), mv.clone()
            if miss_type != 'none' and miss_len > 0:
                if 't' in miss_type:
                    t[:, start_idx:end_idx, :] = 0.0
                    mt[:, start_idx:end_idx] = 0.0
                if 'a' in miss_type:
                    a[:, start_idx:end_idx, :] = 0.0
                    ma[:, start_idx:end_idx] = 0.0
                if 'v' in miss_type:
                    v[:, start_idx:end_idx, :] = 0.0
                    mv[:, start_idx:end_idx] = 0.0

            if miss_type != 'none' and miss_len > 0:
                if 'a' in miss_type:
                    assert torch.any(original_ma[:, start_idx:end_idx] > 0.5), '音频缺失区间未覆盖任何有效位置'
                    assert torch.any(original_a[:, start_idx:end_idx] != a[:, start_idx:end_idx]), '音频特征未发生变化'
                if 'v' in miss_type:
                    assert torch.any(original_mv[:, start_idx:end_idx] > 0.5), '视觉缺失区间未覆盖任何有效位置'
                    assert torch.any(original_v[:, start_idx:end_idx] != v[:, start_idx:end_idx]), '视觉特征未发生变化'

            batch_probs, batch_regs = [], []
            for model in models:
                out = model(t, a, v, mt, ma, mv)
                batch_probs.append(torch.softmax(out['cls_logits'], dim=-1))
                batch_regs.append(out['reg_output'])
            avg_prob = torch.stack(batch_probs).mean(dim=0)
            avg_reg = torch.stack(batch_regs).mean(dim=0)

            preds_cls = torch.argmax(avg_prob, dim=-1).cpu().numpy()
            preds_reg = avg_reg.cpu().numpy()

            all_cls_preds.extend(preds_cls)
            all_cls_targets.extend(batch['cls_label'].numpy())
            all_reg_preds.extend(preds_reg)
            all_reg_targets.extend(batch['reg_label'].numpy())

    return compute_metrics(np.array(all_cls_preds), np.array(all_cls_targets),
                           np.array(all_reg_preds), np.array(all_reg_targets))

def run_ablations():
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"正在加载最优权重: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    d_model = checkpoint.get('d_model', 256)
    num_heads = checkpoint.get('num_heads', 4)
    state_dicts = checkpoint.get('ensemble_state_dicts', [checkpoint['model_state_dict']])
    models = []
    for state_dict in state_dicts:
        model = RobustMultimodalModel(d_model=d_model, num_heads=num_heads).to(DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    print(f"加载 {len(models)} 个 seed 模型进行集成消融")

    _, valid_loader, _, _ = get_dataloaders(PKL_PATH, batch_size=64)

    records = []

    # ---------------- 实验一：缺失模态类型分析 ----------------
    print("\n[实验 1/3] 评估不同缺失模态类型的影响...")
    modality_types = [
        ('完整基准 (Full)', 'none'),
        ('缺失语音模态 (-Audio)', 'a'),
        ('缺失视觉模态 (-Vision)', 'v'),
        ('缺失文本模态 (-Text)', 't'),
        ('双缺失 (-Audio & -Vision)', 'a_v'),
        ('双缺失 (-Text & -Audio)', 't_a'),
        ('双缺失 (-Text & -Vision)', 't_v'),
    ]
    type_results = []
    for name, mtype in modality_types:
        acc, f1, mae, r = evaluate_with_custom_mask(models, valid_loader, miss_type=mtype, pos='middle', ratio=0.4)
        records.append({
            'experiment_group': '模态类型消融',
            'condition': name,
            'Accuracy': acc,
            'Macro_F1': f1,
            'MAE': mae,
            'Pearson_r': r
        })
        type_results.append((name, acc, f1, mae, r))
        print(f"  {name:30s} | Acc: {acc:.4f} | F1: {f1:.4f} | MAE: {mae:.4f} | Pearson: {r:.4f}")

    # 绘图 1: 模态类型影响
    plt.figure(figsize=(10, 5), dpi=300)
    names = [r[0] for r in type_results]
    f1s = [r[2] for r in type_results]
    maes = [r[3] for r in type_results]
    x = np.arange(len(names))
    width = 0.35

    plt.bar(x - width/2, f1s, width, label='Macro-F1 (越高越好)', color='#2b5c8f')
    plt.bar(x + width/2, maes, width, label='MAE (越低越好)', color='#d95f02')
    plt.xticks(x, names, rotation=25, ha='right')
    plt.title('不同缺失模态类型对情感预测性能的影响规律', fontsize=14, pad=12)
    plt.ylabel('指标数值', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    fig1_path = os.path.join(FIG_DIR, "fig1_missing_modality_type.png")
    plt.savefig(fig1_path)
    plt.close()
    print(f"  >>> 图表保存至: {fig1_path}")

    # ---------------- 实验二：缺失位置影响分析 ----------------
    print("\n[实验 2/3] 评估缺失位置 (前段/中段/后段) 的影响...")
    positions = [('前段缺失 (Early 0~30%)', 'early'),
                 ('中段缺失 (Middle 35~65%)', 'middle'),
                 ('后段缺失 (Late 70~100%)', 'late')]
    pos_results = []
    for name, pos in positions:
        acc, f1, mae, r = evaluate_with_custom_mask(models, valid_loader, miss_type='a_v', pos=pos, ratio=0.3)
        records.append({
            'experiment_group': '缺失位置消融',
            'condition': name,
            'Accuracy': acc,
            'Macro_F1': f1,
            'MAE': mae,
            'Pearson_r': r
        })
        pos_results.append((name, acc, f1, mae, r))
        print(f"  {name:30s} | Acc: {acc:.4f} | F1: {f1:.4f} | MAE: {mae:.4f} | Pearson: {r:.4f}")

    # 绘图 2: 缺失位置对比
    plt.figure(figsize=(8, 4.5), dpi=300)
    p_names = [r[0] for r in pos_results]
    p_f1 = [r[2] for r in pos_results]
    p_mae = [r[3] for r in pos_results]
    x_pos = np.arange(len(p_names))
    plt.plot(p_names, p_f1, marker='o', linewidth=2.5, markersize=8, color='#1b9e77', label='Macro-F1')
    plt.plot(p_names, p_mae, marker='s', linewidth=2.5, markersize=8, color='#7570b3', label='MAE')
    plt.title('音视频双模态在不同时序位置缺失下的性能对比', fontsize=14, pad=12)
    plt.ylabel('评估数值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    fig2_path = os.path.join(FIG_DIR, "fig2_missing_position.png")
    plt.savefig(fig2_path)
    plt.close()
    print(f"  >>> 图表保存至: {fig2_path}")

    # ---------------- 实验三：缺失时长衰减规律分析 ----------------
    print("\n[实验 3/3] 评估缺失时长对性能的渐进衰减规律...")
    durations = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    dur_f1, dur_mae = [], []
    for d in durations:
        acc, f1, mae, r = evaluate_with_custom_mask(models, valid_loader, miss_type='a', pos='middle', ratio=d)
        records.append({
            'experiment_group': '缺失时长衰减',
            'condition': f'缺失比例 {int(d*100)}%',
            'Accuracy': acc,
            'Macro_F1': f1,
            'MAE': mae,
            'Pearson_r': r
        })
        dur_f1.append(f1)
        dur_mae.append(mae)
        print(f"  缺失时长: {int(d*100):2d}% | Acc: {acc:.4f} | F1: {f1:.4f} | MAE: {mae:.4f} | Pearson: {r:.4f}")

    # 绘图 3: 衰减曲线
    plt.figure(figsize=(9, 4.8), dpi=300)
    pcts = [int(d * 100) for d in durations]
    plt.plot(pcts, dur_f1, marker='o', linewidth=2.5, color='#e7298a', label='Macro-F1 (分类能力)')
    plt.plot(pcts, dur_mae, marker='^', linewidth=2.5, color='#66a61e', label='MAE (强度预测误差)')
    plt.title('模态缺失时长比例渐进增加时的模型性能衰减曲线', fontsize=14, pad=12)
    plt.xlabel('模态连续缺失比例 (%)', fontsize=12)
    plt.ylabel('评估指标数值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    fig3_path = os.path.join(FIG_DIR, "fig3_missing_duration_decay.png")
    plt.savefig(fig3_path)
    plt.close()
    print(f"  >>> 图表保存至: {fig3_path}")

    # 保存消融数据表格
    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, "ablation_experiments_summary.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 全量消融实验汇总数据已保存至: {csv_path}")

if __name__ == "__main__":
    run_ablations()
