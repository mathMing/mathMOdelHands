import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from scipy.stats import pearsonr

# ================= 配置区域 =================
ROOT_DIR = r"X:\mathModelHands"
SAVE_DIR = os.path.join(ROOT_DIR, "problem2_robust")
MODEL_PATH = os.path.join(SAVE_DIR, "best_robust_model.pt")
PKL_PATH = os.path.join(ROOT_DIR, "DATA", "Att2-Extracted-Features", "aligned_50.pkl")

sys.path.append(SAVE_DIR)
from dataset import get_dataloaders
from model import RobustMultimodalModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运算设备: {DEVICE}")

# ================= 1. 加载 3-Seed DRM-Net 深度网络 =================
print("正在加载已训练的 3-Seed DRM-Net 集成模型...")
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

models = []
for sd in ckpt['ensemble_state_dicts']:
    m = RobustMultimodalModel(d_model=ckpt['d_model'], num_heads=ckpt['num_heads']).to(DEVICE)
    m.load_state_dict(sd)
    m.eval()
    models.append(m)
print(f"成功加载 {len(models)} 个深度模型种子！")

# ================= 2. 提取全量表征与预测概率 =================
train_loader, val_loader, test_loader, train_set = get_dataloaders(PKL_PATH, batch_size=32)

def extract_features_and_logits(loader):
    feats, probs, regs, targets_c, targets_r = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            t = batch['text'].to(DEVICE)
            a = batch['audio'].to(DEVICE)
            v = batch['vision'].to(DEVICE)
            mt = batch['mask_t'].to(DEVICE)
            ma = batch['mask_a'].to(DEVICE)
            mv = batch['mask_v'].to(DEVICE)
            
            b_probs, b_regs, b_feats = [], [], []
            for m in models:
                out = m(t, a, v, mt, ma, mv)
                b_probs.append(torch.softmax(out['cls_logits'], dim=-1).cpu().numpy())
                b_regs.append(out['reg_output'].cpu().numpy())
                b_feats.append(out['fused_feature'].cpu().numpy())
                
            probs.append(np.mean(b_probs, axis=0))
            regs.append(np.mean(b_regs, axis=0))
            feats.append(np.mean(b_feats, axis=0))
            targets_c.extend(batch['cls_label'].numpy())
            targets_r.extend(batch['reg_label'].numpy())
            
    return (
        np.concatenate(feats, axis=0),
        np.concatenate(probs, axis=0),
        np.concatenate(regs, axis=0),
        np.array(targets_c),
        np.array(targets_r)
    )

print("正在提取训练集、验证集与测试集隐空间高维表征...")
X_tr_f, X_tr_p, X_tr_r, y_tr_c, y_tr_r = extract_features_and_logits(train_loader)
X_val_f, X_val_p, X_val_r, y_val_c, y_val_r = extract_features_and_logits(val_loader)
X_te_f, X_te_p, X_te_r, y_te_c, y_te_r = extract_features_and_logits(test_loader)

# 构造元特征向量: [融合特征(256), 深度概率(3), 回归值(1), 绝对强度(1)]
def build_meta_matrix(feats, probs, regs):
    return np.hstack([feats, probs, regs[:, None], np.abs(regs)[:, None]])

X_tr_meta = build_meta_matrix(X_tr_f, X_tr_p, X_tr_r)
X_val_meta = build_meta_matrix(X_val_f, X_val_p, X_val_r)
X_te_meta = build_meta_matrix(X_te_f, X_te_p, X_te_r)

print(f"元特征矩阵构建完毕: Train={X_tr_meta.shape}, Val={X_val_meta.shape}, Test={X_te_meta.shape}")

# ================= 3. 训练 ExtraTrees 非线性拓扑判别器 =================
print("正在训练 ExtraTrees 非线性拓扑结构元判别器...")
# 严格使用 Train 训练，Val 评估
et_clf = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    class_weight={0: 1.1, 1: 1.6, 2: 0.8}, # 强化中性类识别权重
    random_state=42,
    n_jobs=-1
)
et_clf.fit(X_tr_meta, y_tr_c)

p_tr_et = et_clf.predict_proba(X_tr_meta)
p_val_et = et_clf.predict_proba(X_val_meta)
p_te_et = et_clf.predict_proba(X_te_meta)

print(f"ExtraTrees 单独在测试集上的直接准确率: {accuracy_score(y_te_c, np.argmax(p_te_et, axis=-1))*100:.2f}%")

# ================= 4. 验证集无泄露网格搜索最优融合权重与中性软边界 =================
print("\n正在验证集上严格网格搜索最优融合权重与中性补偿超参...")
best_val_score = 0.0
best_config = None

# w_nn: 深度网络权重, w_et: ExtraTrees 权重
# w_neut_boost: 中性概率温度补偿系数
for w_nn in np.linspace(0.25, 0.75, 11):
    w_et = 1.0 - w_nn
    for w_neut_boost in np.linspace(0.8, 1.6, 17):
        # 融合概率
        val_fused_p = w_nn * X_val_p + w_et * p_val_et
        val_fused_p[:, 1] *= w_neut_boost
        
        preds = np.argmax(val_fused_p, axis=-1)
        acc = accuracy_score(y_val_c, preds)
        f1 = f1_score(y_val_c, preds, average='macro', zero_division=0)
        score = acc + 0.5 * f1
        
        if score > best_val_score:
            best_val_score = score
            best_config = (w_nn, w_et, w_neut_boost, acc, f1)

opt_w_nn, opt_w_et, opt_neut_boost, val_acc, val_f1 = best_config
print(f"验证集最优配置: w_DRM={opt_w_nn:.2f}, w_ET={opt_w_et:.2f}, Neut_Boost={opt_neut_boost:.2f}")
print(f"验证集最优指标: Acc-3 = {val_acc*100:.2f}%, Macro-F1 = {val_f1*100:.2f}%")

# ================= 5. 在独立测试集 (Test Split, N=727) 上严格验证 =================
test_fused_p = opt_w_nn * X_te_p + opt_w_et * p_te_et
test_fused_p[:, 1] *= opt_neut_boost

test_preds = np.argmax(test_fused_p, axis=-1)
test_acc3 = accuracy_score(y_te_c, test_preds)
test_f1_3 = f1_score(y_te_c, test_preds, average='macro', zero_division=0)

# 二分类基准 (非负 vs 负)
test_acc2 = accuracy_score((y_te_r >= 0).astype(int), (X_te_r >= 0).astype(int))
test_f1_2 = f1_score((y_te_r >= 0).astype(int), (X_te_r >= 0).astype(int), average='weighted', zero_division=0)
mae = np.mean(np.abs(X_te_r - y_te_r))
r, _ = pearsonr(X_te_r, y_te_r)

print("\n" + "="*65)
print("【最终测试集 (Test Split, N=727) 深度融合评审指标】")
print("="*65)
print(f"  严格三分类准确率 (Acc-3): {test_acc3*100:.2f}%  (较基准 60.30% 暴涨 +{test_acc3*100 - 60.30:.2f}%)")
print(f"  严格三分类宏 F1 (Macro-F1): {test_f1_3*100:.2f}%")
print(f"  顶会二分类基准 (Acc-2):    {test_acc2*100:.2f}%")
print(f"  二分类加权 F1 (F1-2):      {test_f1_2*100:.2f}%")
print(f"  平均绝对误差 (MAE):        {mae:.4f}")
print(f"  皮尔逊相关系数 (Pearson r): {r:.4f}")
print("="*65)

cm = confusion_matrix(y_te_c, test_preds)
print("\n融合后混淆矩阵 (Confusion Matrix):")
print(cm)
print(classification_report(y_te_c, test_preds, target_names=['Negative', 'Neutral', 'Positive']))

# 保存融合后的模型配置与元判别器
dth_checkpoint = {
    'opt_w_nn': opt_w_nn,
    'opt_w_et': opt_w_et,
    'opt_neut_boost': opt_neut_boost,
    'et_clf': et_clf,
    'test_metrics': {
        'Accuracy_3Class': round(float(test_acc3), 4),
        'Macro_F1_3Class': round(float(test_f1_3), 4),
        'Accuracy_2Class_Benchmark': round(float(test_acc2), 4),
        'Weighted_F1_2Class': round(float(test_f1_2), 4),
        'MAE': round(float(mae), 4),
        'Pearson_r': round(float(r), 4)
    }
}
with open(os.path.join(SAVE_DIR, "dth_ensemble_meta.pkl"), "wb") as f:
    pickle.dump(dth_checkpoint, f)
print("DTH-Ensemble 元判别器配置已保存至: dth_ensemble_meta.pkl")

# 更新 key_metrics
key_metrics = {
    'task': '问题二：多模态鲁棒模型 (DRM-Net + ExtraTrees 深度混合决策 DTH-Ensemble)',
    'final_test_metrics': dth_checkpoint['test_metrics'],
    'confusion_matrix': cm.tolist()
}
with open(os.path.join(SAVE_DIR, "problem2_key_metrics.json"), "w", encoding='utf-8') as f:
    json.dump(key_metrics, f, ensure_ascii=False, indent=4)

print("\n已同步更新 problem2_key_metrics.json！下一步将使用最终融合引擎刷新附件 3 与附件 4。")
