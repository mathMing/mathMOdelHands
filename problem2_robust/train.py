import os
import sys
import random
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr

# ================= 配置区域 =================
ROOT_DIR = "/root/E_Problem" if os.path.exists("/root/E_Problem") else r"X:\mathModelHands"
PKL_PATH = os.path.join(ROOT_DIR, "DATA", "Att2-Extracted-Features", "aligned_50.pkl")
SAVE_DIR = os.path.join(ROOT_DIR, "problem2_robust")
MODEL_PATH = os.path.join(SAVE_DIR, "best_robust_model.pt")

BATCH_SIZE = 32
EPOCHS = 80
PATIENCE = 5
LR = 5e-4
WEIGHT_DECAY = 1e-4
D_MODEL = 256
NUM_HEADS = 4
# 3个黄金种子集成 (针对 3050 本地环境兼顾效率与理论最优)
SEEDS = [42, 123, 777] 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(cls_preds, cls_targets, reg_preds, reg_targets):
    acc3 = accuracy_score(cls_targets, cls_preds)
    f1_3 = f1_score(cls_targets, cls_preds, average='macro', zero_division=0)
    mae = np.mean(np.abs(reg_preds - reg_targets))
    try:
        r, _ = pearsonr(reg_preds, reg_targets)
        if np.isnan(r): r = 0.0
    except:
        r = 0.0
    # 顶会标准二分类指标 (Non-neg vs Neg)
    acc2 = accuracy_score((reg_targets >= 0).astype(int), (reg_preds >= 0).astype(int))
    f1_2 = f1_score((reg_targets >= 0).astype(int), (reg_preds >= 0).astype(int), average='weighted', zero_division=0)
    return acc3, f1_3, acc2, f1_2, mae, r

def evaluate_loader(model, loader):
    model.eval()
    all_logits, all_regs = [], []
    all_cls_targets, all_reg_targets = [], []
    total_loss = 0.0
    
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.L1Loss()
    
    with torch.no_grad():
        for batch in loader:
            t = batch['text'].to(DEVICE)
            a = batch['audio'].to(DEVICE)
            v = batch['vision'].to(DEVICE)
            mt = batch['mask_t'].to(DEVICE)
            ma = batch['mask_a'].to(DEVICE)
            mv = batch['mask_v'].to(DEVICE)
            
            c_target = batch['cls_label'].to(DEVICE)
            r_target = batch['reg_label'].to(DEVICE)
            
            out = model(t, a, v, mt, ma, mv)
            
            loss_c = cls_criterion(out['cls_logits'], c_target)
            loss_r = reg_criterion(out['reg_output'], r_target)
            total_loss += (loss_c + loss_r).item()
            
            all_logits.append(out['cls_logits'].cpu().numpy())
            all_regs.append(out['reg_output'].cpu().numpy())
            all_cls_targets.extend(c_target.cpu().numpy())
            all_reg_targets.extend(r_target.cpu().numpy())
            
    all_logits = np.concatenate(all_logits, axis=0)
    all_regs = np.concatenate(all_regs, axis=0)
    all_cls_targets = np.array(all_cls_targets)
    all_reg_targets = np.array(all_reg_targets)
    
    all_cls_preds = np.argmax(all_logits, axis=-1)
    acc3, f1_3, acc2, f1_2, mae, r = compute_metrics(all_cls_preds, all_cls_targets, all_regs, all_reg_targets)
    
    return {
        'loss': total_loss / len(loader),
        'acc3': acc3,
        'f1_3': f1_3,
        'acc2': acc2,
        'f1_2': f1_2,
        'mae': mae,
        'r': r,
        'logits': all_logits,
        'regs': all_regs,
        'cls_targets': all_cls_targets,
        'reg_targets': all_reg_targets
    }

def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(1e-2, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

def train_single_seed(seed, train_loader, valid_loader, train_set):
    set_seed(seed)
    print(f"\n==================== 开始训练 Seed {seed} 深度鲁棒模型 ====================")
    
    from model import RobustMultimodalModel
    model = RobustMultimodalModel(d_model=D_MODEL, num_heads=NUM_HEADS).to(DEVICE)
    
    class_weights = torch.tensor([1.15, 1.45, 0.70], dtype=torch.float32).to(DEVICE)
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    reg_criterion = nn.SmoothL1Loss()
    
    # 序数拓扑惩罚矩阵 (Wasserstein Ordinal Penalty)
    # 严重极性倒置 (负->正 或 正->负) 惩罚 4.0; 相邻错误 (负->中, 中->正) 惩罚 1.0
    wod_matrix = torch.tensor([
        [0.0, 1.0, 4.0],
        [1.0, 0.0, 1.0],
        [4.0, 1.0, 0.0]
    ], dtype=torch.float32).to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_warmup_scheduler(optimizer, warmup_epochs=2, total_epochs=EPOCHS)
    
    best_score = -float('inf')
    best_ckpt = None
    best_val_res = None
    best_ep = 0
    stale_epochs = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            t_s = batch['text'].to(DEVICE)
            a_s = batch['audio'].to(DEVICE)
            v_s = batch['vision'].to(DEVICE)
            mt_s = batch['mask_t'].to(DEVICE)
            ma_s = batch['mask_a'].to(DEVICE)
            mv_s = batch['mask_v'].to(DEVICE)
            
            c_target = batch['cls_label'].to(DEVICE)
            r_target = batch['reg_label'].to(DEVICE)
            
            # Teacher 视角 (接收完备未受损特征)
            t_t = batch['text_full'].to(DEVICE)
            a_t = batch['audio_full'].to(DEVICE)
            v_t = batch['vision_full'].to(DEVICE)
            mt_t = batch['mask_t_full'].to(DEVICE)
            ma_t = batch['mask_a_full'].to(DEVICE)
            mv_t = batch['mask_v_full'].to(DEVICE)
            
            with torch.no_grad():
                out_t = model(t_t, a_t, v_t, mt_t, ma_t, mv_t)
                z_teacher = out_t['fused_feature'].detach()
                logits_teacher = out_t['cls_logits'].detach()
                
            optimizer.zero_grad()
            out_s = model(t_s, a_s, v_s, mt_s, ma_s, mv_s)
            z_student = out_s['fused_feature']
            logits_student = out_s['cls_logits']
            reg_student = out_s['reg_output']
            
            loss_c = cls_criterion(logits_student, c_target)
            loss_r = reg_criterion(reg_student, r_target)
            
            # 1. 难度自适应蒸馏 (Difficulty-Aware Adaptive Distillation, 参考 ICCV 2025 CMAD)
            # 当文本缺失时缺失严重度最高，提升蒸馏权重迫使学生挖掘音视特征
            missing_severity = 1.0 - (mt_s.mean(dim=-1) * 0.6 + ma_s.mean(dim=-1) * 0.2 + mv_s.mean(dim=-1) * 0.2)
            diff_weight = (1.0 + 1.5 * missing_severity).unsqueeze(-1)
            loss_distill_feat = ((1.0 - F.cosine_similarity(z_student, z_teacher, dim=-1, eps=1e-6)).unsqueeze(-1) * diff_weight).mean()
            loss_distill_kl = 4.0 * F.kl_div(
                F.log_softmax(logits_student / 2.0, dim=-1),
                F.softmax(logits_teacher / 2.0, dim=-1),
                reduction='batchmean'
            )
            
            # 2. Wasserstein 序数拓扑惩罚损失 (严厉惩罚极端越级极性倒置)
            probs = F.softmax(logits_student, dim=-1)
            cost_vec = wod_matrix[c_target] # (B, 3)
            loss_wod = (probs * cost_vec).sum(dim=-1).mean()
            
            # 3. 序数连续边缘一致性
            loss_ordinal = torch.relu(0.12 - (probs[:, 2] - probs[:, 0]) * reg_student).mean()
            
            loss = loss_c + 0.85 * loss_r + 0.15 * loss_distill_feat + 0.05 * loss_distill_kl + 0.08 * loss_wod + 0.03 * loss_ordinal
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        train_loss /= len(train_loader)
        
        val_res = evaluate_loader(model, valid_loader)
        val_score = val_res['f1_3'] + val_res['r'] - val_res['mae']
        
        if epoch % 2 == 0 or epoch == EPOCHS:
            print(f"Seed {seed} | Ep [{epoch:02d}/{EPOCHS}] | Train: {train_loss:.4f} | "
                  f"Val: Acc-3={val_res['acc3']*100:.2f}%, F1={val_res['f1_3']*100:.2f}%, MAE={val_res['mae']:.4f}, R={val_res['r']:.4f} | "
                  f"Val: Acc-3={val_res['acc3']*100:.2f}%, Acc-2={val_res['acc2']*100:.2f}%, MAE={val_res['mae']:.4f}, R={val_res['r']:.4f}")
                  
        if val_score > best_score:
            best_score = val_score
            best_ep = epoch
            best_val_res = val_res
            best_ckpt = {
                'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'audio_mean': train_set.audio_mean,
                'audio_std': train_set.audio_std,
                'vision_mean': train_set.vision_mean,
                'vision_std': train_set.vision_std,
                'epoch': epoch,
                'val_metrics': (val_res['acc3'], val_res['f1_3'], val_res['acc2'], val_res['f1_2'], val_res['mae'], val_res['r']),
                'd_model': D_MODEL,
                'num_heads': NUM_HEADS
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= PATIENCE:
            print(f">>> Seed {seed} 在 {PATIENCE} 个 epoch 无验证集提升，提前停止于 epoch {epoch}")
            break
            
    # 集成和后续专项推理必须使用验证集最佳权重，而不是最后一轮权重。
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    print(f">>> Seed {seed} 最佳 Epoch: {best_ep}, Val F1={best_val_res['f1_3']*100:.2f}%")
    return model, best_ckpt, best_val_res

def train():
    from dataset import get_dataloaders
    from model import RobustMultimodalModel
    
    print(f"正在加载附件2对齐特征数据: {PKL_PATH}")
    train_loader, valid_loader, test_loader, train_set = get_dataloaders(PKL_PATH, batch_size=BATCH_SIZE)
    print(f"数据加载就绪: Train={len(train_loader.dataset)}, Valid={len(valid_loader.dataset)}, Test={len(test_loader.dataset)}")
    
    trained_models = []
    trained_ckpts = []
    overall_best_score = -float('inf')
    overall_best_ckpt = None
    
    for seed in SEEDS:
        model, ckpt, v_res = train_single_seed(seed, train_loader, valid_loader, train_set)
        trained_models.append(model)
        trained_ckpts.append(ckpt)
        
        score = v_res['f1_3'] + v_res['r'] - v_res['mae']
        if score > overall_best_score:
            overall_best_score = score
            overall_best_ckpt = ckpt
            
    # ================= 软概率集成与贝叶斯不确定度动态阈值 (BUDT) =================
    print("\n========== 执行 3-Seed 多模型软概率加权集成与贝叶斯动态阈值校准 ==========")
    
    def get_ensemble_predictions(models, loader):
        all_probs = []
        all_regs = []
        all_uncs = []
        targets_c = []
        targets_r = []
        
        with torch.no_grad():
            for batch in loader:
                t = batch['text'].to(DEVICE)
                a = batch['audio'].to(DEVICE)
                v = batch['vision'].to(DEVICE)
                mt = batch['mask_t'].to(DEVICE)
                ma = batch['mask_a'].to(DEVICE)
                mv = batch['mask_v'].to(DEVICE)
                
                batch_probs = []
                batch_regs = []
                for m in models:
                    out = m(t, a, v, mt, ma, mv)
                    prob = torch.softmax(out['cls_logits'], dim=-1).cpu().numpy()
                    reg = out['reg_output'].cpu().numpy()
                    batch_probs.append(prob)
                    batch_regs.append(reg)
                    
                regs_matrix = np.array(batch_regs) # (num_models, B)
                avg_prob = np.mean(batch_probs, axis=0)
                avg_reg = np.mean(batch_regs, axis=0)
                unc = np.std(regs_matrix, axis=0) # 预测标准差表征贝叶斯不确定度 (B,)
                
                all_probs.append(avg_prob)
                all_regs.append(avg_reg)
                all_uncs.append(unc)
                targets_c.extend(batch['cls_label'].numpy())
                targets_r.extend(batch['reg_label'].numpy())
                
        probs = np.concatenate(all_probs, axis=0)
        regs = np.concatenate(all_regs, axis=0)
        uncs = np.concatenate(all_uncs, axis=0)
        targets_c = np.array(targets_c)
        targets_r = np.array(targets_r)
        
        preds_argmax = np.argmax(probs, axis=-1)
        acc3, f1_3, acc2, f1_2, mae, r = compute_metrics(preds_argmax, targets_c, regs, targets_r)
        
        return {
            'acc3': acc3, 'f1_3': f1_3, 'acc2': acc2, 'f1_2': f1_2, 'mae': mae, 'r': r,
            'probs': probs, 'regs': regs, 'uncs': uncs, 'targets_c': targets_c, 'targets_r': targets_r
        }

    val_ens = get_ensemble_predictions(trained_models, valid_loader)
    test_ens = get_ensemble_predictions(trained_models, test_loader)
    
    print("\n【多模型集成测试集表现 (Argmax 投票)】:")
    print(f"  Test Acc-3: {test_ens['acc3']*100:.2f}% | Macro-F1: {test_ens['f1_3']*100:.2f}%")
    print(f"  Test Acc-2: {test_ens['acc2']*100:.2f}% | Weighted-F1: {test_ens['f1_2']*100:.2f}%")
    print(f"  Test MAE:   {test_ens['mae']:.4f} | Pearson r: {test_ens['r']:.4f}")
    
    # 验证集网格搜索最优贝叶斯动态阈值: tau_neg = -th1 - gamma * unc, tau_pos = th2 + gamma * unc
    best_th = (0.10, 0.15)
    best_gamma = 0.0
    best_val_f1 = 0.0
    for th1 in np.linspace(0.04, 0.36, 17):
        for th2 in np.linspace(0.04, 0.36, 17):
            for gamma in np.linspace(0.0, 0.25, 6):
                tau_neg = -th1 - gamma * val_ens['uncs']
                tau_pos = th2 + gamma * val_ens['uncs']
                cal_pred = np.ones_like(val_ens['targets_c'])
                cal_pred[val_ens['regs'] < tau_neg] = 0
                cal_pred[val_ens['regs'] > tau_pos] = 2
                f1 = f1_score(val_ens['targets_c'], cal_pred, average='macro', zero_division=0)
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_th = (th1, th2)
                    best_gamma = gamma
                
    th_neg, th_pos = best_th
    tau_test_neg = -th_neg - best_gamma * test_ens['uncs']
    tau_test_pos = th_pos + best_gamma * test_ens['uncs']
    cal_test_preds = np.ones_like(test_ens['targets_c'])
    cal_test_preds[test_ens['regs'] < tau_test_neg] = 0
    cal_test_preds[test_ens['regs'] > tau_test_pos] = 2
    cal_acc3, cal_f1_3, _, _, _, _ = compute_metrics(cal_test_preds, test_ens['targets_c'], test_ens['regs'], test_ens['targets_r'])
    
    print(f"\n【贝叶斯动态阈值校准边界 (Base [-{th_neg:.3f}, +{th_pos:.3f}], Gamma={best_gamma:.3f})】:")
    print(f"  校准后测试集 Acc-3: {cal_acc3*100:.2f}% | 校准后 Macro-F1: {cal_f1_3*100:.2f}%")
    
    final_acc3 = cal_acc3
    final_f1_3 = cal_f1_3
    final_acc2 = test_ens['acc2']
    final_f1_2 = test_ens['f1_2']
    final_mae = test_ens['mae']
    final_r = test_ens['r']

    # 保存完整的 3-Seed 集成模型与最优单模型
    overall_best_ckpt['ensemble_state_dicts'] = [c['model_state_dict'] for c in trained_ckpts]
    overall_best_ckpt['calibrated_thresholds'] = [-round(float(th_neg), 3), round(float(th_pos), 3)]
    overall_best_ckpt['calibrated_gamma'] = round(float(best_gamma), 4)
    overall_best_ckpt['ensemble_calibrated_metrics'] = {
        'Accuracy_3Class': round(float(final_acc3), 4),
        'Macro_F1_3Class': round(float(final_f1_3), 4),
        'Accuracy_2Class_Benchmark': round(float(final_acc2), 4),
        'Weighted_F1_2Class': round(float(final_f1_2), 4),
        'MAE': round(float(final_mae), 4),
        'Pearson_r': round(float(final_r), 4)
    }
    torch.save(overall_best_ckpt, MODEL_PATH)
    print(f"\n>>> 包含 3-Seed 全套集成权重与 BUDT 阈值已保存至: {MODEL_PATH}")

    # 保存关键指标到 JSON 和 CSV 文件
    p2_data = {
        'task': '问题二：多模态鲁棒模型 (4头跨模态注意力 + 3-Seed集成 + 极性冲突感知门控 + Wasserstein序数校准 + 贝叶斯动态阈值)',
        'best_single_model_epoch': int(overall_best_ckpt['epoch']),
        'best_single_model_validation_metrics': {
            'Accuracy_3Class': round(float(overall_best_ckpt['val_metrics'][0]), 4),
            'Macro_F1_3Class': round(float(overall_best_ckpt['val_metrics'][1]), 4),
            'Accuracy_2Class_Benchmark': round(float(overall_best_ckpt['val_metrics'][2]), 4),
            'Weighted_F1_2Class': round(float(overall_best_ckpt['val_metrics'][3]), 4),
            'MAE': round(float(overall_best_ckpt['val_metrics'][4]), 4),
            'Pearson_r': round(float(overall_best_ckpt['val_metrics'][5]), 4)
        },
        'ensemble_calibrated_metrics': overall_best_ckpt['ensemble_calibrated_metrics']
    }
    with open(os.path.join(SAVE_DIR, "problem2_key_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(p2_data, f, ensure_ascii=False, indent=4)

    df_p2 = pd.DataFrame([
        {
            '评估方案': '最优单模型验证集指标 (Best Single Model, Validation)',
            '三分类准确率 (Acc-3)': f"{float(overall_best_ckpt['val_metrics'][0])*100:.2f}%",
            '三分类宏 F1 (Macro-F1)': f"{float(overall_best_ckpt['val_metrics'][1])*100:.2f}%",
            '二分类基准准确率 (Acc-2)': f"{float(overall_best_ckpt['val_metrics'][2])*100:.2f}%",
            '二分类加权 F1 (F1-2)': f"{float(overall_best_ckpt['val_metrics'][3])*100:.2f}%",
            '平均绝对误差 (MAE)': round(float(overall_best_ckpt['val_metrics'][4]), 4),
            '皮尔逊相关系数 (Pearson r)': round(float(overall_best_ckpt['val_metrics'][5]), 4)
        },
        {
            '评估方案': '3-Seed 加权集成与序数校准 (Ensemble Calibrated + BUDT)',
            '三分类准确率 (Acc-3)': f"{float(final_acc3)*100:.2f}%",
            '三分类宏 F1 (Macro-F1)': f"{float(final_f1_3)*100:.2f}%",
            '二分类基准准确率 (Acc-2)': f"{float(final_acc2)*100:.2f}%",
            '二分类加权 F1 (F1-2)': f"{float(final_f1_2)*100:.2f}%",
            '平均绝对误差 (MAE)': round(float(final_mae), 4),
            '皮尔逊相关系数 (Pearson r)': round(float(final_r), 4)
        }
    ])
    df_p2.to_csv(os.path.join(SAVE_DIR, "problem2_key_metrics.csv"), index=False, encoding='utf-8-sig')
    print("\n关键指标已成功保存至 problem2_key_metrics.json 和 problem2_key_metrics.csv！")

if __name__ == "__main__":
    train()
