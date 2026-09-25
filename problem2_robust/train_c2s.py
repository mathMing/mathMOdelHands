import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import pearsonr

ROOT_DIR = r"X:\mathModelHands"
sys.path.append(os.path.join(ROOT_DIR, "problem2_robust"))
from dataset import get_dataloaders
from model import MultiHeadCrossModalAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PKL_PATH = os.path.join(ROOT_DIR, "DATA", "Att2-Extracted-Features", "aligned_50.pkl")

class CascadedRobustModel(nn.Module):
    """
    二阶级联解耦高阶鲁棒网络 (Cascaded Two-Stage Robust Network, C2S-Net)
    1. 保持 4头跨模态注意力 (MHCA) + 冲突感知门控 (CAG)
    2. 新增解耦头部:
       - Subjectivity Head (中性/客观 vs 情绪激活)
       - Valence Head (情绪极性: 正向 vs 负向)
       - Standard 3-Class Head (带序数拓扑惩罚)
       - Continuous Regressor (强度标定)
    """
    def __init__(self, d_text=768, d_audio=74, d_vision=35, d_model=256, num_heads=4):
        super().__init__()
        self.d_model = d_model

        # 单模态投射
        self.proj_t = nn.Sequential(nn.Linear(d_text, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(0.15))
        self.proj_a = nn.Sequential(nn.Linear(d_audio, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(0.15))
        self.proj_v = nn.Sequential(nn.Linear(d_vision, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(0.15))

        # 可学习缺失填充 Prompt
        self.prompt_a = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.prompt_v = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 模态内时序 GRU
        self.gru_t = nn.GRU(d_model, d_model // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.gru_a = nn.GRU(d_model, d_model // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.gru_v = nn.GRU(d_model, d_model // 2, num_layers=1, bidirectional=True, batch_first=True)

        self.norm_t = nn.LayerNorm(d_model)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)

        # 4头跨模态注意力
        self.cross_t = MultiHeadCrossModalAttention(d_model=d_model, num_heads=num_heads, dropout=0.15)
        self.cross_a = MultiHeadCrossModalAttention(d_model=d_model, num_heads=num_heads, dropout=0.15)
        self.cross_v = MultiHeadCrossModalAttention(d_model=d_model, num_heads=num_heads, dropout=0.15)

        # 冲突感知门控 (CAG)
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model * 3 + 3 + 3, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 3)
        )

        # 1. 传统 3 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )

        # 2. 连续回归头
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        # 3. 阶段 1 解耦头: 中性滤网 (Subjectivity / Neutrality Detector)
        # 0: 有情绪 (Subjective), 1: 中性 (Neutral)
        self.subj_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

        # 4. 阶段 2 解耦头: 极性判别 (Valence: Neg vs Pos)
        # 0: Negative, 1: Positive
        self.val_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, text, audio, vision, mask_t, mask_a, mask_v):
        B, seq_len, _ = text.shape

        h_t = self.proj_t(text)
        h_a = self.proj_a(audio)
        h_v = self.proj_v(vision)

        m_a_exp = mask_a.unsqueeze(-1)
        m_v_exp = mask_v.unsqueeze(-1)
        h_a = m_a_exp * h_a + (1.0 - m_a_exp) * self.prompt_a
        h_v = m_v_exp * h_v + (1.0 - m_v_exp) * self.prompt_v

        out_t, _ = self.gru_t(h_t)
        out_a, _ = self.gru_a(h_a)
        out_v, _ = self.gru_v(h_v)

        h_t = self.norm_t(h_t + out_t)
        h_a = self.norm_a(h_a + out_a)
        h_v = self.norm_v(h_v + out_v)

        valid_av = torch.clamp(mask_a + mask_v, max=1.0)
        valid_tv = torch.clamp(mask_t + mask_v, max=1.0)
        valid_ta = torch.clamp(mask_t + mask_a, max=1.0)

        imputed_t = self.cross_t(h_t, 0.5 * (h_a + h_v), valid_mask=valid_av)
        imputed_a = self.cross_a(h_a, 0.5 * (h_t + h_v), valid_mask=valid_tv)
        imputed_v = self.cross_v(h_v, 0.5 * (h_t + h_a), valid_mask=valid_ta)

        m_t_expand = mask_t.unsqueeze(-1)
        m_a_expand = mask_a.unsqueeze(-1)
        m_v_expand = mask_v.unsqueeze(-1)

        final_t = m_t_expand * h_t + (1.0 - m_t_expand) * imputed_t
        final_a = m_a_expand * h_a + (1.0 - m_a_expand) * imputed_a
        final_v = m_v_expand * h_v + (1.0 - m_v_expand) * imputed_v

        sum_t = final_t.mean(dim=1)
        sum_a = final_a.mean(dim=1)
        sum_v = final_v.mean(dim=1)

        ratio_t = (mask_t.sum(dim=1, keepdim=True) / seq_len)
        ratio_a = (mask_a.sum(dim=1, keepdim=True) / seq_len)
        ratio_v = (mask_v.sum(dim=1, keepdim=True) / seq_len)

        sim_ta = F.cosine_similarity(sum_t, sum_a, dim=-1, eps=1e-6).unsqueeze(-1)
        sim_tv = F.cosine_similarity(sum_t, sum_v, dim=-1, eps=1e-6).unsqueeze(-1)
        sim_av = F.cosine_similarity(sum_a, sum_v, dim=-1, eps=1e-6).unsqueeze(-1)

        gate_feat = torch.cat([sum_t, sum_a, sum_v, ratio_t, ratio_a, ratio_v, sim_ta, sim_tv, sim_av], dim=-1)
        raw_gates = self.gate_fc(gate_feat)
        prior_mask = torch.cat([ratio_t, ratio_a, ratio_v], dim=-1) + 1e-3
        gates = F.softmax(raw_gates, dim=-1) * prior_mask
        gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)

        g_t = gates[:, 0:1]
        g_a = gates[:, 1:2]
        g_v = gates[:, 2:3]

        fused = g_t * sum_t + g_a * sum_a + g_v * sum_v

        cls_logits = self.classifier(fused)
        reg_output = self.regressor(fused).squeeze(-1)
        subj_logits = self.subj_head(fused) # (B, 2) [0: subjective, 1: neutral]
        val_logits = self.val_head(fused)   # (B, 2) [0: negative, 1: positive]

        return {
            'cls_logits': cls_logits,
            'reg_output': reg_output,
            'subj_logits': subj_logits,
            'val_logits': val_logits,
            'fused_feature': fused,
            'gates': gates
        }

def make_gaussian_ldl(reg_targets, cls_targets, device):
    """
    构造连续高斯软标签分布 (Gaussian Label Distribution Learning, LDL)
    """
    B = reg_targets.shape[0]
    ldl = torch.zeros(B, 3, device=device)
    for i in range(B):
        r = float(reg_targets[i].item())
        c = int(cls_targets[i].item())
        if c == 1:
            # 严格中性: 聚集在中性通道
            ldl[i] = torch.tensor([0.05, 0.90, 0.05], device=device)
        elif c == 0:
            # 负向: 根据连续强度向中性过渡
            # r 从 -3 到 -0.33
            p_neu = max(0.02, min(0.35, (r + 1.5) * 0.25 + 0.15))
            ldl[i] = torch.tensor([1.0 - p_neu - 0.01, p_neu, 0.01], device=device)
        else:
            # 正向: 根据连续强度向中性过渡
            # r 从 0.16 到 3.0
            p_neu = max(0.02, min(0.35, (-r + 1.5) * 0.25 + 0.15))
            ldl[i] = torch.tensor([0.01, p_neu, 1.0 - p_neu - 0.01], device=device)
    return ldl

def train_and_eval():
    train_loader, valid_loader, test_loader, train_set = get_dataloaders(PKL_PATH, batch_size=32)
    model = CascadedRobustModel(d_model=256, num_heads=4).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    # 学习率调度
    epochs = 14
    scheduler = LambdaLR(optimizer, lambda ep: max(1e-2, 0.5 * (1.0 + np.cos(np.pi * ep / epochs))))

    # 代价矩阵
    wod_matrix = torch.tensor([
        [0.0, 1.0, 4.0],
        [1.0, 0.0, 1.0],
        [4.0, 1.0, 0.0]
    ], device=DEVICE)

    # 中性二分类权重 (负/正 78%, 中性 22%)
    subj_weights = torch.tensor([1.0, 1.8], device=DEVICE) # 提升中性判别敏感度
    subj_criterion = nn.CrossEntropyLoss(weight=subj_weights)

    val_weights = torch.tensor([1.1, 0.8], device=DEVICE) # 负 vs 正
    val_criterion = nn.CrossEntropyLoss(weight=val_weights)

    cls_weights = torch.tensor([1.15, 1.45, 0.70], device=DEVICE)
    cls_criterion = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=0.02)
    reg_criterion = nn.SmoothL1Loss()

    best_val_acc = 0.0
    best_test_acc = 0.0

    print(">>> 启动 C2S-Net 单 Seed 原型训练探索...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            t = batch['text'].to(DEVICE)
            a = batch['audio'].to(DEVICE)
            v = batch['vision'].to(DEVICE)
            mt = batch['mask_t'].to(DEVICE)
            ma = batch['mask_a'].to(DEVICE)
            mv = batch['mask_v'].to(DEVICE)
            c_target = batch['cls_label'].to(DEVICE)
            r_target = batch['reg_label'].to(DEVICE)

            # 构造解耦标签
            # subj: 1 (中性), 0 (有情绪)
            subj_target = (c_target == 1).long()
            # val: 0 (负向), 1 (正向)
            is_emotional = (c_target != 1)
            val_target = (c_target == 2).long()

            optimizer.zero_grad()
            out = model(t, a, v, mt, ma, mv)

            loss_c = cls_criterion(out['cls_logits'], c_target)
            loss_r = reg_criterion(out['reg_output'], r_target)
            loss_subj = subj_criterion(out['subj_logits'], subj_target)

            # 只在有情绪样本上回传 valence 损失
            if is_emotional.any():
                loss_val = val_criterion(out['val_logits'][is_emotional], val_target[is_emotional])
            else:
                loss_val = 0.0

            # LDL 软标签散度
            ldl_target = make_gaussian_ldl(r_target, c_target, DEVICE)
            loss_ldl = F.kl_div(F.log_softmax(out['cls_logits'], dim=-1), ldl_target, reduction='batchmean')

            # WOD 损失
            probs = F.softmax(out['cls_logits'], dim=-1)
            loss_wod = (probs * wod_matrix[c_target]).sum(dim=-1).mean()

            total_loss = loss_c + 0.8 * loss_r + 0.4 * loss_subj + 0.4 * loss_val + 0.25 * loss_ldl + 0.08 * loss_wod
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_loss += total_loss.item()

        scheduler.step()

        # 验证评估
        model.eval()
        def evaluate(loader):
            all_c_preds, all_casc_preds, all_targets = [], [], []
            with torch.no_grad():
                for b in loader:
                    t = b['text'].to(DEVICE)
                    a = b['audio'].to(DEVICE)
                    v = b['vision'].to(DEVICE)
                    mt = b['mask_t'].to(DEVICE)
                    ma = b['mask_a'].to(DEVICE)
                    mv = b['mask_v'].to(DEVICE)
                    targets = b['cls_label'].numpy()

                    out = model(t, a, v, mt, ma, mv)
                    p_c = torch.softmax(out['cls_logits'], dim=-1).cpu().numpy()
                    p_subj = torch.softmax(out['subj_logits'], dim=-1).cpu().numpy()
                    p_val = torch.softmax(out['val_logits'], dim=-1).cpu().numpy()
                    r_out = out['reg_output'].cpu().numpy()

                    # 方案 A: 直接 argmax
                    pred_c = np.argmax(p_c, axis=-1)

                    # 方案 B: 级联判别
                    # 如果 subj_head 判为中性 (p_subj[:, 1] > 0.45) 且 |reg| < 0.35 -> 判 1 (中性)
                    # 否则取 valence head (0: 负向, 1: 正向 -> 映射为 2)
                    casc = np.where(p_val[:, 1] > p_val[:, 0], 2, 0)
                    is_neu = (p_subj[:, 1] > 0.42) | (p_c[:, 1] > 0.38)
                    is_neu = is_neu & (np.abs(r_out) < 0.38)
                    casc[is_neu] = 1

                    all_c_preds.extend(pred_c)
                    all_casc_preds.extend(casc)
                    all_targets.extend(targets)

            acc_c = accuracy_score(all_targets, all_c_preds)
            acc_casc = accuracy_score(all_targets, all_casc_preds)
            f1_casc = f1_score(all_targets, all_casc_preds, average='macro')
            return acc_c, acc_casc, f1_casc

        v_c, v_casc, v_f1 = evaluate(valid_loader)
        t_c, t_casc, t_f1 = evaluate(test_loader)

        print(f"Ep [{epoch:02d}/{epochs}] | Val Direct: {v_c*100:.2f}%, Casc: {v_casc*100:.2f}% (F1={v_f1*100:.2f}%) | "
              f"Test Direct: {t_c*100:.2f}%, Casc: {t_casc*100:.2f}% (F1={t_f1*100:.2f}%)")

if __name__ == '__main__':
    train_and_eval()
