import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossModalAttention(nn.Module):
    """
    4头多头跨模态注意力交互模块 (Multi-Head Cross-Modal Attention, MHCA)
    集成多头缩放点积、缺失安全掩码与前馈网络 (FFN) 残差块，全面对标 MulT 架构
    """
    def __init__(self, d_model=256, num_heads=4, dropout=0.15):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model 必须被 num_heads 整除"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # 前馈残差网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, query, key_value, valid_mask=None):
        # query: (B, S, D), key_value: (B, S, D)
        # valid_mask: (B, S), 1 为有效, 0 为缺失
        B, S, D = query.shape

        q = self.q_proj(query).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)     # (B, H, S, head_dim)
        k = self.k_proj(key_value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, S, head_dim)
        v = self.v_proj(key_value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, S, head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # (B, H, S, S)

        if valid_mask is not None:
            # valid_mask: (B, S) -> (B, 1, 1, S)
            mask_exp = valid_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_exp < 0.5, torch.finfo(scores.dtype).min)

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        if valid_mask is not None:
            weights = weights * valid_mask.unsqueeze(1).unsqueeze(2)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            all_missing = (valid_mask.sum(dim=-1) == 0).view(B, 1, 1, 1)
            weights = torch.where(all_missing, torch.zeros_like(weights), weights)
        weights = self.drop(weights)

        attn_out = torch.matmul(weights, v) # (B, H, S, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.out_proj(attn_out)

        # 第一次残差与归一化
        x = self.norm1(query + self.drop(attn_out))
        # 第二次 FFN 残差与归一化
        x = self.norm2(x + self.ffn(x))
        return x

class RobustMultimodalModel(nn.Module):
    """
    高阶鲁棒多模态情感预测网络 (Deep Robust Multimodal Network, DRM-Net)
    深度融合：
      1. 4头多头跨模态注意力补齐 (MHCA)
      2. 模态内双向深层时序编码 (Bi-GRU + LayerNorm)
      3. 可学习缺失 Prompt 自适应填充 (思路 C)
      4. 动态信噪比置信度门控分配器 (Dynamic Reliability Gate)
    """
    def __init__(self, d_text=768, d_audio=74, d_vision=35, d_model=256, num_heads=4, num_classes=3,
                 use_cross_attention=True, use_dynamic_gate=True):
        super().__init__()
        self.d_model = d_model
        self.use_cross_attention = use_cross_attention
        self.use_dynamic_gate = use_dynamic_gate

        # 1. 单模态高维特征对齐投射层
        self.proj_t = nn.Sequential(
            nn.Linear(d_text, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.2)
        )
        self.proj_a = nn.Sequential(
            nn.Linear(d_audio, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.2)
        )
        self.proj_v = nn.Sequential(
            nn.Linear(d_vision, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.2)
        )

        # 思路 C：可学习缺失代理向量 (Learnable Missing Prompts)
        self.prompt_a = nn.Parameter(torch.zeros(1, 1, d_model))
        self.prompt_v = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.prompt_a, std=0.02)
        nn.init.normal_(self.prompt_v, std=0.02)

        # 2. 模态内深层时序编码器 (Bi-GRU, hidden_size = d_model // 2)
        self.gru_t = nn.GRU(d_model, d_model // 2, batch_first=True, bidirectional=True)
        self.gru_a = nn.GRU(d_model, d_model // 2, batch_first=True, bidirectional=True)
        self.gru_v = nn.GRU(d_model, d_model // 2, batch_first=True, bidirectional=True)

        self.norm_t = nn.LayerNorm(d_model)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)

        # 3. 4头多头跨模态注意力时空插值交互网络
        self.cross_t = MultiHeadCrossModalAttention(d_model=d_model, num_heads=num_heads, dropout=0.15)
        self.cross_a = MultiHeadCrossModalAttention(d_model=d_model, num_heads=num_heads, dropout=0.15)
        self.cross_v = MultiHeadCrossModalAttention(d_model=d_model, num_heads=num_heads, dropout=0.15)

        # 4. 高阶极性冲突与动态信度门控 (Conflict & Reliability-Aware Gating, CAG)
        # 输入维度: sum_t(256) + sum_a(256) + sum_v(256) + ratios(3) + cross_modal_similarities(3) = d_model * 3 + 6
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model * 3 + 6, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3) # 输出 (g_t, g_a, g_v)
        )

        # 5. 双任务预测头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes) # 3 分类
        )
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1) # 回归连续强度 [-3, 3]
        )

    def forward(self, text, audio, vision, mask_t, mask_a, mask_v):
        """
        前向推断计算流程
        text: (B, 50, 768)
        audio: (B, 50, 74)
        vision: (B, 50, 35)
        mask_*: (B, 50), 1 为有效, 0 为缺失
        """
        B, seq_len, _ = text.shape

        # 1. 投射到同一高维隐空间
        h_t = self.proj_t(text)
        h_a = self.proj_a(audio)
        h_v = self.proj_v(vision)

        # 思路 C：注入自适应可学习缺失先验
        m_a_exp = mask_a.unsqueeze(-1)
        m_v_exp = mask_v.unsqueeze(-1)
        h_a = m_a_exp * h_a + (1.0 - m_a_exp) * self.prompt_a
        h_v = m_v_exp * h_v + (1.0 - m_v_exp) * self.prompt_v

        # 2. 模态内时序编码
        out_t, _ = self.gru_t(h_t)
        out_a, _ = self.gru_a(h_a)
        out_v, _ = self.gru_v(h_v)

        h_t = self.norm_t(h_t + out_t)
        h_a = self.norm_a(h_a + out_a)
        h_v = self.norm_v(h_v + out_v)

        # 3. 跨模态时空补齐机制
        valid_av = torch.clamp(mask_a + mask_v, max=1.0)
        valid_tv = torch.clamp(mask_t + mask_v, max=1.0)
        valid_ta = torch.clamp(mask_t + mask_a, max=1.0)

        if self.use_cross_attention:
            imputed_t = self.cross_t(h_t, 0.5 * (h_a + h_v), valid_mask=valid_av)
            imputed_a = self.cross_a(h_a, 0.5 * (h_t + h_v), valid_mask=valid_tv)
            imputed_v = self.cross_v(h_v, 0.5 * (h_t + h_a), valid_mask=valid_ta)
        else:
            imputed_t, imputed_a, imputed_v = h_t, h_a, h_v

        # 4. 残差自适应融合
        m_t_expand = mask_t.unsqueeze(-1)
        m_a_expand = mask_a.unsqueeze(-1)
        m_v_expand = mask_v.unsqueeze(-1)

        final_t = m_t_expand * h_t + (1.0 - m_t_expand) * imputed_t
        final_a = m_a_expand * h_a + (1.0 - m_a_expand) * imputed_a
        final_v = m_v_expand * h_v + (1.0 - m_v_expand) * imputed_v

        # 5. 全局时序池化
        def masked_pool(features, mask):
            valid = mask.unsqueeze(-1)
            return (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        sum_t = masked_pool(final_t, mask_t)
        sum_a = masked_pool(final_a, mask_a)
        sum_v = masked_pool(final_v, mask_v)

        # 6. 极性冲突感知与动态可靠度门控评估 (Conflict-Aware Gating, CAG)
        ratio_t = (mask_t.sum(dim=1, keepdim=True) / seq_len)
        ratio_a = (mask_a.sum(dim=1, keepdim=True) / seq_len)
        ratio_v = (mask_v.sum(dim=1, keepdim=True) / seq_len)
        
        # 计算跨模态隐层余弦相似度 (检测反讽与极性冲突)
        sim_ta = F.cosine_similarity(sum_t, sum_a, dim=-1, eps=1e-6).unsqueeze(-1)
        sim_tv = F.cosine_similarity(sum_t, sum_v, dim=-1, eps=1e-6).unsqueeze(-1)
        sim_av = F.cosine_similarity(sum_a, sum_v, dim=-1, eps=1e-6).unsqueeze(-1)
        
        gate_feat = torch.cat([sum_t, sum_a, sum_v, ratio_t, ratio_a, ratio_v, sim_ta, sim_tv, sim_av], dim=-1)

        if self.use_dynamic_gate:
            raw_gates = self.gate_fc(gate_feat) # (B, 3)
            prior_mask = torch.cat([ratio_t, ratio_a, ratio_v], dim=-1) + 1e-3
            gates = F.softmax(raw_gates, dim=-1) * prior_mask
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            prior_mask = torch.cat([ratio_t, ratio_a, ratio_v], dim=-1)
            gates = prior_mask / prior_mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        g_t = gates[:, 0:1]
        g_a = gates[:, 1:2]
        g_v = gates[:, 2:3]

        # 7. 加权门控多模态融合特征
        fused = g_t * sum_t + g_a * sum_a + g_v * sum_v

        # 8. 双任务分支输出
        cls_logits = self.classifier(fused)
        reg_output = self.regressor(fused).squeeze(-1) # (B,)

        return {
            'cls_logits': cls_logits,
            'reg_output': reg_output,
            'gates': gates,
            'fused_feature': fused,
            'fused_repr': fused,
            'final_t': final_t,
            'final_a': final_a,
            'final_v': final_v
        }
