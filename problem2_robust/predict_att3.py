import os
import glob
import pickle
import torch
import numpy as np
import pandas as pd
from transformers import BertModel

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import RobustMultimodalModel

# ================= 配置区域 =================
ROOT_DIR = "/root/E_Problem" if os.path.exists("/root/E_Problem") else r"X:\mathModelHands"

def find_att3_aligned_dir(root_dir):
    att3_base = os.path.join(root_dir, "DATA", "Att3-Incomplete-Modality-Features")
    std_dir = os.path.join(att3_base, "对齐版本")
    if os.path.exists(std_dir):
        return std_dir
    for d in os.listdir(att3_base):
        full_d = os.path.join(att3_base, d)
        if os.path.isdir(full_d):
            pkls = [f for f in os.listdir(full_d) if f.endswith('.pkl')]
            if len(pkls) == 30 and not any('δ' in f for f in pkls):
                return full_d
    return std_dir

ATT3_DIR = find_att3_aligned_dir(ROOT_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "problem2_robust", "best_robust_model.pt")
OUTPUT_CSV = os.path.join(ROOT_DIR, "problem2_robust", "附件3_模态局部缺失专项测试集预测结果.csv")
OUTPUT_CSV2 = os.path.join(ROOT_DIR, "problem2_robust", "附件3_多模态部分缺失专项测试集预测结果.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================

def get_missing_description(mask_t, mask_a, mask_v):
    """分析各模态的缺失区间，方便论文结果展示"""
    desc = []
    for name, mask in [('文本', mask_t), ('语音', mask_a), ('视觉', mask_v)]:
        zeros = np.where(mask < 0.5)[0]
        if len(zeros) > 0:
            desc.append(f"{name}缺失(步长{zeros[0]}~{zeros[-1]})")
        else:
            desc.append(f"{name}完整")
    return "; ".join(desc)

def get_att3_file(aligned_dir, i):
    suffix = f"{i:02d}.pkl"
    for f in os.listdir(aligned_dir):
        if f.endswith(suffix):
            return os.path.join(aligned_dir, f)
    raise FileNotFoundError(f"Cannot find pkl file for index {i:02d} in {aligned_dir}")

def generate_tta_views(t, a, v, mt, ma, mv):
    """
    测试时数据增强 (Test-Time Augmentation, TTA-MME)
    生成多视角时序扰动以实现贝叶斯共识平滑
    """
    views = [(t, a, v, mt, ma, mv)]
    
    def shift_zero(x, mask, shift):
        shifted = torch.zeros_like(x)
        shifted_mask = torch.zeros_like(mask)
        if shift > 0:
            shifted[:, shift:] = x[:, :-shift]
            shifted_mask[:, shift:] = mask[:, :-shift]
        else:
            step = -shift
            shifted[:, :-step] = x[:, step:]
            shifted_mask[:, :-step] = mask[:, step:]
        return shifted, shifted_mask

    # 零填充平移，避免 torch.roll 将末端时间步循环搬运到首端。
    t_roll1, mt_roll1 = shift_zero(t, mt, 1)
    a_roll1, ma_roll1 = shift_zero(a, ma, 1)
    v_roll1, mv_roll1 = shift_zero(v, mv, 1)
    views.append((t_roll1, a_roll1, v_roll1, mt_roll1, ma_roll1, mv_roll1))
    
    # 视角 2: 时序向后微偏 1 步
    t_roll2, mt_roll2 = shift_zero(t, mt, -1)
    a_roll2, ma_roll2 = shift_zero(a, ma, -1)
    v_roll2, mv_roll2 = shift_zero(v, mv, -1)
    views.append((t_roll2, a_roll2, v_roll2, mt_roll2, ma_roll2, mv_roll2))
    
    return views

def main():
    print(f"当前运行设备: {DEVICE}")
    print(f"识别到的附件3对齐目录: {ATT3_DIR}")
    print(f"正在加载最优权重: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
    d_model = checkpoint.get('d_model', 256)
    num_heads = checkpoint.get('num_heads', 4)

    # 载入多模型集成 (3-Seed Ensemble)
    if 'ensemble_state_dicts' in checkpoint:
        state_dicts = checkpoint['ensemble_state_dicts']
        print(f"成功载入 {len(state_dicts)} 个独立种子的模型参数构建平滑集成！")
    else:
        state_dicts = [checkpoint['model_state_dict']]
        print("未检测到多模型权重，使用单模型。")

    models = []
    for sd in state_dicts:
        m = RobustMultimodalModel(d_model=d_model, num_heads=num_heads).to(DEVICE)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)

    audio_mean = checkpoint['audio_mean']
    audio_std = checkpoint['audio_std']
    vision_mean = checkpoint['vision_mean']
    vision_std = checkpoint['vision_std']
    
    print("附件3采用最终锁定策略：三 seed 平均分类概率 Argmax。")

    # 加载本地缓存 BERT
    hf_cache = os.path.expanduser(r"~/.cache/huggingface/hub/models--bert-base-uncased/snapshots")
    snapshot_dirs = glob.glob(os.path.join(hf_cache, "*"))
    bert_path = snapshot_dirs[0] if snapshot_dirs else "bert-base-uncased"
    print(f"正在加载 BERT 语言模型: {bert_path}")
    bert = BertModel.from_pretrained(bert_path).to(DEVICE)
    bert.eval()

    results = []

    print("\n========== 开始对附件3专项测试集 (30个缺失样本) 执行多视角贝叶斯前向推断 ==========")
    for i in range(1, 31):
        fpath = get_att3_file(ATT3_DIR, i)
        fname = os.path.basename(fpath)

        with open(fpath, 'rb') as f:
            data = pickle.load(f)

        if 'test' in data:
            data = data['test']

        # 1. 文本模态特征与掩码获取 (兼容直接预存的 text 与需 BERT 编码的 text_bert)
        if 'text' in data:
            t_arr = data['text']
            if t_arr.ndim == 3: t_arr = t_arr[0]
            feat_t = torch.tensor(t_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mask_t_arr = (np.abs(t_arr).sum(axis=-1) > 1e-5).astype(np.float32)
            mask_t_t = torch.tensor(mask_t_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        elif 'text_bert' in data:
            t_bert = data['text_bert']
            if t_bert.ndim == 3: t_bert = t_bert[0] # (3, 50)
            t_ids = torch.tensor(t_bert[0:1], dtype=torch.long).to(DEVICE)
            t_mask = torch.tensor(t_bert[1:2], dtype=torch.long).to(DEVICE)
            t_seg = torch.tensor(t_bert[2:3], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                bert_out = bert(input_ids=t_ids, attention_mask=t_mask, token_type_ids=t_seg)
                feat_t = bert_out.last_hidden_state # (1, 50, 768)
            mask_t_arr = t_mask.cpu().numpy()[0]
            mask_t_t = torch.tensor(mask_t_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        else:
            raise KeyError(f"Neither 'text' nor 'text_bert' found in keys: {list(data.keys())}")

        # 2. 语音特征标准化
        feat_a = data['audio']
        if feat_a.ndim == 3: feat_a = feat_a[0]
        a_mask = np.abs(feat_a).sum(axis=-1) > 1e-5
        norm_a = np.where(a_mask[:, None], (feat_a - audio_mean) / audio_std, 0.0)
        feat_a_t = torch.tensor(norm_a, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 3. 视觉特征标准化
        feat_v = data['vision']
        if feat_v.ndim == 3: feat_v = feat_v[0]
        v_mask = np.abs(feat_v).sum(axis=-1) > 1e-5
        norm_v = np.where(v_mask[:, None], (feat_v - vision_mean) / vision_std, 0.0)
        feat_v_t = torch.tensor(norm_v, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 4. 音视频掩码张量构建
        mask_a_t = torch.tensor(a_mask.astype(np.float32)).unsqueeze(0).to(DEVICE)
        mask_v_t = torch.tensor(v_mask.astype(np.float32)).unsqueeze(0).to(DEVICE)

        # 5. 测试时数据增强 (TTA-MME) + 多模型贝叶斯共识
        views = generate_tta_views(feat_t, feat_a_t, feat_v_t, mask_t_t, mask_a_t, mask_v_t)
        
        all_view_probs = []
        all_view_regs = []
        all_view_gates = []
        
        with torch.no_grad():
            for (vt, va, vv, vmt, vma, vmv) in views:
                for m in models:
                    out = m(vt, va, vv, vmt, vma, vmv)
                    p = torch.softmax(out['cls_logits'], dim=-1).cpu().numpy()[0]
                    r = out['reg_output'].cpu().numpy()[0]
                    g = out['gates'].cpu().numpy()[0]
                    all_view_probs.append(p)
                    all_view_regs.append(r)
                    all_view_gates.append(g)

        # 贝叶斯后验均值与不确定度
        avg_prob = np.mean(all_view_probs, axis=0) # (3,)
        avg_reg = float(np.mean(all_view_regs))
        uncertainty = float(np.std(all_view_regs)) # 预测标准差
        avg_gates = np.mean(all_view_gates, axis=0)

        # 与最终测试及验证分析一致：平均分类概率 Argmax。
        pred_idx = int(np.argmax(avg_prob))
        pred_class = ["Negative", "Neutral", "Positive"][pred_idx]

        # 置信度 (基于分类概率与不确定度衰减)
        prob_conf = avg_prob[pred_idx]
        calibrated_conf = float(np.clip(prob_conf * (1.0 - 0.2 * uncertainty), 0.50, 0.99))

        missing_info = get_missing_description(mask_t_arr, a_mask, v_mask)

        print(f"Sample [{i:02d}/30] | 预测极性: {pred_class:<8} | 强度: {avg_reg:+.3f} (±{uncertainty:.3f}) | 置信度: {calibrated_conf*100:.1f}% | 门控 T/A/V: {avg_gates[0]:.2f}/{avg_gates[1]:.2f}/{avg_gates[2]:.2f}")

        results.append({
            '样本编号': f"test_{i:02d}",
            '特征文件名': fname,
            '情感极性预测 (Polarity)': pred_class,
            '情感强度预测 (Intensity)': round(avg_reg, 4),
            '预测不确定度 (Uncertainty)': round(uncertainty, 4),
            '预测置信度 (Confidence)': round(calibrated_conf, 4),
            '文本模态门控 (Text Gate)': round(float(avg_gates[0]), 3),
            '语音模态门控 (Audio Gate)': round(float(avg_gates[1]), 3),
            '视觉模态门控 (Vision Gate)': round(float(avg_gates[2]), 3),
            '模态局部缺失情况描述': missing_info
        })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    df.to_csv(OUTPUT_CSV2, index=False, encoding='utf-8-sig')
    print(f"\n=================================================================")
    print(f"所有 30 个缺失样本预测完成！结果已保存至:\n  -> {OUTPUT_CSV}\n  -> {OUTPUT_CSV2}")
    print(f"=================================================================")

if __name__ == "__main__":
    main()
