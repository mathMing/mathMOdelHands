import os
import glob
import pickle
import torch
import numpy as np
import pandas as pd
import json
import cv2

import sys
import numpy.core
sys.modules['numpy._core'] = numpy.core
sys.modules['numpy._core.numeric'] = numpy.core.numeric
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'problem2_robust'))
from model import RobustMultimodalModel

# ================= 配置区域 =================
ROOT_DIR = "/root/E_Problem" if os.path.exists("/root/E_Problem") else r"X:\mathModelHands"

def find_att4_aligned_dir(root_dir):
    data_dir = os.path.join(root_dir, "DATA")
    for root, dirs, files in os.walk(data_dir):
        pkls = [f for f in files if f.endswith('.pkl')]
        if len(pkls) == 20:
            try:
                with open(os.path.join(root, pkls[0]), 'rb') as f:
                    sample = pickle.load(f)
                    if sample['vision'].shape[0] == 50:
                        return root
            except:
                pass
    return os.path.join(data_dir, "Att4-Interpretable-Video-Samples-Features", "附件4-可解释专项视频样本与特征文件", "对齐版本")

def find_att4_video_dir(root_dir):
    data_dir = os.path.join(root_dir, "DATA")
    for root, dirs, files in os.walk(data_dir):
        mp4s = [f for f in files if f.endswith('.mp4')]
        if len(mp4s) == 20:
            return root
    return os.path.join(data_dir, "Att4-Interpretable-Video-Samples-Features", "附件4-可解释专项视频样本与特征文件", "4-原始视频")

ATT4_DIR = find_att4_aligned_dir(ROOT_DIR)
VIDEO_DIR = find_att4_video_dir(ROOT_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "problem2_robust", "best_robust_model.pt")
OUTPUT_DIR = os.path.join(ROOT_DIR, "problem3_interpretability")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "附件4_可解释专项测试集预测与解释结果.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================

def get_word_segment(raw_text, t_start, t_end, eff_len=50):
    """将时序步长 [t_start, t_end] 映射到原始文本的相应词段"""
    words = raw_text.strip().split()
    if len(words) == 0:
        return raw_text
    num_words = len(words)
    w_start = int((t_start / max(1, eff_len)) * num_words)
    w_end = int(((t_end + 1) / max(1, eff_len)) * num_words)
    w_start = max(0, min(w_start, num_words - 1))
    w_end = max(w_start + 1, min(w_end, num_words))
    return " ".join(words[w_start:w_end])

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"当前运行设备: {DEVICE}")
    print(f"识别到的附件4对齐目录: {ATT4_DIR}")
    print(f"正在加载最优权重: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
    d_model = checkpoint.get('d_model', 256)
    num_heads = checkpoint.get('num_heads', 4)

    # 载入多模型集成 (3-Seed Ensemble)
    if 'ensemble_state_dicts' in checkpoint:
        state_dicts = checkpoint['ensemble_state_dicts']
        print(f"成功载入 {len(state_dicts)} 个独立种子的模型参数构建可解释性平滑集成！")
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

    files = [os.path.join(ATT4_DIR, f"{i:02d}.pkl") for i in range(1, 21)]
    print(f"待处理附件 4 样本数: {len(files)}")

    polarity_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    results = []

    with torch.no_grad():
        for i, pkl_file in enumerate(files, 1):
            sample_id = f"{i:02d}"
            video_file = os.path.join(VIDEO_DIR, f"{sample_id}.mp4")

            # 1. 获取视频元数据
            cap = cv2.VideoCapture(video_file)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps if fps > 0 else 0.0
            cap.release()

            # 2. 读取特征
            with open(pkl_file, 'rb') as f:
                d = pickle.load(f)

            raw_text = str(d.get('raw_text', ''))
            t_feat = d['text'].astype(np.float32)[:50]   # (50, 768)
            a_raw = d['audio'].astype(np.float32)[:50]  # (50, 74)
            v_raw = d['vision'].astype(np.float32)[:50] # (50, 35)

            # 标准化
            a_norm = np.where(np.abs(a_raw) > 1e-5, (a_raw - audio_mean) / audio_std, 0.0)
            v_norm = np.where(np.abs(v_raw) > 1e-5, (v_raw - vision_mean) / vision_std, 0.0)

            t_ten = torch.tensor(t_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            a_ten = torch.tensor(a_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            v_ten = torch.tensor(v_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            mt = (torch.abs(t_ten).sum(dim=-1) > 1e-5).float()
            ma = (torch.abs(a_ten).sum(dim=-1) > 1e-5).float()
            mv = (torch.abs(v_ten).sum(dim=-1) > 1e-5).float()

            eff_len = int(mt.sum().item())
            if eff_len == 0: eff_len = 50

            # 3. 基准全模态前向预测 (集成平均)
            base_preds = []
            base_cls_probs = []
            for m in models:
                out = m(t_ten, a_ten, v_ten, mt, ma, mv)
                base_preds.append(float(out['reg_output'].item()))
                base_cls_probs.append(torch.softmax(out['cls_logits'], dim=-1).cpu().numpy()[0])
                
            base_pred_intensity = float(np.mean(base_preds))
            avg_cls_prob = np.mean(base_cls_probs, axis=0)
            base_cls_idx = int(np.argmax(avg_cls_prob))
            base_polarity = polarity_map[base_cls_idx]

            # 4. 因果反事实干预：计算三模态独立贡献程度
            # 遮挡文本
            delta_t_list = []
            for m in models:
                out_no_t = m(torch.zeros_like(t_ten), a_ten, v_ten, torch.zeros_like(mt), ma, mv)
                delta_t_list.append(abs(base_pred_intensity - float(out_no_t['reg_output'].item())))
            delta_t = float(np.mean(delta_t_list))

            # 遮挡音频
            delta_a_list = []
            for m in models:
                out_no_a = m(t_ten, torch.zeros_like(a_ten), v_ten, mt, torch.zeros_like(ma), mv)
                delta_a_list.append(abs(base_pred_intensity - float(out_no_a['reg_output'].item())))
            delta_a = float(np.mean(delta_a_list))

            # 遮挡视觉
            delta_v_list = []
            for m in models:
                out_no_v = m(t_ten, a_ten, torch.zeros_like(v_ten), mt, ma, torch.zeros_like(mv))
                delta_v_list.append(abs(base_pred_intensity - float(out_no_v['reg_output'].item())))
            delta_v = float(np.mean(delta_v_list))

            total_delta = delta_t + delta_a + delta_v + 1e-6
            c_t = (delta_t / total_delta) * 100
            c_a = (delta_a / total_delta) * 100
            c_v = (delta_v / total_delta) * 100

            # 判定主要参考模态
            if c_t >= c_a and c_t >= c_v:
                primary_modality = "文本模态 (Text)"
                primary_key = 't'
            elif c_a >= c_t and c_a >= c_v:
                primary_modality = "语音模态 (Audio)"
                primary_key = 'a'
            else:
                primary_modality = "视觉模态 (Vision)"
                primary_key = 'v'

            # 5. 微观时序显著性与关键证据定位 (Temporal Saliency Localization)
            saliency = []
            for step in range(eff_len):
                step_impacts = []
                for m in models:
                    if primary_key == 't':
                        t_mod = t_ten.clone()
                        t_mod[:, step, :] = 0.0
                        mt_mod = mt.clone()
                        mt_mod[:, step] = 0.0
                        step_out = m(t_mod, a_ten, v_ten, mt_mod, ma, mv)
                    elif primary_key == 'a':
                        a_mod = a_ten.clone()
                        a_mod[:, step, :] = 0.0
                        ma_mod = ma.clone()
                        ma_mod[:, step] = 0.0
                        step_out = m(t_ten, a_mod, v_ten, mt, ma_mod, mv)
                    else:
                        v_mod = v_ten.clone()
                        v_mod[:, step, :] = 0.0
                        mv_mod = mv.clone()
                        mv_mod[:, step] = 0.0
                        step_out = m(t_ten, a_ten, v_mod, mt, ma, mv_mod)
                    step_impacts.append(abs(base_pred_intensity - float(step_out['reg_output'].item())))
                saliency.append(float(np.mean(step_impacts)))

            saliency = np.array(saliency)
            if len(saliency) > 0 and saliency.max() > 0:
                saliency_norm = saliency / saliency.max()
            else:
                saliency_norm = np.ones(eff_len)

            # 提取贡献最显著的连续时序窗口 (Top Window)
            window_size = min(3, eff_len)
            rolling_sums = np.convolve(saliency_norm, np.ones(window_size), mode='valid')
            top_step_start = int(np.argmax(rolling_sums))
            top_step_end = top_step_start + window_size - 1

            # 对应回原始素材
            step_duration = duration_sec / 50.0 if duration_sec > 0 else 0.1
            t_sec_start = round(top_step_start * step_duration, 2)
            t_sec_end = round((top_step_end + 1) * step_duration, 2)

            frame_start = int(top_step_start * step_duration * fps)
            frame_end = min(total_frames, int((top_step_end + 1) * step_duration * fps))

            evidence_text = get_word_segment(raw_text, top_step_start, top_step_end, eff_len)

            print(f"Sample {sample_id} | 极性: {base_polarity:<8} | 强度: {base_pred_intensity:+.3f} | 主导模态: {primary_modality} (T:{c_t:.1f}% A:{c_a:.1f}% V:{c_v:.1f}%) | 显著时段: {t_sec_start}s~{t_sec_end}s (帧: {frame_start}~{frame_end})")

            results.append({
                '样本编号': f"test_{sample_id}",
                '视频文件名': f"{sample_id}.mp4",
                '原始转写文本 (Raw Text)': raw_text,
                '情感极性预测 (Polarity)': base_polarity,
                '情感强度预测 (Intensity)': round(base_pred_intensity, 4),
                '主要参考模态 (Primary Modality)': primary_modality,
                '文本模态贡献度 (Text %)': round(float(c_t), 2),
                '语音模态贡献度 (Audio %)': round(float(c_a), 2),
                '视觉模态贡献度 (Vision %)': round(float(c_v), 2),
                '关键证据时序步长区间': f"[{top_step_start}, {top_step_end}]",
                '时间步显著性序列 (Saliency JSON)': json.dumps([round(float(x), 6) for x in saliency_norm.tolist()]),
                '语音/视频关键时间段 (秒)': f"{t_sec_start}s - {t_sec_end}s",
                '视觉关键视频帧区间 (Frame)': f"{frame_start} - {frame_end}帧",
                '文本核心情感关键词句': evidence_text
            })

    # 导出 CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n=================================================================")
    print(f"问题 3 可解释性专项全量 20 样本分析完成！已导出至:\n  -> {OUTPUT_CSV}")
    print(f"=================================================================")

    # 汇总宏观统计量
    p3_metrics = {
        'task': '问题三：可解释性多模态情感预测建模与验证 (3-Seed平滑遮挡敏感性分析)',
        'total_test_samples': len(results),
        'primary_modality_distribution': {
            '文本模态 (Text)': int((df['主要参考模态 (Primary Modality)'] == '文本模态 (Text)').sum()),
            '视觉模态 (Vision)': int((df['主要参考模态 (Primary Modality)'] == '视觉模态 (Vision)').sum()),
            '语音模态 (Audio)': int((df['主要参考模态 (Primary Modality)'] == '语音模态 (Audio)').sum())
        },
        'average_modality_contributions_pct': {
            'text_avg_pct': round(float(df['文本模态贡献度 (Text %)'].mean()), 2),
            'audio_avg_pct': round(float(df['语音模态贡献度 (Audio %)'].mean()), 2),
            'vision_avg_pct': round(float(df['视觉模态贡献度 (Vision %)'].mean()), 2)
        },
        'polarity_distribution': {
            'Negative': int((df['情感极性预测 (Polarity)'] == 'Negative').sum()),
            'Positive': int((df['情感极性预测 (Polarity)'] == 'Positive').sum()),
            'Neutral': int((df['情感极性预测 (Polarity)'] == 'Neutral').sum())
        },
        'intensity_stats': {
            'mean': round(float(df['情感强度预测 (Intensity)'].mean()), 4),
            'min': round(float(df['情感强度预测 (Intensity)'].min()), 4),
            'max': round(float(df['情感强度预测 (Intensity)'].max()), 4)
        }
    }
    with open(os.path.join(OUTPUT_DIR, "problem3_key_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(p3_metrics, f, ensure_ascii=False, indent=4)

    df_p3 = pd.DataFrame([
        {
            '分析指标': '主导模态样本分布 (Text / Vision / Audio)',
            '统计结果': f"{p3_metrics['primary_modality_distribution']['文本模态 (Text)']} / {p3_metrics['primary_modality_distribution']['视觉模态 (Vision)']} / {p3_metrics['primary_modality_distribution']['语音模态 (Audio)']}"
        },
        {
            '分析指标': '全样本三模态平均贡献率 (Text% / Vision% / Audio%)',
            '统计结果': f"{p3_metrics['average_modality_contributions_pct']['text_avg_pct']}% / {p3_metrics['average_modality_contributions_pct']['vision_avg_pct']}% / {p3_metrics['average_modality_contributions_pct']['audio_avg_pct']}%"
        },
        {
            '分析指标': '情感极性分布 (Negative / Positive / Neutral)',
            '统计结果': f"{p3_metrics['polarity_distribution']['Negative']} / {p3_metrics['polarity_distribution']['Positive']} / {p3_metrics['polarity_distribution']['Neutral']}"
        },
        {
            '分析指标': '预测情感强度均值与极值区间',
            '统计结果': f"均值 {p3_metrics['intensity_stats']['mean']}, 区间 [{p3_metrics['intensity_stats']['min']}, {p3_metrics['intensity_stats']['max']}]"
        }
    ])
    df_p3.to_csv(os.path.join(OUTPUT_DIR, "problem3_key_metrics.csv"), index=False, encoding='utf-8-sig')
    print("问题3关键指标已同步更新至 problem3_key_metrics.json 和 problem3_key_metrics.csv！")

if __name__ == "__main__":
    main()
