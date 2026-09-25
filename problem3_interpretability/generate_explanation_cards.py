import os
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================= 配置区域 =================
ROOT_DIR = "/root/E_Problem" if os.path.exists("/root/E_Problem") else r"X:\mathModelHands"
CSV_PATH = os.path.join(ROOT_DIR, "problem3_interpretability", "附件4_可解释专项测试集预测与解释结果.csv")
FIG_DIR = os.path.join(ROOT_DIR, "problem3_interpretability", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
# ==========================================

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(CSV_PATH)

def find_col(candidates):
    for c in df.columns:
        for cand in candidates:
            if cand in c:
                return c
    return None

col_id = find_col(['样本编号', 'sample_id', '编号'])
col_pol = find_col(['情感极性', '极性', 'Polarity'])
col_reg = find_col(['情感强度', '强度', 'Intensity'])
col_prim = find_col(['主要参考模态', '主导模态', 'Primary'])
col_text_pct = find_col(['文本模态贡献度', '文本作用程度', 'Text %'])
col_audio_pct = find_col(['语音模态贡献度', '语音作用程度', 'Audio %'])
col_vision_pct = find_col(['视觉模态贡献度', '视觉作用程度', 'Vision %'])
col_sec = find_col(['时间段', '秒', 'sec'])
col_frame = find_col(['视频帧', '帧区间', 'Frame'])
col_ev_text = find_col(['关键词', '证据片段', '词句', 'Text'])

# ================= 1. 三模态作用程度对比堆叠柱状图 =================
plt.figure(figsize=(12, 5.5), dpi=300)
x = np.arange(len(df))
width = 0.65

c_t = df[col_text_pct].values
c_a = df[col_audio_pct].values
c_v = df[col_vision_pct].values
labels = [f"#{r[col_id]}" for _, r in df.iterrows()]

p1 = plt.bar(x, c_t, width, label='文本模态贡献 (Text)', color='#1f77b4', alpha=0.9)
p2 = plt.bar(x, c_a, width, bottom=c_t, label='语音模态贡献 (Audio)', color='#ff7f0e', alpha=0.9)
p3 = plt.bar(x, c_v, width, bottom=c_t + c_a, label='视觉模态贡献 (Vision)', color='#2ca02c', alpha=0.9)

plt.xticks(x, labels, fontsize=10, rotation=35)
plt.ylabel('各模态归因贡献度占比 (%)', fontsize=12)
plt.title('附件 4 全部 20 条专项测试样本的三模态决策作用程度分布对比', fontsize=14, pad=12)
plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.ylim(0, 105)
plt.tight_layout()
fig1_path = os.path.join(FIG_DIR, "fig4_modality_contributions_all20.png")
plt.savefig(fig1_path)
plt.close()
print(f"图表已保存: {fig1_path}")

# ================= 2. 真实模型显著性曲线 (以 Sample 04 为例) =================
plt.figure(figsize=(9.5, 4.8), dpi=300)
sample_rows = df[df[col_id].astype(str).str.contains('04')]
if len(sample_rows) == 0 or '时间步显著性序列 (Saliency JSON)' not in df.columns:
    raise ValueError('缺少附件4真实显著性序列，请先重新运行 predict_att4_interpretable.py')
sample_row = sample_rows.iloc[0]
saliency = np.asarray(json.loads(sample_row['时间步显著性序列 (Saliency JSON)']), dtype=float)
steps = np.arange(1, len(saliency) + 1)

plt.plot(steps, saliency, color='#d62728', linewidth=2.5, marker='o', markersize=6, label='时序遮挡敏感性 S(t)')
step_text = str(sample_row['关键证据时序步长区间']).strip('[]')
step_start, step_end = [int(x.strip()) for x in step_text.split(',')]
plt.axvspan(step_start + 1, step_end + 1,
            color='#ff9896', alpha=0.4, label='模型定位的关键证据时段')
peak = int(np.argmax(saliency))
plt.annotate(f"模型显著性峰值\n{sample_row[col_ev_text]}",
             xy=(peak + 1, saliency[peak]), xytext=(max(peak + 3, 2), max(float(saliency.max()) * 0.8, 0.5)),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
             fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.6))

plt.title('典型正向样本 #04 模型反事实遮挡敏感性曲线', fontsize=13, pad=12)
plt.xlabel('对齐序列时间步 (Time Steps)', fontsize=11)
plt.ylabel('反事实重要性响应度', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()
fig2_path = os.path.join(FIG_DIR, "fig5_temporal_saliency_sample04.png")
plt.savefig(fig2_path)
plt.close()
print(f"图表已保存: {fig2_path}")

# ================= 3. 典型样本多模态解释卡片 (视觉汇报卡) =================
def draw_explanation_card(sample_idx_str, save_name):
    match_rows = df[df[col_id].astype(str).str.contains(str(sample_idx_str))]
    if len(match_rows) == 0:
        row = df.iloc[0]
    else:
        row = match_rows.iloc[0]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.axis('off')
    
    rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                         facecolor='#f8f9fa', edgecolor='#ced4da', linewidth=2, zorder=0)
    ax.add_patch(rect)
    
    pol_val = str(row[col_pol])
    header_color = '#28a745' if 'Pos' in pol_val else ('#dc3545' if 'Neg' in pol_val else '#6c757d')
    header_rect = plt.Rectangle((0.02, 0.85), 0.96, 0.13, transform=ax.transAxes,
                                facecolor=header_color, zorder=1)
    ax.add_patch(header_rect)
    
    plt.text(0.05, 0.91, f"多模态情感预测决策解释卡片 (Explanation Card) - 样本 #{row[col_id]}",
             fontsize=14, color='white', fontweight='bold', transform=ax.transAxes, va='center')
    
    content_y = 0.76
    plt.text(0.05, content_y, f"【1. 模型预测结论】", fontsize=12, fontweight='bold', color='#212529', transform=ax.transAxes)
    plt.text(0.08, content_y - 0.07, f"• 预测情感极性: {pol_val}", fontsize=11, transform=ax.transAxes)
    plt.text(0.08, content_y - 0.13, f"• 预测情感强度: {float(row[col_reg]):+.4f} (取值区间: [-3.0, 3.0])", fontsize=11, transform=ax.transAxes)
    
    content_y2 = 0.52
    plt.text(0.05, content_y2, f"【2. 宏观模态作用程度与决策归因】", fontsize=12, fontweight='bold', color='#212529', transform=ax.transAxes)
    plt.text(0.08, content_y2 - 0.07, f"• 主要参考模态: {row[col_prim]}", fontsize=11, fontweight='bold', color='#0056b3', transform=ax.transAxes)
    plt.text(0.08, content_y2 - 0.13, f"• 三模态贡献占比: 文本 (Text): {row[col_text_pct]}% | 语音 (Audio): {row[col_audio_pct]}% | 视觉 (Vision): {row[col_vision_pct]}%", fontsize=11, transform=ax.transAxes)
    
    content_y3 = 0.28
    plt.text(0.05, content_y3, f"【3. 微观时序关键证据定位与溯源】", fontsize=12, fontweight='bold', color='#212529', transform=ax.transAxes)
    plt.text(0.08, content_y3 - 0.07, f"• 关键时序秒数: {row[col_sec]}", fontsize=11, transform=ax.transAxes)
    plt.text(0.08, content_y3 - 0.13, f"• 关键视频帧位: {row[col_frame]}", fontsize=11, transform=ax.transAxes)
    plt.text(0.08, content_y3 - 0.19, f"• 触发文本证据: \"{row[col_ev_text]}\"", fontsize=11, fontweight='bold', color='#856404', transform=ax.transAxes)
    
    plt.tight_layout()
    card_path = os.path.join(FIG_DIR, save_name)
    plt.savefig(card_path)
    plt.close()
    print(f"解释卡片已保存: {card_path}")

draw_explanation_card("04", "fig6_explanation_card_sample04_positive.png")
draw_explanation_card("02", "fig7_explanation_card_sample02_negative.png")

print("\n全套可解释性图表生成完毕！")
