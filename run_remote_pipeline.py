import os
import sys
import time
import paramiko

hostname = "connect.bjb2.seetacloud.com"
port = 33757
username = "root"
password = "Fpd1YRIp1G35"

local_base = r"X:\mathModelHands"
remote_base = "/root/E_Problem"

print("=================================================================")
print("=== 正在连接远程 5090 服务器并启动全套模型训练与消融实验流水线 ===")
print("=================================================================")

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname=hostname, port=port, username=username, password=password, timeout=30)

# 步骤 0: 同步最新工程代码至远程
print("\n>>> 同步最新工程代码至远程 5090 服务器...")
sftp = ssh.open_sftp()
scripts_to_sync = [
    ("problem2_robust/model.py", "/root/E_Problem/problem2_robust/model.py"),
    ("problem2_robust/train.py", "/root/E_Problem/problem2_robust/train.py"),
    ("problem2_robust/ablation_experiments.py", "/root/E_Problem/problem2_robust/ablation_experiments.py"),
    ("problem2_robust/evaluate_validation_and_error_analysis.py", "/root/E_Problem/problem2_robust/evaluate_validation_and_error_analysis.py"),
    ("problem2_robust/predict_att3.py", "/root/E_Problem/problem2_robust/predict_att3.py"),
    ("problem3_interpretability/predict_att4_interpretable.py", "/root/E_Problem/problem3_interpretability/predict_att4_interpretable.py"),
    ("problem3_interpretability/generate_explanation_cards.py", "/root/E_Problem/problem3_interpretability/generate_explanation_cards.py"),
]
for loc_rel, rem_path in scripts_to_sync:
    loc_path = os.path.join(local_base, loc_rel)
    print(f"  正在同步: {loc_rel} -> {rem_path}")
    sftp.put(loc_path, rem_path)
sftp.close()
print(">>> 所有升级代码同步就绪！\n")

def execute_remote_cmd(step_name, cmd):
    print(f"\n{'='*70}")
    print(f"[{step_name}] 开始执行: {cmd}")
    print(f"{'='*70}")
    t0 = time.time()
    
    full_cmd = f"bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && export HF_ENDPOINT=https://hf-mirror.com && cd {remote_base} && {cmd}'"
    stdin, stdout, stderr = ssh.exec_command(full_cmd, get_pty=True)
    
    for line in iter(stdout.readline, ""):
        print(line, end="", flush=True)
        
    code = stdout.channel.recv_exit_status()
    elapsed = time.time() - t0
    if code == 0:
        print(f"[{step_name}] 成功完成！耗时: {elapsed:.1f} 秒\n")
    else:
        print(f"[{step_name}] 发生错误！退出代码: {code}\n")
        raise RuntimeError(f"Step {step_name} failed with code {code}")

try:
    # 步骤 1: 训练鲁棒多模态组合拳网络 (5-Seed 集成 + 极性冲突感知门控 + Wasserstein损失)
    execute_remote_cmd("步骤 1: 鲁棒多模态网络 5-Seed 训练与 BUDT 校准 (5090 GPU)", "python problem2_robust/train.py")
    
    # 步骤 2: 运行消融实验与评估分析
    execute_remote_cmd("步骤 2: 模态缺失消融实验评估 (18组控制变量)", "python problem2_robust/ablation_experiments.py")

    # 步骤 3: 运行验证集混淆矩阵与错误归因分析
    execute_remote_cmd("步骤 3: 验证集性能评价、混淆矩阵与错误归因分析", "python problem2_robust/evaluate_validation_and_error_analysis.py")
    
    # 步骤 4: 附件3残缺专项测试集前向预测 (TTA-MME 多视角共识 + BUDT)
    execute_remote_cmd("步骤 4: 附件3残缺测试集前向预测 (TTA 多视角平滑)", "python problem2_robust/predict_att3.py")
    
    # 步骤 5: 附件4可解释性专项测试集预测与归因溯源 (5-Seed 集成反事实干预)
    execute_remote_cmd("步骤 5: 附件4可解释性预测与时空因果溯源", "python problem3_interpretability/predict_att4_interpretable.py")
    
    # 步骤 6: 生成论文级多模态解释卡片与图表
    execute_remote_cmd("步骤 6: 生成全套可解释性图表与解释卡片", "python problem3_interpretability/generate_explanation_cards.py")

    print("\n=================================================================")
    print("=== 所有远程实验与推断执行完毕！正在将成果同步回本地工作空间 ===")
    print("=================================================================")
    
    sftp = ssh.open_sftp()
    
    # 需要拉取回本地的文件清单
    files_to_download = [
        # 问题2产出
        ("problem2_robust/best_robust_model.pt", r"X:\mathModelHands\problem2_robust\best_robust_model.pt"),
        ("problem2_robust/problem2_key_metrics.json", r"X:\mathModelHands\problem2_robust\problem2_key_metrics.json"),
        ("problem2_robust/problem2_key_metrics.csv", r"X:\mathModelHands\problem2_robust\problem2_key_metrics.csv"),
        ("problem2_robust/ablation_experiments_summary.csv", r"X:\mathModelHands\problem2_robust\ablation_experiments_summary.csv"),
        ("problem2_robust/validation_error_attribution_summary.json", r"X:\mathModelHands\problem2_robust\validation_error_attribution_summary.json"),
        ("problem2_robust/附件3_模态局部缺失专项测试集预测结果.csv", r"X:\mathModelHands\problem2_robust\附件3_模态局部缺失专项测试集预测结果.csv"),
        ("problem2_robust/附件3_多模态部分缺失专项测试集预测结果.csv", r"X:\mathModelHands\problem2_robust\附件3_多模态部分缺失专项测试集预测结果.csv"),
        ("problem2_robust/figures/fig1_missing_modality_type.png", r"X:\mathModelHands\problem2_robust\figures\fig1_missing_modality_type.png"),
        ("problem2_robust/figures/fig2_missing_position.png", r"X:\mathModelHands\problem2_robust\figures\fig2_missing_position.png"),
        ("problem2_robust/figures/fig3_missing_duration_decay.png", r"X:\mathModelHands\problem2_robust\figures\fig3_missing_duration_decay.png"),
        ("problem2_robust/figures/fig8_validation_confusion_matrix.png", r"X:\mathModelHands\problem2_robust\figures\fig8_validation_confusion_matrix.png"),
        ("problem2_robust/figures/fig9_validation_regression_residuals.png", r"X:\mathModelHands\problem2_robust\figures\fig9_validation_regression_residuals.png"),
        ("problem2_robust/figures/fig10_global_modality_gate_distribution.png", r"X:\mathModelHands\problem2_robust\figures\fig10_global_modality_gate_distribution.png"),
        
        # 问题3产出
        ("problem3_interpretability/problem3_key_metrics.json", r"X:\mathModelHands\problem3_interpretability\problem3_key_metrics.json"),
        ("problem3_interpretability/problem3_key_metrics.csv", r"X:\mathModelHands\problem3_interpretability\problem3_key_metrics.csv"),
        ("problem3_interpretability/附件4_可解释专项测试集预测与解释结果.csv", r"X:\mathModelHands\problem3_interpretability\附件4_可解释专项测试集预测与解释结果.csv"),
        ("problem3_interpretability/figures/fig4_modality_contributions_all20.png", r"X:\mathModelHands\problem3_interpretability\figures\fig4_modality_contributions_all20.png"),
        ("problem3_interpretability/figures/fig5_temporal_saliency_sample04.png", r"X:\mathModelHands\problem3_interpretability\figures\fig5_temporal_saliency_sample04.png"),
        ("problem3_interpretability/figures/fig6_explanation_card_sample04_positive.png", r"X:\mathModelHands\problem3_interpretability\figures\fig6_explanation_card_sample04_positive.png"),
        ("problem3_interpretability/figures/fig7_explanation_card_sample02_negative.png", r"X:\mathModelHands\problem3_interpretability\figures\fig7_explanation_card_sample02_negative.png"),
    ]
    
    for rem_rel, loc_path in files_to_download:
        rem_path = f"{remote_base}/{rem_rel}"
        os.makedirs(os.path.dirname(loc_path), exist_ok=True)
        try:
            print(f"正在下载: {rem_path} -> {loc_path}...")
            sftp.get(rem_path, loc_path)
            print(f"  √ 已下载: {os.path.basename(loc_path)} ({os.path.getsize(loc_path)/1024:.1f} KB)")
        except Exception as e:
            print(f"  × 下载跳过或异常 ({rem_path}): {e}")
            
    sftp.close()
    print("\n=================================================================")
    print("=== 全流程执行与成果同步完毕！本地所有指标、图表与预测文件已更新 ===")
    print("=================================================================")

finally:
    ssh.close()
