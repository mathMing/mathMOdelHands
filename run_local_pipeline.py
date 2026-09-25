import os
import sys
import subprocess
import time

PYTHON_EXE = r"D:\conda\envs\pytorch\python.exe"
BASE_DIR = r"X:\mathModelHands"

# 设置环境变量，强制 Python 子进程输出 UTF-8 与无缓冲流，彻底解决 Windows GBK 与输出阻塞
my_env = os.environ.copy()
my_env["PYTHONIOENCODING"] = "utf-8"
my_env["PYTHONUTF8"] = "1"
my_env["PYTHONUNBUFFERED"] = "1"

def run_step(step_name, script_rel_path):
    print(f"\n{'='*75}", flush=True)
    print(f"[{step_name}] 开始本地执行: {script_rel_path}", flush=True)
    print(f"{'='*75}", flush=True)
    t0 = time.time()
    
    script_full = os.path.join(BASE_DIR, script_rel_path)
    cwd = os.path.dirname(script_full)
    
    cmd = [PYTHON_EXE, "-u", os.path.basename(script_full)]
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                            text=True, bufsize=1, encoding='utf-8', errors='replace', env=my_env)
    
    for line in proc.stdout:
        print(line, end='', flush=True)
        
    proc.wait()
    elapsed = time.time() - t0
    if proc.returncode == 0:
        print(f"\n>>> [{step_name}] 成功完成！耗时: {elapsed:.1f} 秒\n")
    else:
        print(f"\n>>> [{step_name}] 出现异常！退出码: {proc.returncode}\n")
        sys.exit(proc.returncode)

if __name__ == "__main__":
    print("==========================================================================")
    print("=== 本机环境启动：RTX 3050 GPU 全流程模型升级 (Idea 1-3) 与端到端建模流水线 ===")
    print("==========================================================================")
    
    # 步骤 1: 训练鲁棒模型 (Idea 1: CAG门控 + Idea 2: Wasserstein序数损失 + Idea 3: BUDT动态阈值)
    run_step("步骤 1: 鲁棒多模态组合拳网络训练 (RTX 3050 本地加速)", r"problem2_robust\train.py")
    
    # 步骤 2: 运行消融实验与评估分析 (18组控制变量)
    run_step("步骤 2: 模态缺失消融实验评估 (18组控制变量)", r"problem2_robust\ablation_experiments.py")

    # 步骤 3: 运行验证集混淆矩阵与错误归因分析 (满足赛题第4条要求)
    run_step("步骤 3: 验证集性能评价、混淆矩阵与错误归因分析", r"problem2_robust\evaluate_validation_and_error_analysis.py")
    
    # 步骤 4: 附件3残缺专项测试集前向预测 (TTA-MME 多视角共识 + BUDT)
    run_step("步骤 4: 附件3残缺测试集前向预测 (TTA 多视角平滑)", r"problem2_robust\predict_att3.py")
    
    # 步骤 5: 附件4可解释性专项测试集预测与因果归因溯源
    run_step("步骤 5: 附件4可解释性预测与时空因果溯源", r"problem3_interpretability\predict_att4_interpretable.py")
    
    # 步骤 6: 生成论文级多模态解释卡片与图表
    run_step("步骤 6: 生成全套可解释性图表与解释卡片", r"problem3_interpretability\generate_explanation_cards.py")

    print("\n==========================================================================")
    print("=== 本地全部实验与推断执行完毕！所有指标、图表与预测 CSV 已完整更新 ===")
    print("==========================================================================")
