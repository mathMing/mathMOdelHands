"""本地问题1-3端到端执行入口。

默认不覆盖原始数据；问题1输出到 problem1_unaligned/features，问题2/3按现有脚本输出。
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

STEPS = [
    ("问题1：特征提取与对齐", ROOT / "problem1_unaligned" / "scripts" / "extract_and_align.py"),
    ("问题2：鲁棒模型训练", ROOT / "problem2_robust" / "train.py"),
    ("问题2：缺失模态消融", ROOT / "problem2_robust" / "ablation_experiments.py"),
    ("问题2：验证集分析", ROOT / "problem2_robust" / "evaluate_validation_and_error_analysis.py"),
    ("问题2：附件3预测", ROOT / "problem2_robust" / "predict_att3.py"),
    ("问题3：附件4预测与解释", ROOT / "problem3_interpretability" / "predict_att4_interpretable.py"),
    ("问题3：解释图表", ROOT / "problem3_interpretability" / "generate_explanation_cards.py"),
]


def validate_inputs():
    required = [
        ROOT / "problem1_unaligned" / "labels" / "label-100.xlsx",
        ROOT / "problem1_unaligned" / "raw_videos",
        ROOT / "DATA" / "Att2-Extracted-Features" / "aligned_50.pkl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("缺少必要输入：\n" + "\n".join(missing))


def run_step(name, script):
    print(f"\n{'=' * 80}\n{name}: {script}\n{'=' * 80}", flush=True)
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    result = subprocess.run([PYTHON, "-u", str(script)], cwd=str(script.parent), env=env)
    if result.returncode:
        raise SystemExit(f"{name}失败，退出码={result.returncode}")


if __name__ == "__main__":
    validate_inputs()
    for name, script in STEPS:
        run_step(name, script)
    print("\n问题1、问题2、问题3全部流程完成。")
