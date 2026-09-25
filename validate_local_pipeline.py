"""只读检查问题1-3输入输出，避免在未完成实验时误报最终结果。"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)


def check_csv(path, minimum_rows):
    require(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) - 1 < minimum_rows:
        raise ValueError(f"{path} 行数不足：{len(rows)-1} < {minimum_rows}")
    return len(rows) - 1


def main():
    p1 = ROOT / "problem1_unaligned" / "features"
    require(p1 / "aligned_100.pkl")
    n1 = check_csv(p1 / "problem1_feature_summary_100.csv", 100)
    n3a = check_csv(ROOT / "problem2_robust" / "附件3_模态局部缺失专项测试集预测结果.csv", 30)
    n3b = check_csv(ROOT / "problem2_robust" / "附件3_多模态部分缺失专项测试集预测结果.csv", 30)
    n4 = check_csv(ROOT / "problem3_interpretability" / "附件4_可解释专项测试集预测与解释结果.csv", 20)
    for metrics in [ROOT / "problem2_robust" / "problem2_key_metrics.json", ROOT / "problem3_interpretability" / "problem3_key_metrics.json"]:
        require(metrics)
        with metrics.open("r", encoding="utf-8") as f:
            json.load(f)
    print(f"问题1={n1}，附件3结果={n3a}/{n3b}，附件4结果={n4}；JSON可解析。")


if __name__ == "__main__":
    main()
