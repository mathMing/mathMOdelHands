"""公平基线：固定数据划分、训练集标准化，测试集仅最终评估。"""
import os, pickle, json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from scipy.stats import pearsonr

ROOT = "/root/E_Problem" if os.path.exists("/root/E_Problem") else r"X:\\mathModelHands"
PKL = os.path.join(ROOT, "DATA", "Att2-Extracted-Features", "aligned_50.pkl")
OUT = os.path.join(ROOT, "problem2_robust", "fair_baseline_metrics.json")

def pool(x):
    # 与当前 mask-aware pooling 对齐：只平均非零时间步。
    m = (np.abs(x).sum(axis=-1) > 1e-5)
    return (x * m[..., None]).sum(axis=1) / np.maximum(m.sum(axis=1, keepdims=True), 1)

def metrics(y, pred, reg_y, reg_pred):
    r = pearsonr(reg_y, reg_pred)[0] if np.std(reg_pred) > 0 else 0.0
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)), "mae": float(mean_absolute_error(reg_y, reg_pred)), "pearson": float(r)}

with open(PKL, "rb") as f: data = pickle.load(f)
results = {}
for name, fields in {"text":["text"], "audio":["audio"], "vision":["vision"], "early_fusion":["text","audio","vision"]}.items():
    feats = {}
    for split in ["train", "valid", "test"]:
        parts = [pool(data[split][k].astype(np.float32)) for k in fields]
        feats[split] = np.concatenate(parts, axis=1)
    ytr, yv, yte = [data[s]["classification_labels"].reshape(-1) for s in ["train","valid","test"]]
    rtr, rv, rte = [data[s]["regression_labels"].reshape(-1) for s in ["train","valid","test"]]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    clf.fit(feats["train"], ytr); reg.fit(feats["train"], rtr)
    results[name] = metrics(yte, clf.predict(feats["test"]), rte, reg.predict(feats["test"]))
    print(name, results[name])
with open(OUT, "w", encoding="utf-8") as f: json.dump(results, f, ensure_ascii=False, indent=2)
print("saved", OUT)
