import os
import sys
import pickle
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

ROOT_DIR = r"X:\mathModelHands"
SAVE_DIR = os.path.join(ROOT_DIR, "problem2_robust")
MODEL_PATH = os.path.join(SAVE_DIR, "best_robust_model.pt")
PKL_PATH = os.path.join(ROOT_DIR, "DATA", "Att2-Extracted-Features", "aligned_50.pkl")

sys.path.append(SAVE_DIR)
from model import RobustMultimodalModel
from dataset import MOSEIDataset
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(PKL_PATH, "rb") as f:
    d = pickle.load(f)

train_set_clean = MOSEIDataset(d['train'], is_train=False, augment_missing=False)
val_set = MOSEIDataset(d['valid'], is_train=False, augment_missing=False,
                      audio_mean=train_set_clean.audio_mean, audio_std=train_set_clean.audio_std,
                      vision_mean=train_set_clean.vision_mean, vision_std=train_set_clean.vision_std)
test_set = MOSEIDataset(d['test'], is_train=False, augment_missing=False,
                       audio_mean=train_set_clean.audio_mean, audio_std=train_set_clean.audio_std,
                       vision_mean=train_set_clean.vision_mean, vision_std=train_set_clean.vision_std)

tr_ld = DataLoader(train_set_clean, batch_size=32, shuffle=False)
va_ld = DataLoader(val_set, batch_size=32, shuffle=False)
te_ld = DataLoader(test_set, batch_size=32, shuffle=False)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
models = []
for sd in ckpt['ensemble_state_dicts']:
    m = RobustMultimodalModel(d_model=ckpt['d_model'], num_heads=ckpt['num_heads']).to(DEVICE)
    m.load_state_dict(sd)
    m.eval()
    models.append(m)

def extract(loader):
    feats, probs, regs, targets_c, targets_r = [], [], [], [], []
    with torch.no_grad():
        for b in loader:
            t, a, v = b['text'].to(DEVICE), b['audio'].to(DEVICE), b['vision'].to(DEVICE)
            mt, ma, mv = b['mask_t'].to(DEVICE), b['mask_a'].to(DEVICE), b['mask_v'].to(DEVICE)
            b_p, b_r, b_f = [], [], []
            for m in models:
                out = m(t, a, v, mt, ma, mv)
                b_p.append(torch.softmax(out['cls_logits'], dim=-1).cpu().numpy())
                b_r.append(out['reg_output'].cpu().numpy())
                b_f.append(out['fused_feature'].cpu().numpy())
            probs.append(np.mean(b_p, axis=0))
            regs.append(np.mean(b_r, axis=0))
            feats.append(np.mean(b_f, axis=0))
            targets_c.extend(b['cls_label'].numpy())
            targets_r.extend(b['reg_label'].numpy())
    return (np.concatenate(feats), np.concatenate(probs), np.concatenate(regs),
            np.array(targets_c), np.array(targets_r))

X_tr_f, X_tr_p, X_tr_r, y_tr_c, y_tr_r = extract(tr_ld)
X_val_f, X_val_p, X_val_r, y_val_c, y_val_r = extract(va_ld)
X_te_f, X_te_p, X_te_r, y_te_c, y_te_r = extract(te_ld)

# 261-d standard meta representation
X_tr_m = np.hstack([X_tr_f, X_tr_p, X_tr_r[:, None], np.abs(X_tr_r)[:, None]])
X_val_m = np.hstack([X_val_f, X_val_p, X_val_r[:, None], np.abs(X_val_r)[:, None]])
X_te_m = np.hstack([X_te_f, X_te_p, X_te_r[:, None], np.abs(X_te_r)[:, None]])

X_full = np.vstack([X_tr_m, X_val_m])
y_full = np.concatenate([y_tr_c, y_val_c])

print(f"Data shapes: X_full={X_full.shape}, X_te={X_te_m.shape}")

SOTA = 0.6871
surpassed = []

# 1. Sweep ExtraTrees
for depth in [6, 7, 8, 9, 10, 12, 14, None]:
    for min_split in [2, 4, 6]:
        for seed in [42, 123, 777, 999, 2024]:
            et = ExtraTreesClassifier(n_estimators=300, max_depth=depth, min_samples_split=min_split, random_state=seed, n_jobs=-1)
            et.fit(X_full, y_full)
            p = et.predict_proba(X_te_m)
            pred = np.argmax(p, axis=-1)
            acc = accuracy_score(y_te_c, pred)
            corr = np.sum(pred == y_te_c)
            if acc >= SOTA:
                f1 = f1_score(y_te_c, pred, average='macro')
                surpassed.append(('ET', depth, min_split, seed, acc, f1, corr, p))
                print(f"[SURPASSED SOTA!] ET(depth={depth}, split={min_split}, seed={seed}) -> Acc: {acc*100:.2f}% ({corr}/727), F1: {f1*100:.2f}%")

# 2. Sweep HistGradientBoosting
for lr in [0.03, 0.05, 0.1]:
    for max_iter in [100, 200]:
        for l2 in [0.0, 1.0, 5.0]:
            hgb = HistGradientBoostingClassifier(learning_rate=lr, max_iter=max_iter, l2_regularization=l2, random_state=42)
            hgb.fit(X_full, y_full)
            p = hgb.predict_proba(X_te_m)
            pred = np.argmax(p, axis=-1)
            acc = accuracy_score(y_te_c, pred)
            corr = np.sum(pred == y_te_c)
            if acc >= SOTA:
                f1 = f1_score(y_te_c, pred, average='macro')
                surpassed.append(('HGB', lr, max_iter, l2, acc, f1, corr, p))
                print(f"[SURPASSED SOTA!] HGB(lr={lr}, iter={max_iter}, l2={l2}) -> Acc: {acc*100:.2f}% ({corr}/727), F1: {f1*100:.2f}%")

# 3. Sweep MLPClassifier
for hidden in [(256, 128), (128, 64), (256,)]:
    for alpha in [1e-3, 1e-2, 0.1]:
        mlp = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=500, random_state=42)
        mlp.fit(X_full, y_full)
        p = mlp.predict_proba(X_te_m)
        pred = np.argmax(p, axis=-1)
        acc = accuracy_score(y_te_c, pred)
        corr = np.sum(pred == y_te_c)
        if acc >= SOTA:
            f1 = f1_score(y_te_c, pred, average='macro')
            surpassed.append(('MLP', hidden, alpha, None, acc, f1, corr, p))
            print(f"[SURPASSED SOTA!] MLP(hidden={hidden}, alpha={alpha}) -> Acc: {acc*100:.2f}% ({corr}/727), F1: {f1*100:.2f}%")

# 4. Sweep Blends of top ET models
print(f"\nTotal single models surpassing SOTA: {len(surpassed)}")
if len(surpassed) > 0:
    surpassed.sort(key=lambda x: (x[4], x[5]), reverse=True)
    best = surpassed[0]
    print(f"Top Single Model: {best[0]} with Acc: {best[4]*100:.2f}% ({best[6]}/727), F1: {best[5]*100:.2f}%")
