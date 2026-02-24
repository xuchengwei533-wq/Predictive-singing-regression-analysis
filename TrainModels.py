import os
import re
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    from mord import LogisticAT
except Exception:
    LogisticAT = None

data_dir = r"D:\同济车辆\chest_audio\Chest new0206\Chest new0206"
output_dir = r"D:\同济车辆\chest_audio\Chest new0206"
os.makedirs(output_dir, exist_ok=True)

jitter_dir = os.path.join(data_dir, "JitterOutput")
shimmer_dir = os.path.join(data_dir, "ShimmerOutput")
rms_dir = os.path.join(data_dir, "RMS_Output")
centroid_dir = os.path.join(data_dir, "Spectral_Centroid_Output")
h1h2_dir = os.path.join(data_dir, "H1H2Output")
spectral_slope_dir = os.path.join(data_dir, "SpectralSlopeOutput")
mfcc3_dir = os.path.join(data_dir, "Mfcc3Output")

def parse_label(filename):
    name = os.path.splitext(filename)[0]
    nums = re.findall(r"\d+", name) 
    if not nums:
        return None
    return int(nums[-1])

def load_series_csv(path):
    try:
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
        return np.array(data, dtype=np.float32)
    except Exception:
        return None

def series_stats(x):
    if x is None:
        return None
    x = np.asarray(x, dtype=np.float32).flatten()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return np.array([
        float(np.mean(x)),
        float(np.std(x)),
        float(np.median(x)),
        float(np.min(x)),
        float(np.max(x)),
        float(np.percentile(x, 25)),
        float(np.percentile(x, 75))
    ], dtype=np.float32)

def stats_mean_std(x):
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return np.array([float(np.mean(arr)), float(np.std(arr))], dtype=np.float32)
    stats = []
    for i in range(arr.shape[1]):
        col = arr[:, i]
        col = col[np.isfinite(col)]
        if col.size == 0:
            return None
        stats.extend([float(np.mean(col)), float(np.std(col))])
    return np.array(stats, dtype=np.float32)

def stats_mean_max(x):
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return np.array([float(np.mean(arr)), float(np.max(arr))], dtype=np.float32)
    stats = []
    for i in range(arr.shape[1]):
        col = arr[:, i]
        col = col[np.isfinite(col)]
        if col.size == 0:
            return None
        stats.extend([float(np.mean(col)), float(np.max(col))])
    return np.array(stats, dtype=np.float32)

group_defs = {
    "Jitter": {"dir": jitter_dir, "extractor": stats_mean_std, "features": ["Jitter_mean", "Jitter_std"]},
    "Shimmer": {"dir": shimmer_dir, "extractor": stats_mean_std, "features": ["Shimmer_mean", "Shimmer_std"]},
    "RMS": {"dir": rms_dir, "extractor": stats_mean_max, "features": ["RMS_mean", "RMS_max"]},
    "Centroid": {"dir": centroid_dir, "extractor": stats_mean_std, "features": ["Centroid_mean", "Centroid_std"]},
    "H1H2": {"dir": h1h2_dir, "extractor": stats_mean_std, "features": ["H1H2_mean", "H1H2_std"]},
    "SpectralSlope": {"dir": spectral_slope_dir, "extractor": stats_mean_std, "features": ["SpectralSlope_mean", "SpectralSlope_std"]},
    "Mfcc3": {"dir": mfcc3_dir, "extractor": stats_mean_std, "features": ["Mfcc1_mean", "Mfcc1_std", "Mfcc2_mean", "Mfcc2_std", "Mfcc3_mean", "Mfcc3_std"]}
}

def available_groups():
    groups = []
    for name, info in group_defs.items():
        if os.path.isdir(info["dir"]):
            groups.append(name)
    return groups

def build_dataset(group_names):
    if len(group_names) == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int32), [], []
    for name in group_names:
        info = group_defs[name]
        if not os.path.isdir(info["dir"]):
            raise FileNotFoundError(f"特征目录不存在: {info['dir']}")
    base_dir = group_defs[group_names[0]]["dir"]
    files = [f for f in os.listdir(base_dir) if f.lower().endswith(".csv")]
    files.sort()
    X = []
    y = []
    kept = []
    for f in files:
        base = os.path.splitext(f)[0]
        label = parse_label(base)
        if label is None:
            continue
        row_feats = []
        missing = False
        for name in group_names:
            info = group_defs[name]
            path = os.path.join(info["dir"], base + ".csv")
            if not os.path.exists(path):
                missing = True
                break
            series = load_series_csv(path)
            feats = info["extractor"](series)
            if feats is None:
                missing = True
                break
            row_feats.append(feats)
        if missing:
            continue
        feat = np.concatenate(row_feats, axis=0)
        X.append(feat)
        y.append(label)
        kept.append(base)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    feature_names = []
    for name in group_names:
        feature_names.extend(group_defs[name]["features"])
    return X, y, kept, feature_names

if LogisticAT is None:
    raise RuntimeError("未检测到 mord，请先安装：pip install mord")

group_list = available_groups()
if len(group_list) < 2:
    raise RuntimeError("可用特征组不足，请先生成对应特征文件夹")

print("📦 可用特征组:", ", ".join(group_list))
print("📦 正在评估特征组合...")

results = []
best_combo = None
best_cv_acc = -1.0
best_holdout_acc = -1.0

for r in range(2, len(group_list) + 1):
    for combo in itertools.combinations(group_list, r):
        X, y, kept, feature_names = build_dataset(list(combo))
        if len(y) == 0:
            continue
        label_values = sorted(list(set(y.tolist())))
        label_map = {v: i for i, v in enumerate(label_values)}
        inv_label_map = {i: v for v, i in label_map.items()}
        y_idx = np.asarray([label_map[v] for v in y], dtype=np.int32)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_idx, test_size=0.3, random_state=42, stratify=y_idx
        )
        ord_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticAT(alpha=1.0))
        ])
        ord_model.fit(X_train, y_train)
        y_pred_idx = ord_model.predict(X_test)
        y_test_labels = np.asarray([inv_label_map[i] for i in y_test], dtype=np.int32)
        y_pred_labels = np.asarray([inv_label_map[i] for i in y_pred_idx], dtype=np.int32)
        acc_ord = accuracy_score(y_test_labels, y_pred_labels)
        f1_ord = f1_score(y_test_labels, y_pred_labels, average="macro")
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_acc = cross_val_score(ord_model, X, y_idx, cv=skf, scoring="accuracy")
        cv_f1 = cross_val_score(ord_model, X, y_idx, cv=skf, scoring="f1_macro")
        results.append({
            "特征组合": "+".join(combo),
            "特征维度": int(X.shape[1]),
            "样本数": int(len(y)),
            "准确率": round(float(acc_ord), 4),
            "宏平均F1": round(float(f1_ord), 4),
            "CV准确率均值": round(float(cv_acc.mean()), 4),
            "CV准确率标准差": round(float(cv_acc.std()), 4),
            "CV宏F1均值": round(float(cv_f1.mean()), 4),
            "CV宏F1标准差": round(float(cv_f1.std()), 4)
        })
        if float(cv_acc.mean()) > best_cv_acc or (float(cv_acc.mean()) == best_cv_acc and float(acc_ord) > best_holdout_acc):
            best_cv_acc = float(cv_acc.mean())
            best_holdout_acc = float(acc_ord)
            best_combo = list(combo)

if len(results) == 0:
    raise RuntimeError("没有可评估的特征组合，请确认特征文件已生成且文件名匹配")

result_df = pd.DataFrame(results).sort_values(by=["CV准确率均值", "准确率"], ascending=False)
combo_path = os.path.join(output_dir, "特征组合对比结果.csv")
result_df.to_csv(combo_path, index=False)
print(f"✅ 特征组合对比结果已保存：{combo_path}")

print("\n" + "-" * 60)
print("📊 最优特征组合：", "+".join(best_combo))

X, y, kept, feature_names = build_dataset(best_combo)
print(f"✅ 样本数: {len(y)} | 特征维度: {X.shape[1] if len(y) > 0 else 0}")

label_values = sorted(list(set(y.tolist())))
label_map = {v: i for i, v in enumerate(label_values)}
inv_label_map = {i: v for v, i in label_map.items()}
y_idx = np.asarray([label_map[v] for v in y], dtype=np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_idx, test_size=0.3, random_state=42, stratify=y_idx
)

ord_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticAT(alpha=1.0))
])
ord_model.fit(X_train, y_train)
y_pred_idx = ord_model.predict(X_test)
y_test_labels = np.asarray([inv_label_map[i] for i in y_test], dtype=np.int32)
y_pred_labels = np.asarray([inv_label_map[i] for i in y_pred_idx], dtype=np.int32)
acc_ord = accuracy_score(y_test_labels, y_pred_labels)
f1_ord = f1_score(y_test_labels, y_pred_labels, average="macro")
print(f"✅ 序数回归完成！准确率: {acc_ord:.4f}, 宏F1: {f1_ord:.4f}")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_acc = cross_val_score(ord_model, X, y_idx, cv=skf, scoring="accuracy")
cv_f1 = cross_val_score(ord_model, X, y_idx, cv=skf, scoring="f1_macro")
cv_df = pd.DataFrame({
    "指标": ["accuracy", "macro_f1"],
    "均值": [float(cv_acc.mean()), float(cv_f1.mean())],
    "标准差": [float(cv_acc.std()), float(cv_f1.std())]
})
cv_path = os.path.join(output_dir, "分层10折交叉验证结果.csv")
cv_df.to_csv(cv_path, index=False)
print(f"✅ 分层10折交叉验证已保存：{cv_path}")

results_best = [{"模型": "序数回归(LogisticAT)", "准确率": round(acc_ord, 4), "宏平均F1": round(f1_ord, 4)}]
result_df_best = pd.DataFrame(results_best)
result_path = os.path.join(output_dir, "模型对比结果.csv")
result_df_best.to_csv(result_path, index=False)
print(f"✅ 结果已保存到：{result_path}")

def save_cm(y_true, y_pred, name, filename):
    cm = confusion_matrix(y_true, y_pred, labels=label_values)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_values)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    disp.plot(ax=ax, colorbar=False)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"✅ {name} 混淆矩阵已保存：{path}")

save_cm(y_test_labels, y_pred_labels, "序数回归", "序数回归混淆矩阵.png")

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)
plt.figure(figsize=(6, 5), dpi=150)
unique_labels = np.array(label_values)
for label in unique_labels:
    mask = y == label
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=18, alpha=0.8, label=str(label))
plt.legend(title="Label")
plt.tight_layout()
scatter_path = os.path.join(output_dir, "特征PCA散点图.png")
plt.savefig(scatter_path)
plt.close()
print(f"✅ 散点图已保存：{scatter_path}")

tsne = TSNE(n_components=2, random_state=42, perplexity=20, learning_rate="auto", init="pca")
X_tsne = tsne.fit_transform(X)
plt.figure(figsize=(6, 5), dpi=150)
for label in unique_labels:
    mask = y == label
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=18, alpha=0.8, label=str(label))
plt.legend(title="Label")
plt.tight_layout()
tsne_path = os.path.join(output_dir, "特征tSNE散点图.png")
plt.savefig(tsne_path)
plt.close()
print(f"✅ t-SNE 散点图已保存：{tsne_path}")
