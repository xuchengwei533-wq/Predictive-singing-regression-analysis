import os
import re
import shutil
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
output_dir = r"D:\同济车辆\chest_audio\Train_QValue"
os.makedirs(output_dir, exist_ok=True)

q_dir = os.path.join(data_dir, "QValueOutput")

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

def build_dataset():
    if not os.path.isdir(q_dir):
        raise FileNotFoundError(f"特征目录不存在: {q_dir}")
    files = [f for f in os.listdir(q_dir) if f.lower().endswith(".csv")]
    files.sort()
    X = []
    y = []
    kept = []
    for f in files:
        base = os.path.splitext(f)[0]
        label = parse_label(base)
        if label is None:
            continue
        q_path = os.path.join(q_dir, base + ".csv")
        if not os.path.exists(q_path):
            continue
        q_stats = stats_mean_std(load_series_csv(q_path))
        if q_stats is None:
            continue
        feat = np.array([q_stats[0], q_stats[1]], dtype=np.float32)
        X.append(feat)
        y.append(label)
        kept.append(base)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    return X, y, kept

print("📦 正在构建特征数据集...")
X, y, kept = build_dataset()
print(f"✅ 样本数: {len(y)} | 特征维度: {X.shape[1] if len(y) > 0 else 0}")

if len(y) == 0:
    raise RuntimeError("没有可用样本，请确认特征文件已生成且文件名匹配")

label_values = sorted(list(set(y.tolist())))
label_map = {v: i for i, v in enumerate(label_values)}
inv_label_map = {i: v for v, i in label_map.items()}
y_idx = np.asarray([label_map[v] for v in y], dtype=np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_idx, test_size=0.3, random_state=42, stratify=y_idx
)

if LogisticAT is None:
    raise RuntimeError("未检测到 mord，请先安装：pip install mord")

print("\n🚀 正在训练序数回归模型（LogisticAT）...")
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
results = [{"模型": "序数回归(LogisticAT)", "准确率": round(acc_ord, 4), "宏平均F1": round(f1_ord, 4)}]
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

print("\n" + "-" * 60)
print("📊 最终模型效果对比表：")
result_df = pd.DataFrame(results)
print(result_df)
result_path = os.path.join(output_dir, "模型对比结果.csv")
result_df.to_csv(result_path, index=False)
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

try:
    script_path = os.path.abspath(__file__)
    dst_path = os.path.join(output_dir, os.path.basename(script_path))
    if os.path.abspath(script_path) != os.path.abspath(dst_path):
        shutil.copyfile(script_path, dst_path)
        print(f"✅ 脚本已保存：{dst_path}")
except Exception:
    pass
