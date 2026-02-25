import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
try:
    from mord import LogisticAT
except Exception:
    LogisticAT = None

data_dir = r"D:\同济车辆\chest_audio\Chest new0206\Chest new0206"
output_dir = r"D:\同济车辆\chest_audio\Train_Jitter"
os.makedirs(output_dir, exist_ok=True)

jitter_dir = os.path.join(data_dir, "JitterOutput")

def parse_label(filename):
    name = os.path.splitext(filename)[0]
    nums = re.findall(r"\d+", name)
    if not nums:
        return None
    return int(nums[-1])

def parse_suffix_type(filename):
    base = os.path.splitext(filename)[0]
    if re.search(r"-A$", base, flags=re.IGNORECASE):
        return "A"
    if re.search(r"-B$", base, flags=re.IGNORECASE):
        return "B"
    if re.search(r"-1$", base):
        return "1"
    return None

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

def build_dataset(allowed_suffixes):
    if not os.path.isdir(jitter_dir):
        raise FileNotFoundError(f"特征目录不存在: {jitter_dir}")
    files = [f for f in os.listdir(jitter_dir) if f.lower().endswith(".csv")]
    files.sort()
    X = []
    y = []
    kept = []
    for f in files:
        suffix_type = parse_suffix_type(f)
        if suffix_type not in allowed_suffixes:
            continue
        base = os.path.splitext(f)[0]
        label = parse_label(base)
        if label is None:
            continue
        jitter_path = os.path.join(jitter_dir, base + ".csv")
        if not os.path.exists(jitter_path):
            continue
        jitter_stats = stats_mean_std(load_series_csv(jitter_path))
        if jitter_stats is None:
            continue
        feat = np.array([jitter_stats[0], jitter_stats[1]], dtype=np.float32)
        X.append(feat)
        y.append(label)
        kept.append(base)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    return X, y, kept

def build_median_points(allowed_suffixes):
    if not os.path.isdir(jitter_dir):
        raise FileNotFoundError(f"特征目录不存在: {jitter_dir}")
    files = [f for f in os.listdir(jitter_dir) if f.lower().endswith(".csv")]
    files.sort()
    scores = []
    medians = []
    for f in files:
        suffix_type = parse_suffix_type(f)
        if suffix_type not in allowed_suffixes:
            continue
        base = os.path.splitext(f)[0]
        label = parse_label(base)
        if label is None:
            continue
        path = os.path.join(jitter_dir, f)
        series = load_series_csv(path)
        if series is None:
            continue
        arr = np.asarray(series, dtype=np.float32).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        scores.append(int(label))
        medians.append(float(np.median(arr)))
    return np.asarray(scores, dtype=np.int32), np.asarray(medians, dtype=np.float32)

def save_median_scatter(scores, medians, filename):
    plt.figure(figsize=(6, 4.5), dpi=150)
    np.random.seed(42)
    xs_jitter = scores + (np.random.rand(scores.size) - 0.5) * 0.15
    plt.scatter(xs_jitter, medians, s=18, alpha=0.8)
    plt.xticks([1, 3, 5], ["1", "3", "5"])
    plt.xlabel("Score")
    plt.ylabel("Jitter Median")
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()
    return path

def save_cm(y_true, y_pred, label_values, filename):
    cm = confusion_matrix(y_true, y_pred, labels=label_values)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_values)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    disp.plot(ax=ax, colorbar=False)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()
    return path

def run_one(name, allowed_suffixes):
    print(f"\n📦 正在构建数据集: {name} | 选择后缀: {sorted(list(allowed_suffixes))}")
    X, y, kept = build_dataset(allowed_suffixes)
    print(f"✅ {name} 样本数: {len(y)} | 特征维度: {X.shape[1] if len(y) > 0 else 0}")
    if len(y) == 0:
        raise RuntimeError(f"{name} 没有可用样本，请确认特征文件名后缀与目录内容")

    scores, medians = build_median_points(allowed_suffixes)
    if scores.size > 0 and medians.size > 0:
        median_plot_path = save_median_scatter(scores, medians, f"{name}_Jitter_median_by_score.png")
        print(f"✅ {name} Median散点图已保存：{median_plot_path}")

    label_values = sorted(list(set(y.tolist())))
    label_map = {v: i for i, v in enumerate(label_values)}
    inv_label_map = {i: v for v, i in label_map.items()}
    y_idx = np.asarray([label_map[v] for v in y], dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=0.3, random_state=42, stratify=y_idx
    )

    if LogisticAT is None:
        raise RuntimeError("未检测到 mord，请先安装：pip install mord")

    ord_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticAT(alpha=1.0))
    ])
    ord_model.fit(X_train, y_train)
    y_pred_idx = ord_model.predict(X_test)
    y_test_labels = np.asarray([inv_label_map[i] for i in y_test], dtype=np.int32)
    y_pred_labels = np.asarray([inv_label_map[i] for i in y_pred_idx], dtype=np.int32)
    acc = accuracy_score(y_test_labels, y_pred_labels)
    f1 = f1_score(y_test_labels, y_pred_labels, average="macro")

    results = pd.DataFrame([{
        "子集": name,
        "样本数": int(len(y)),
        "特征维度": int(X.shape[1]),
        "准确率": round(float(acc), 4),
        "宏平均F1": round(float(f1), 4)
    }])
    results_path = os.path.join(output_dir, f"{name}_模型对比结果.csv")
    results.to_csv(results_path, index=False)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_acc = cross_val_score(ord_model, X, y_idx, cv=skf, scoring="accuracy")
    cv_f1 = cross_val_score(ord_model, X, y_idx, cv=skf, scoring="f1_macro")
    cv_df = pd.DataFrame({
        "指标": ["accuracy", "macro_f1"],
        "均值": [float(cv_acc.mean()), float(cv_f1.mean())],
        "标准差": [float(cv_acc.std()), float(cv_f1.std())]
    })
    cv_path = os.path.join(output_dir, f"{name}_分层10折交叉验证结果.csv")
    cv_df.to_csv(cv_path, index=False)

    cm_path = save_cm(y_test_labels, y_pred_labels, label_values, f"{name}_序数回归混淆矩阵.png")
    print(f"✅ {name} 结果已保存：{results_path}")
    print(f"✅ {name} CV结果已保存：{cv_path}")
    print(f"✅ {name} 混淆矩阵已保存：{cm_path}")

print("🚀 开始训练两个子集：A+1 与 B+1（Jitter 序数回归）")
run_one("A1", {"A", "1"})
run_one("B1", {"B", "1"})
