import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
try:
    from mord import LogisticAT
except Exception:
    LogisticAT = None

data_dir = r"D:\同济车辆\chest_audio\Chest new0206\Chest new0206"
output_dir = r"D:\同济车辆\chest_audio\Train_Shimer"
os.makedirs(output_dir, exist_ok=True)

feature_dir = os.path.join(data_dir, "ShimmerOutput")

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
        return float(np.mean(arr)), float(np.std(arr))
    stats = []
    for i in range(arr.shape[1]):
        col = arr[:, i]
        col = col[np.isfinite(col)]
        if col.size == 0:
            return None
        stats.extend([float(np.mean(col)), float(np.std(col))])
    if len(stats) < 2:
        return None
    return stats[0], stats[1]

files = [f for f in os.listdir(feature_dir) if f.lower().endswith(".csv")]
files.sort()

xs = []
ys = []
labels = []
for f in files:
    label = parse_label(f)
    if label is None:
        continue
    path = os.path.join(feature_dir, f)
    series = load_series_csv(path)
    mean_std = stats_mean_std(series)
    if mean_std is None:
        continue
    mean_val, std_val = mean_std
    xs.append(mean_val)
    ys.append(std_val)
    labels.append(label)

xs = np.asarray(xs, dtype=np.float32)
ys = np.asarray(ys, dtype=np.float32)
labels = np.asarray(labels, dtype=np.int32)

if xs.size == 0:
    raise RuntimeError("没有可用样本，请确认 ShimmerOutput 已生成")

label_values = sorted(list(set(labels.tolist())))
label_map = {v: i for i, v in enumerate(label_values)}
labels_idx = np.asarray([label_map[v] for v in labels], dtype=np.int32)

if LogisticAT is None:
    raise RuntimeError("未检测到 mord，请先安装：pip install mord")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticAT(alpha=1.0))
])
X = np.column_stack([xs, ys]).astype(np.float32)
model.fit(X, labels_idx)

pad_x = (xs.max() - xs.min()) * 0.08 if xs.size > 1 else 0.01
pad_y = (ys.max() - ys.min()) * 0.08 if ys.size > 1 else 0.01
x_min, x_max = xs.min() - pad_x, xs.max() + pad_x
y_min, y_max = ys.min() - pad_y, ys.max() + pad_y
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
pred_idx = model.predict(grid).reshape(xx.shape)

plt.figure(figsize=(6, 4.5), dpi=150)
for lab, color in [(1, "#1f77b4"), (3, "#ff7f0e"), (5, "#2ca02c")]:
    mask = labels == lab
    if np.any(mask):
        plt.scatter(xs[mask], ys[mask], s=20, alpha=0.85, label=str(lab), color=color)
plt.contour(xx, yy, pred_idx, levels=[0.5, 1.5], colors=["#111111", "#111111"], linewidths=1.2)
plt.xlabel("Shimmer Mean")
plt.ylabel("Shimmer Std")
plt.legend(title="Score")
plt.tight_layout()
out_path = os.path.join(output_dir, "Shimmer_mean_std_by_score.png")
plt.savefig(out_path)
plt.close()
print(f"✅ 已保存：{out_path}")
