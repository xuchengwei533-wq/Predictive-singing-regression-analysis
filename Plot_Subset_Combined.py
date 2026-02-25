import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt

root_dir = r"D:\同济车辆\chest_audio"
data_dir = os.path.join(root_dir, "Chest new0206", "Chest new0206")
TECHNIQUE = "chest"

features = [
    {
        "name": "Jitter",
        "feature_dir": os.path.join(data_dir, "JitterOutput"),
        "train_dir": os.path.join(root_dir, "Train_Jitter"),
        "ylabel": "Jitter Median"
    },
    {
        "name": "Shimmer",
        "feature_dir": os.path.join(data_dir, "ShimmerOutput"),
        "train_dir": os.path.join(root_dir, "Train_Shimer"),
        "ylabel": "Shimmer Median"
    },
    {
        "name": "H1H2",
        "feature_dir": os.path.join(data_dir, "H1H2Output"),
        "train_dir": os.path.join(root_dir, "Train_H1H2"),
        "ylabel": "H1H2 Median"
    },
    {
        "name": "HNR",
        "feature_dir": os.path.join(data_dir, "HNR_Output"),
        "train_dir": os.path.join(root_dir, "Train_Hnr"),
        "ylabel": "HNR Median"
    },
    {
        "name": "QValue",
        "feature_dir": os.path.join(data_dir, "QValueOutput"),
        "train_dir": os.path.join(root_dir, "Train_QValue"),
        "ylabel": "QValue Median"
    },
    {
        "name": "SpectralSlope",
        "feature_dir": os.path.join(data_dir, "SpectralSlopeOutput"),
        "train_dir": os.path.join(root_dir, "Train_SpectralSlope"),
        "ylabel": "SpectralSlope Median"
    },
    {
        "name": "LowFreqEnergyRatio",
        "feature_dir": os.path.join(data_dir, "LowFreqEnergyRatioOutput"),
        "train_dir": os.path.join(root_dir, "Train_LowFreqEnergyRatio"),
        "ylabel": "LowFreqEnergyRatio Median"
    },
    {
        "name": "HighFreqNoiseRatio",
        "feature_dir": os.path.join(data_dir, "HighFreqNoiseRatioOutput"),
        "train_dir": os.path.join(root_dir, "Train_HighFreqNoiseRatio"),
        "ylabel": "HighFreqNoiseRatio Median"
    },
    {
        "name": "CPP",
        "feature_dir": os.path.join(data_dir, "CPP_Output"),
        "train_dir": os.path.join(root_dir, "Train_CPP"),
        "ylabel": "CPP Median"
    }
]

def parse_label(filename):
    base = os.path.splitext(filename)[0]
    nums = re.findall(r"\d+", base)
    if nums:
        return int(nums[-1])
    return None

def parse_suffix_type(filename):
    base = os.path.splitext(filename)[0]
    if re.search(r"-A$", base, flags=re.IGNORECASE):
        return "A"
    if re.search(r"-B$", base, flags=re.IGNORECASE):
        return "B"
    if re.search(r"-1$", base):
        return "1"
    return None

def parse_pitch_digit(filename):
    base = os.path.splitext(filename)[0]
    match = re.search(r"([A-Ga-g])(\d)", base)
    if match:
        return int(match.group(2))
    return None

def load_series(path):
    try:
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except Exception:
        return None
    if data is None or np.size(data) == 0:
        return None
    arr = np.asarray(data, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return arr

def build_median_points(feature_dir, allowed_suffixes=None):
    xs = []
    ys = []
    pitches = []
    if not os.path.isdir(feature_dir):
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    files = [f for f in os.listdir(feature_dir) if f.lower().endswith(".csv")]
    files.sort()
    for f in files:
        label = parse_label(f)
        if label is None:
            continue
        suffix = parse_suffix_type(f)
        if allowed_suffixes is not None and suffix not in allowed_suffixes:
            continue
        series = load_series(os.path.join(feature_dir, f))
        if series is None:
            continue
        pitch = parse_pitch_digit(f)
        if pitch is None:
            continue
        xs.append(label)
        ys.append(float(np.median(series)))
        pitches.append(pitch)
    return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.float32), np.asarray(pitches, dtype=np.int32)

def scatter_subplot(ax, scores, medians, pitches, ylabel=None, title=None):
    if scores.size == 0:
        ax.set_axis_off()
        return
    rng = np.random.default_rng(42)
    xs = scores.astype(np.float32) + rng.uniform(-0.12, 0.12, size=scores.shape[0]).astype(np.float32)
    style_map = {
        3: {"color": "#1f77b4", "marker": "^"},
        4: {"color": "#2ca02c", "marker": "s"},
        5: {"color": "#ff7f0e", "marker": "o"}
    }
    for pitch_val, style in style_map.items():
        mask = pitches == pitch_val
        if np.any(mask):
            ax.scatter(xs[mask], medians[mask], s=18, alpha=0.8, color=style["color"], marker=style["marker"])
    ax.set_xticks([1, 3, 5])
    ax.set_xticklabels(["1", "3", "5"])
    ax.set_xlabel("Score")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

def save_triplet(feature):
    all_scores, all_medians, all_pitches = build_median_points(feature["feature_dir"], None)
    a1_scores, a1_medians, a1_pitches = build_median_points(feature["feature_dir"], {"A", "1"})
    b1_scores, b1_medians, b1_pitches = build_median_points(feature["feature_dir"], {"B", "1"})
    all_med = np.concatenate([all_medians, a1_medians, b1_medians], axis=0) if all_medians.size + a1_medians.size + b1_medians.size > 0 else np.array([])
    if all_med.size == 0:
        return
    y_min = float(np.min(all_med))
    y_max = float(np.max(all_med))
    pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.01
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), dpi=150, sharey=True)
    scatter_subplot(axes[0], a1_scores, a1_medians, a1_pitches, ylabel=feature["ylabel"], title="A1")
    scatter_subplot(axes[1], b1_scores, b1_medians, b1_pitches, title="B1")
    scatter_subplot(axes[2], all_scores, all_medians, all_pitches, title="ALL")
    for ax in axes:
        if ax.has_data():
            ax.set_ylim(y_min - pad, y_max + pad)
    fig.suptitle(f"{TECHNIQUE} - {feature['name']} (A1/B1/ALL)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(feature["train_dir"], exist_ok=True)
    out_path = os.path.join(feature["train_dir"], f"{feature['name']}_A1_B1_ALL.png")
    fig.savefig(out_path)
    plt.close(fig)

def save_all_grid(suffix_tag, allowed_suffixes):
    imgs = []
    for feature in features:
        scores, medians, pitches = build_median_points(feature["feature_dir"], allowed_suffixes)
        imgs.append((feature, scores, medians, pitches))
    cols = 3
    rows = int(math.ceil(len(imgs) / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2), dpi=150)
    axes = np.array(axes).reshape(-1)
    for idx, (feature, scores, medians, pitches) in enumerate(imgs):
        ax = axes[idx]
        if scores.size == 0:
            ax.set_axis_off()
            continue
        scatter_subplot(ax, scores, medians, pitches, ylabel=feature["ylabel"], title=feature["name"])
    for j in range(len(imgs), len(axes)):
        axes[j].set_axis_off()
    fig.suptitle(f"{TECHNIQUE} - {suffix_tag} (9 Features)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir = os.path.join(root_dir, "all")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"all_{suffix_tag}.png")
    fig.savefig(out_path)
    plt.close(fig)

for feature in features:
    save_triplet(feature)

save_all_grid("A1", {"A", "1"})
save_all_grid("B1", {"B", "1"})

print("🎉 合并图生成完成。")
