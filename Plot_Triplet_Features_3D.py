import os
import re
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir, "Chest new0206", "Chest new0206")
score_path = os.path.join(root_dir, "打分Chest_new0206_scores_matrix.xlsx")
out_root = os.path.join(root_dir, "triplet_3d")
all_root = os.path.join(out_root, "all")

features = [
    {
        "name": "Jitter",
        "feature_dir": os.path.join(data_dir, "JitterOutput"),
        "unit": "%"
    },
    {
        "name": "Shimmer",
        "feature_dir": os.path.join(data_dir, "ShimmerOutput"),
        "unit": "%"
    },
    {
        "name": "H1H2",
        "feature_dir": os.path.join(data_dir, "H1H2Output"),
        "unit": "dB"
    },
    {
        "name": "HNR",
        "feature_dir": os.path.join(data_dir, "HNR_Output"),
        "unit": "dB"
    },
    {
        "name": "QValue",
        "feature_dir": os.path.join(data_dir, "QValueOutput"),
        "unit": "ratio"
    },
    {
        "name": "SpectralSlope",
        "feature_dir": os.path.join(data_dir, "SpectralSlopeOutput"),
        "unit": "dB/Hz"
    },
    {
        "name": "LowFreqEnergyRatio",
        "feature_dir": os.path.join(data_dir, "LowFreqEnergyRatioOutput"),
        "unit": "ratio"
    },
    {
        "name": "HighFreqNoiseRatio",
        "feature_dir": os.path.join(data_dir, "HighFreqNoiseRatioOutput"),
        "unit": "ratio"
    },
    {
        "name": "CPP",
        "feature_dir": os.path.join(data_dir, "CPP_Output"),
        "unit": "dB"
    }
]

x_priority = {
    "HighFreqNoiseRatio": 0,
    "LowFreqEnergyRatio": 1,
    "CPP": 2
}

def normalize_id(value):
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    s = s.replace("\\", "/")
    s = os.path.basename(s)
    s = re.sub(r"\.(wav|csv|xlsx)$", "", s, flags=re.IGNORECASE)
    return s.lower()

def parse_suffix_type(filename):
    base = os.path.splitext(filename)[0]
    if re.search(r"-A$", base, flags=re.IGNORECASE):
        return "A"
    if re.search(r"-B$", base, flags=re.IGNORECASE):
        return "B"
    if re.search(r"-1$", base):
        return "1"
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

def load_score_matrix(path):
    df = pd.read_excel(path)
    if df.empty:
        return df
    mask_delete = df.apply(lambda row: row.astype(str).str.contains("删除", na=False).any(), axis=1)
    df = df[~mask_delete].copy()
    df = df.dropna(axis=0, how="all")
    return df

def detect_id_column(df):
    keywords = ["文件", "filename", "file", "name", "音频", "sample", "id"]
    for col in df.columns:
        name = str(col).lower()
        if any(k in name for k in keywords):
            return col
    return df.columns[0]

def build_score_map(df, technique):
    id_col = detect_id_column(df)
    id_series = df[id_col].astype(str)
    id_norm = id_series.apply(normalize_id)
    if technique not in df.columns:
        return {}
    scores = pd.to_numeric(df[technique], errors="coerce")
    mapping = {}
    for sid, score in zip(id_norm, scores):
        if not sid:
            continue
        if pd.isna(score):
            continue
        mapping[sid] = int(round(float(score)))
    return mapping

def build_feature_medians(feature_dir, score_map, allowed_suffixes=None):
    vals = {}
    if not os.path.isdir(feature_dir):
        return vals
    files = [f for f in os.listdir(feature_dir) if f.lower().endswith(".csv")]
    files.sort()
    for f in files:
        suffix = parse_suffix_type(f)
        if allowed_suffixes is not None and suffix not in allowed_suffixes:
            continue
        base_id = normalize_id(os.path.splitext(f)[0])
        if base_id not in score_map:
            continue
        series = load_series(os.path.join(feature_dir, f))
        if series is None:
            continue
        vals[base_id] = float(np.median(series))
    return vals

def build_triplet_points(feat_a, feat_b, feat_c, score_map, allowed_suffixes=None):
    vals_a = build_feature_medians(feat_a["feature_dir"], score_map, allowed_suffixes)
    vals_b = build_feature_medians(feat_b["feature_dir"], score_map, allowed_suffixes)
    vals_c = build_feature_medians(feat_c["feature_dir"], score_map, allowed_suffixes)
    keys = sorted(set(vals_a.keys()) & set(vals_b.keys()) & set(vals_c.keys()))
    xs, ys, zs, scores = [], [], [], []
    for k in keys:
        score = score_map.get(k)
        if score is None:
            continue
        xs.append(vals_a[k])
        ys.append(vals_b[k])
        zs.append(vals_c[k])
        scores.append(int(score))
    if len(xs) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(zs, dtype=np.float32),
        np.asarray(scores, dtype=np.int32)
    )

def label_with_unit(feature):
    unit = feature.get("unit", "")
    if unit:
        return f'{feature["name"]} ({unit})'
    return feature["name"]

def robust_limits(arr, lo=2, hi=98, pad_ratio=0.08):
    lo_v, hi_v = np.percentile(arr, [lo, hi])
    r = hi_v - lo_v
    pad = r * pad_ratio if r > 0 else 1.0
    return float(lo_v - pad), float(hi_v + pad)

def order_triplet(feat_a, feat_b, feat_c):
    feats = [feat_a, feat_b, feat_c]
    feats.sort(key=lambda f: (x_priority.get(f["name"], 99), f["name"]))
    return feats[0], feats[1], feats[2]

def plot_triplet(technique, feat_a, feat_b, feat_c, score_map, allowed_suffixes=None, out_subdir="ALL"):
    xs, ys, zs, scores = build_triplet_points(feat_a, feat_b, feat_c, score_map, allowed_suffixes)
    if xs.size == 0:
        return False
    fig = plt.figure(figsize=(14.0, 7.2), dpi=320)
    ax = fig.add_subplot(111, projection="3d")
    color_map = {1: "#1f77b4", 3: "#2ca02c", 5: "#ff7f0e"}
    for score_value in [1, 3, 5]:
        mask = scores == score_value
        if np.any(mask):
            ax.scatter(xs[mask], ys[mask], zs[mask], s=22, alpha=0.85, color=color_map[score_value], marker="o", label=str(score_value))
    other_mask = ~np.isin(scores, [1, 3, 5])
    if np.any(other_mask):
        ax.scatter(xs[other_mask], ys[other_mask], zs[other_mask], s=22, alpha=0.6, color="#7f7f7f", marker="o", label="Other")
    ax.set_xlabel(label_with_unit(feat_a), labelpad=14)
    ax.set_ylabel(label_with_unit(feat_b), labelpad=14)
    ax.set_zlabel(label_with_unit(feat_c), labelpad=16)
    ax.tick_params(axis="both", labelsize=9, pad=2)
    ax.tick_params(axis="z", labelsize=9, pad=2)
    ax.set_proj_type("persp")
    ax.set_xlim(*robust_limits(xs, 1, 99))
    ax.set_ylim(*robust_limits(ys, 1, 99))
    ax.set_zlim(*robust_limits(zs, 1, 99))
    ax.view_init(elev=28, azim=-55)
    ax.xaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.25))
    ax.yaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.25))
    ax.zaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.25))
    ax.grid(True, linewidth=0.6, alpha=0.6)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle(f"{technique} - {feat_a['name']} vs {feat_b['name']} vs {feat_c['name']}", fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.08, top=0.9)
    out_dir = os.path.join(all_root, out_subdir, technique)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{feat_a['name']}_{feat_b['name']}_{feat_c['name']}.png"
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=320)
    plt.close(fig)
    return True

def plot_triplet_combo(technique, feat_a, feat_b, feat_c, score_map, out_subdir="A1_B1_ALL"):
    a1 = build_triplet_points(feat_a, feat_b, feat_c, score_map, {"A", "1"})
    b1 = build_triplet_points(feat_a, feat_b, feat_c, score_map, {"B", "1"})
    all_pts = build_triplet_points(feat_a, feat_b, feat_c, score_map, None)
    if all_pts[0].size == 0 and a1[0].size == 0 and b1[0].size == 0:
        return False
    xs_all = np.concatenate([a1[0], b1[0], all_pts[0]]) if a1[0].size + b1[0].size + all_pts[0].size > 0 else np.array([])
    ys_all = np.concatenate([a1[1], b1[1], all_pts[1]]) if a1[1].size + b1[1].size + all_pts[1].size > 0 else np.array([])
    zs_all = np.concatenate([a1[2], b1[2], all_pts[2]]) if a1[2].size + b1[2].size + all_pts[2].size > 0 else np.array([])
    if xs_all.size == 0:
        return False
    xlim = robust_limits(xs_all, 1, 99)
    ylim = robust_limits(ys_all, 1, 99)
    zlim = robust_limits(zs_all, 1, 99)
    fig = plt.figure(figsize=(18.0, 6.8), dpi=320)
    axes = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d")
    ]
    groups = [("A1", a1), ("B1", b1), ("ALL", all_pts)]
    color_map = {1: "#1f77b4", 3: "#2ca02c", 5: "#ff7f0e"}
    for ax, (title, (xs, ys, zs, scores)) in zip(axes, groups):
        if xs.size > 0:
            for score_value in [1, 3, 5]:
                mask = scores == score_value
                if np.any(mask):
                    ax.scatter(xs[mask], ys[mask], zs[mask], s=18, alpha=0.85, color=color_map[score_value], marker="o", label=str(score_value))
            other_mask = ~np.isin(scores, [1, 3, 5])
            if np.any(other_mask):
                ax.scatter(xs[other_mask], ys[other_mask], zs[other_mask], s=18, alpha=0.6, color="#7f7f7f", marker="o", label="Other")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(label_with_unit(feat_a), labelpad=12)
        ax.set_ylabel(label_with_unit(feat_b), labelpad=12)
        ax.set_zlabel(label_with_unit(feat_c), labelpad=12)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.view_init(elev=28, azim=-55)
        ax.grid(True, linewidth=0.6, alpha=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(f"{technique} - {feat_a['name']} vs {feat_b['name']} vs {feat_c['name']}", fontsize=12)
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.08, top=0.88, wspace=0.05)
    out_dir = os.path.join(all_root, out_subdir, technique)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{feat_a['name']}_{feat_b['name']}_{feat_c['name']}.png"
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=320)
    plt.close(fig)
    return True

def plot_triplet_pair(technique, feat_a, feat_b, feat_c, score_map, out_subdir="A1_B1"):
    a1 = build_triplet_points(feat_a, feat_b, feat_c, score_map, {"A", "1"})
    b1 = build_triplet_points(feat_a, feat_b, feat_c, score_map, {"B", "1"})
    if a1[0].size == 0 and b1[0].size == 0:
        return False
    xs_all = np.concatenate([a1[0], b1[0]]) if a1[0].size + b1[0].size > 0 else np.array([])
    ys_all = np.concatenate([a1[1], b1[1]]) if a1[1].size + b1[1].size > 0 else np.array([])
    zs_all = np.concatenate([a1[2], b1[2]]) if a1[2].size + b1[2].size > 0 else np.array([])
    if xs_all.size == 0:
        return False
    xlim = robust_limits(xs_all, 1, 99)
    ylim = robust_limits(ys_all, 1, 99)
    zlim = robust_limits(zs_all, 1, 99)
    fig = plt.figure(figsize=(12.8, 6.8), dpi=320)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d")
    ]
    groups = [("A1", a1), ("B1", b1)]
    color_map = {1: "#1f77b4", 3: "#2ca02c", 5: "#ff7f0e"}
    for ax, (title, (xs, ys, zs, scores)) in zip(axes, groups):
        if xs.size > 0:
            for score_value in [1, 3, 5]:
                mask = scores == score_value
                if np.any(mask):
                    ax.scatter(xs[mask], ys[mask], zs[mask], s=18, alpha=0.85, color=color_map[score_value], marker="o", label=str(score_value))
            other_mask = ~np.isin(scores, [1, 3, 5])
            if np.any(other_mask):
                ax.scatter(xs[other_mask], ys[other_mask], zs[other_mask], s=18, alpha=0.6, color="#7f7f7f", marker="o", label="Other")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(label_with_unit(feat_a), labelpad=12)
        ax.set_ylabel(label_with_unit(feat_b), labelpad=12)
        ax.set_zlabel(label_with_unit(feat_c), labelpad=12)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.view_init(elev=28, azim=-55)
        ax.grid(True, linewidth=0.6, alpha=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(f"{technique} - {feat_a['name']} vs {feat_b['name']} vs {feat_c['name']}", fontsize=12)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.08, top=0.88, wspace=0.08)
    out_dir = os.path.join(all_root, out_subdir, technique)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{feat_a['name']}_{feat_b['name']}_{feat_c['name']}.png"
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=320)
    plt.close(fig)
    return True

df_scores = load_score_matrix(score_path)
technique = "chest"
score_map = build_score_map(df_scores, technique)

count = 0
for feat_a, feat_b, feat_c in itertools.combinations(features, 3):
    feat_x, feat_y, feat_z = order_triplet(feat_a, feat_b, feat_c)
    if plot_triplet(technique, feat_x, feat_y, feat_z, score_map, {"A", "1"}, "A1"):
        count += 1
    if plot_triplet(technique, feat_x, feat_y, feat_z, score_map, {"B", "1"}, "B1"):
        count += 1
    if plot_triplet(technique, feat_x, feat_y, feat_z, score_map, None, "ALL"):
        count += 1
    if plot_triplet_pair(technique, feat_x, feat_y, feat_z, score_map, "A1_B1"):
        count += 1
    if plot_triplet_combo(technique, feat_x, feat_y, feat_z, score_map, "A1_B1_ALL"):
        count += 1

print(f"🎉 chest 三维特征图生成完成，共输出 {count} 张。")
