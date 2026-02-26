import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from mord import LogisticAT

root_dir = r"D:\同济车辆\chest_audio"
data_dir = os.path.join(root_dir, "Chest new0206", "Chest new0206")

features = [
    {
        "name": "Jitter",
        "feature_dir": os.path.join(data_dir, "JitterOutput"),
        "train_dir": os.path.join(root_dir, "Train_Jitter"),
        "ylabel": "Jitter Median",
        "scatter_all": "Jitter_median_by_score.png"
    },
    {
        "name": "Shimmer",
        "feature_dir": os.path.join(data_dir, "ShimmerOutput"),
        "train_dir": os.path.join(root_dir, "Train_Shimer"),
        "ylabel": "Shimmer Median",
        "scatter_all": "Shimmer_median_by_score.png"
    },
    {
        "name": "H1H2",
        "feature_dir": os.path.join(data_dir, "H1H2Output"),
        "train_dir": os.path.join(root_dir, "Train_H1H2"),
        "ylabel": "H1H2 Median",
        "scatter_all": "H1H2_median_by_score.png"
    },
    {
        "name": "HNR",
        "feature_dir": os.path.join(data_dir, "HNR_Output"),
        "train_dir": os.path.join(root_dir, "Train_Hnr"),
        "ylabel": "HNR Median",
        "scatter_all": "HNR_median_by_score.png"
    },
    {
        "name": "QValue",
        "feature_dir": os.path.join(data_dir, "QValueOutput"),
        "train_dir": os.path.join(root_dir, "Train_QValue"),
        "ylabel": "QValue Median",
        "scatter_all": "QValue_median_by_score.png"
    },
    {
        "name": "SpectralSlope",
        "feature_dir": os.path.join(data_dir, "SpectralSlopeOutput"),
        "train_dir": os.path.join(root_dir, "Train_SpectralSlope"),
        "ylabel": "SpectralSlope Median",
        "scatter_all": "SpectralSlope_median_by_score.png"
    },
    {
        "name": "LowFreqEnergyRatio",
        "feature_dir": os.path.join(data_dir, "LowFreqEnergyRatioOutput"),
        "train_dir": os.path.join(root_dir, "Train_LowFreqEnergyRatio"),
        "ylabel": "LowFreqEnergyRatio Median",
        "scatter_all": "LowFreqEnergyRatio_median_by_score.png"
    },
    {
        "name": "HighFreqNoiseRatio",
        "feature_dir": os.path.join(data_dir, "HighFreqNoiseRatioOutput"),
        "train_dir": os.path.join(root_dir, "Train_HighFreqNoiseRatio"),
        "ylabel": "HighFreqNoiseRatio Median",
        "scatter_all": "HighFreqNoiseRatio_median_by_score.png"
    },
    {
        "name": "CPP",
        "feature_dir": os.path.join(data_dir, "CPP_Output"),
        "train_dir": os.path.join(root_dir, "Train_CPP"),
        "ylabel": "CPP Median",
        "scatter_all": "CPP_median_by_score.png"
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

def stats_mean_std(values):
    if values is None or len(values) == 0:
        return None
    mean = float(np.mean(values))
    std = float(np.std(values))
    return [mean, std]

def build_dataset(feature_dir, allowed_suffixes=None):
    X = []
    y = []
    medians = []
    scores = []
    if not os.path.isdir(feature_dir):
        return None, None, None, None
    files = [f for f in os.listdir(feature_dir) if f.lower().endswith(".csv")]
    files.sort()
    for f in files:
        label = parse_label(f)
        if label is None:
            continue
        suffix = parse_suffix_type(f)
        if allowed_suffixes is not None and suffix not in allowed_suffixes:
            continue
        fpath = os.path.join(feature_dir, f)
        try:
            values = np.loadtxt(fpath, delimiter=",", dtype=np.float32)
        except Exception:
            continue
        if values is None or np.size(values) == 0:
            continue
        values = np.asarray(values).reshape(-1)
        feat = stats_mean_std(values)
        if feat is None:
            continue
        X.append(feat)
        y.append(label)
        medians.append(float(np.median(values)))
        scores.append(label)
    if len(X) == 0:
        return None, None, None, None
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32), np.asarray(medians, dtype=np.float32), np.asarray(scores, dtype=np.int32)

def save_median_scatter(scores, medians, ylabel, output_path):
    uniq = sorted(list(set(scores.tolist())))
    rng = np.random.default_rng(42)
    xs = scores.astype(np.float32) + rng.uniform(-0.12, 0.12, size=scores.shape[0]).astype(np.float32)
    plt.figure(figsize=(6, 5), dpi=150)
    plt.scatter(xs, medians, s=30, alpha=0.85)
    plt.xticks(uniq)
    plt.xlabel("Score")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_confusion_matrix(y_true, y_pred, output_path):
    labels = sorted(list(set(y_true.tolist())))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title="Ordinal Regression Confusion Matrix"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def train_and_save(feature_name, train_dir, X, y, scatter_path, cm_path, result_path, cv_path, ylabel, scores, medians):
    os.makedirs(train_dir, exist_ok=True)
    save_median_scatter(scores, medians, ylabel, scatter_path)
    model = LogisticAT()
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
    label_values = sorted(list(set(y.tolist())))
    label_map = {v: i for i, v in enumerate(label_values)}
    inv_label_map = {i: v for v, i in label_map.items()}
    y_idx = np.asarray([label_map[v] for v in y], dtype=np.int32)
    pipeline.fit(X, y_idx)
    preds_idx = pipeline.predict(X)
    preds = np.asarray([inv_label_map[i] for i in preds_idx], dtype=np.int32)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, average="weighted", zero_division=0)
    rec = recall_score(y, preds, average="weighted", zero_division=0)
    f1 = f1_score(y, preds, average="weighted", zero_division=0)
    result_df = pd.DataFrame(
        [{"Model": "Ordinal Regression", "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}]
    )
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores_cv = cross_validate(
        pipeline,
        X,
        y_idx,
        cv=skf,
        scoring=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
        return_train_score=False
    )
    cv_df = pd.DataFrame(scores_cv)
    cv_df.to_csv(cv_path, index=False, encoding="utf-8-sig")
    save_confusion_matrix(y, preds, cm_path)

def run_one(feature, suffix_tag, allowed_suffixes, scatter_name, cm_name, result_name, cv_name):
    X, y, medians, scores = build_dataset(feature["feature_dir"], allowed_suffixes)
    if X is None:
        print(f"❌ {feature['name']} {suffix_tag} 无有效数据，跳过")
        return
    scatter_path = os.path.join(feature["train_dir"], scatter_name)
    cm_path = os.path.join(feature["train_dir"], cm_name)
    result_path = os.path.join(feature["train_dir"], result_name)
    cv_path = os.path.join(feature["train_dir"], cv_name)
    train_and_save(
        feature["name"],
        feature["train_dir"],
        X,
        y,
        scatter_path,
        cm_path,
        result_path,
        cv_path,
        feature["ylabel"],
        scores,
        medians
    )
    print(f"✅ {feature['name']} {suffix_tag} 完成: {feature['train_dir']}")

for feature in features:
    run_one(
        feature,
        "ALL",
        None,
        feature["scatter_all"],
        "序数回归混淆矩阵.png",
        "模型对比结果.csv",
        "分层10折交叉验证结果.csv"
    )
    run_one(
        feature,
        "A1",
        ["A", "1"],
        "A1_" + feature["scatter_all"],
        "A1_序数回归混淆矩阵.png",
        "A1_模型对比结果.csv",
        "A1_分层10折交叉验证结果.csv"
    )
    run_one(
        feature,
        "B1",
        ["B", "1"],
        "B1_" + feature["scatter_all"],
        "B1_序数回归混淆矩阵.png",
        "B1_模型对比结果.csv",
        "B1_分层10折交叉验证结果.csv"
    )

print("🎉 全部特征处理完成。")
