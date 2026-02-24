import os
import re
import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
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
mfcc3_dir = os.path.join(data_dir, "Mfcc3Output")

feature_groups = {
    "Jitter": ["Jitter_mean", "Jitter_std"],
    "Shimmer": ["Shimmer_mean", "Shimmer_std"],
    "RMS": ["RMS_mean", "RMS_max"],
    "Centroid": ["Centroid_mean", "Centroid_std"],
    "H1H2": ["H1H2_mean", "H1H2_std"],
    "Mfcc3": ["Mfcc1_mean", "Mfcc1_std", "Mfcc2_mean", "Mfcc2_std", "Mfcc3_mean", "Mfcc3_std"]
}

all_features = []
for group in feature_groups.values():
    all_features.extend(group)

label_col = "score"

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

def build_dataset():
    for req_dir in [jitter_dir, shimmer_dir, rms_dir, centroid_dir, h1h2_dir, mfcc3_dir]:
        if not os.path.isdir(req_dir):
            raise FileNotFoundError(f"特征目录不存在: {req_dir}")
    files = [f for f in os.listdir(jitter_dir) if f.lower().endswith(".csv")]
    files.sort()
    rows = []
    for f in files:
        base = os.path.splitext(f)[0]
        label = parse_label(base)
        if label is None:
            continue
        jitter_path = os.path.join(jitter_dir, base + ".csv")
        shimmer_path = os.path.join(shimmer_dir, base + ".csv")
        rms_path = os.path.join(rms_dir, base + ".csv")
        centroid_path = os.path.join(centroid_dir, base + ".csv")
        h1h2_path = os.path.join(h1h2_dir, base + ".csv")
        mfcc3_path = os.path.join(mfcc3_dir, base + ".csv")
        if not (os.path.exists(jitter_path) and os.path.exists(shimmer_path) and os.path.exists(rms_path) and os.path.exists(centroid_path) and os.path.exists(h1h2_path) and os.path.exists(mfcc3_path)):
            continue
        jitter_stats = stats_mean_std(load_series_csv(jitter_path))
        shimmer_stats = stats_mean_std(load_series_csv(shimmer_path))
        rms_stats = stats_mean_max(load_series_csv(rms_path))
        centroid_stats = stats_mean_std(load_series_csv(centroid_path))
        h1h2_stats = stats_mean_std(load_series_csv(h1h2_path))
        mfcc3_stats = stats_mean_std(load_series_csv(mfcc3_path))
        if any(v is None for v in [jitter_stats, shimmer_stats, rms_stats, centroid_stats, h1h2_stats, mfcc3_stats]):
            continue
        row = {}
        row["Jitter_mean"] = float(jitter_stats[0])
        row["Jitter_std"] = float(jitter_stats[1])
        row["Shimmer_mean"] = float(shimmer_stats[0])
        row["Shimmer_std"] = float(shimmer_stats[1])
        row["RMS_mean"] = float(rms_stats[0])
        row["RMS_max"] = float(rms_stats[1])
        row["Centroid_mean"] = float(centroid_stats[0])
        row["Centroid_std"] = float(centroid_stats[1])
        row["H1H2_mean"] = float(h1h2_stats[0])
        row["H1H2_std"] = float(h1h2_stats[1])
        row["Mfcc1_mean"] = float(mfcc3_stats[0])
        row["Mfcc1_std"] = float(mfcc3_stats[1])
        row["Mfcc2_mean"] = float(mfcc3_stats[2])
        row["Mfcc2_std"] = float(mfcc3_stats[3])
        row["Mfcc3_mean"] = float(mfcc3_stats[4])
        row["Mfcc3_std"] = float(mfcc3_stats[5])
        row[label_col] = label
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def run_ordered_model(X, y):
    model = OrderedModel(y, X, distr="probit")
    result = model.fit(method="bfgs", maxiter=10000, disp=False)
    return result

def build_result_table(result, feature_list):
    table = pd.DataFrame({
        "特征名称": result.params.index,
        "回归系数": result.params.values.round(4),
        "标准误": result.bse.values.round(4),
        "Z值": result.tvalues.values.round(4),
        "P值": result.pvalues.values.round(4),
        "优势比OR": np.exp(result.params.values).round(4),
        "OR_95%CI下限": np.exp(result.conf_int()[0]).round(4),
        "OR_95%CI上限": np.exp(result.conf_int()[1]).round(4)
    })
    table = table[table["特征名称"].isin(feature_list)].reset_index(drop=True)
    table["影响度"] = table["回归系数"].abs().round(4)
    return table

def compute_vif(X):
    values = X.values
    vif_list = []
    for i, col in enumerate(X.columns):
        vif_value = variance_inflation_factor(values, i)
        vif_list.append({"特征名称": col, "VIF": float(vif_value)})
    return pd.DataFrame(vif_list)

def drop_high_vif(X, threshold=5.0):
    current = X.copy()
    while current.shape[1] > 1:
        vif_table = compute_vif(current)
        max_vif = vif_table["VIF"].max()
        if max_vif <= threshold:
            return current, vif_table
        drop_feature = vif_table.sort_values("VIF", ascending=False).iloc[0]["特征名称"]
        current = current.drop(columns=[drop_feature])
    return current, compute_vif(current)

df = build_dataset()
if df.empty:
    raise RuntimeError("没有可用样本，请确认特征文件已生成且文件名匹配")

dataset_path = os.path.join(output_dir, "声学特征标签数据.csv")
df.to_csv(dataset_path, index=False, encoding="utf-8-sig")

X_raw = df[all_features].copy()

scaler = StandardScaler()
df[all_features] = scaler.fit_transform(df[all_features])

label_values = sorted(df[label_col].unique().tolist())
label_map = {v: i for i, v in enumerate(label_values)}
y_idx = np.asarray([label_map[v] for v in df[label_col]], dtype=np.int32)

if LogisticAT is None:
    raise RuntimeError("未检测到 mord，请先安装：pip install mord")

cv_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticAT(alpha=1.0))
])
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_acc = cross_val_score(cv_model, X_raw.values, y_idx, cv=skf, scoring="accuracy")
cv_f1 = cross_val_score(cv_model, X_raw.values, y_idx, cv=skf, scoring="f1_macro")
cv_df = pd.DataFrame({
    "指标": ["accuracy", "macro_f1"],
    "均值": [float(cv_acc.mean()), float(cv_f1.mean())],
    "标准差": [float(cv_acc.std()), float(cv_f1.std())]
})
cv_path = os.path.join(output_dir, "分层10折交叉验证结果.csv")
cv_df.to_csv(cv_path, index=False, encoding="utf-8-sig")
print(f"✅ 分层10折交叉验证已保存到：{cv_path}")

categories = sorted(df[label_col].unique().tolist())
y = pd.Categorical(df[label_col], categories=categories, ordered=True)

univariate_rows = []
for feat in all_features:
    X_uni = df[[feat]]
    result_uni = run_ordered_model(X_uni, y)
    table_uni = build_result_table(result_uni, [feat])
    univariate_rows.append(table_uni)

univariate_table = pd.concat(univariate_rows, axis=0).reset_index(drop=True)
univariate_path = os.path.join(output_dir, "单因素序数回归结果.csv")
univariate_table.to_csv(univariate_path, index=False, encoding="utf-8-sig")
print(f"✅ 单因素结果已保存到：{univariate_path}")

selected_features = univariate_table[univariate_table["P值"] < 0.1]["特征名称"].tolist()
if len(selected_features) == 0:
    selected_features = all_features

corr_matrix = df[selected_features].corr()
corr_path = os.path.join(output_dir, "特征相关矩阵.csv")
corr_matrix.to_csv(corr_path, encoding="utf-8-sig")
print(f"✅ 相关矩阵已保存到：{corr_path}")

X_multi = df[selected_features].copy()
X_vif, vif_table = drop_high_vif(X_multi, threshold=5.0)
vif_path = os.path.join(output_dir, "特征VIF结果.csv")
vif_table.to_csv(vif_path, index=False, encoding="utf-8-sig")
print(f"✅ VIF结果已保存到：{vif_path}")

result = run_ordered_model(X_vif, y)

print("=" * 80)
print("📊 有序逻辑回归完整结果")
print("=" * 80)
print(result.summary())

result_table = build_result_table(result, list(X_vif.columns))
result_path = os.path.join(output_dir, "声学参数与评分关联分析结果.csv")
result_table.to_csv(result_path, index=False, encoding="utf-8-sig")
print(f"\n✅ 核心结果已保存到：{result_path}")

significant_table = result_table[
    (result_table["P值"] < 0.05)
    & ((result_table["OR_95%CI下限"] > 1) | (result_table["OR_95%CI上限"] < 1))
].reset_index(drop=True)

print("\n✅ 对胸声评分有显著影响的声学参数：")
if significant_table.empty:
    print("未筛到显著特征")
else:
    print(significant_table[["特征名称", "P值", "优势比OR", "OR_95%CI下限", "OR_95%CI上限"]])
