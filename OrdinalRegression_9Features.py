import os
import re
import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.preprocessing import StandardScaler


root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir, "Chest new0206", "Chest new0206")
score_path = os.path.join(root_dir, "打分Chest_new0206_scores_matrix.xlsx")
out_root = os.path.join(root_dir, "OrdinalRegression_9Features_Output")

features = [
    {
        "name": "Jitter",
        "feature_dir": os.path.join(data_dir, "JitterOutput")
    },
    {
        "name": "Shimmer",
        "feature_dir": os.path.join(data_dir, "ShimmerOutput")
    },
    {
        "name": "H1H2",
        "feature_dir": os.path.join(data_dir, "H1H2Output")
    },
    {
        "name": "HNR",
        "feature_dir": os.path.join(data_dir, "HNR_Output")
    },
    {
        "name": "QValue",
        "feature_dir": os.path.join(data_dir, "QValueOutput")
    },
    {
        "name": "SpectralSlope",
        "feature_dir": os.path.join(data_dir, "SpectralSlopeOutput")
    },
    {
        "name": "LowFreqEnergyRatio",
        "feature_dir": os.path.join(data_dir, "LowFreqEnergyRatioOutput")
    },
    {
        "name": "HighFreqNoiseRatio",
        "feature_dir": os.path.join(data_dir, "HighFreqNoiseRatioOutput")
    },
    {
        "name": "CPP",
        "feature_dir": os.path.join(data_dir, "CPP_Output")
    }
]


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


def build_score_maps(df):
    id_col = detect_id_column(df)
    id_series = df[id_col].astype(str)
    id_norm = id_series.apply(normalize_id)
    score_cols = [c for c in df.columns if c != id_col]
    maps = {}
    for col in score_cols:
        scores = pd.to_numeric(df[col], errors="coerce")
        if scores.notna().sum() == 0:
            continue
        mapping = {}
        for sid, score in zip(id_norm, scores):
            if not sid:
                continue
            if pd.isna(score):
                continue
            mapping[sid] = int(round(float(score)))
        if mapping:
            maps[str(col)] = mapping
    return maps


def build_feature_medians(feature_dir, score_map):
    vals = {}
    if not os.path.isdir(feature_dir):
        return vals
    files = [f for f in os.listdir(feature_dir) if f.lower().endswith(".csv")]
    files.sort()
    for f in files:
        base_id = normalize_id(os.path.splitext(f)[0])
        if base_id not in score_map:
            continue
        series = load_series(os.path.join(feature_dir, f))
        if series is None:
            continue
        vals[base_id] = float(np.median(series))
    return vals


def build_dataset(score_map):
    feature_values = {}
    for feat in features:
        feature_values[feat["name"]] = build_feature_medians(feat["feature_dir"], score_map)
    keys = None
    for vals in feature_values.values():
        keys = set(vals.keys()) if keys is None else keys & set(vals.keys())
    if not keys:
        return pd.DataFrame()
    rows = []
    for sid in sorted(keys):
        row = {"sample_id": sid, "score": score_map[sid]}
        for feat in features:
            row[feat["name"]] = feature_values[feat["name"]][sid]
        rows.append(row)
    return pd.DataFrame(rows)


def build_result_table(result, feature_names):
    table = pd.DataFrame({
        "特征名称": result.params.index,
        "回归系数": result.params.values,
        "标准误": result.bse.values,
        "Z值": result.tvalues.values,
        "P值": result.pvalues.values,
        "优势比OR": np.exp(result.params.values),
        "OR_95%CI下限": np.exp(result.conf_int()[0]),
        "OR_95%CI上限": np.exp(result.conf_int()[1])
    })
    table = table[table["特征名称"].isin(feature_names)].reset_index(drop=True)
    table["影响度"] = table["回归系数"].abs()
    return table


def run_ordered_model(df, out_dir):
    feature_names = [f["name"] for f in features]
    X_raw = df[feature_names].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw.values)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

    label_values = sorted(df["score"].unique().tolist())
    y = pd.Categorical(df["score"], categories=label_values, ordered=True)
    model = OrderedModel(y, X_scaled, distr="probit")
    result = model.fit(method="bfgs", maxiter=10000, disp=False)

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "dataset_9features.csv"), index=False, encoding="utf-8-sig")
    X_scaled.to_csv(os.path.join(out_dir, "dataset_9features_scaled.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(out_dir, "ordinal_regression_summary.txt"), "w", encoding="utf-8") as f:
        f.write(result.summary().as_text())
    table = build_result_table(result, feature_names)
    table = table.sort_values(by="影响度", ascending=False)
    table.to_csv(os.path.join(out_dir, "ordinal_regression_coefficients.csv"), index=False, encoding="utf-8-sig")


def main():
    os.makedirs(out_root, exist_ok=True)
    df_scores = load_score_matrix(score_path)
    if df_scores.empty:
        raise RuntimeError("评分矩阵为空，无法进行分析")
    score_maps = build_score_maps(df_scores)
    for technique, score_map in score_maps.items():
        df = build_dataset(score_map)
        if df.empty:
            continue
        out_dir = os.path.join(out_root, technique)
        run_ordered_model(df, out_dir)
    print(f"✅ 9参数序数回归已输出到：{out_root}")


if __name__ == "__main__":
    main()
