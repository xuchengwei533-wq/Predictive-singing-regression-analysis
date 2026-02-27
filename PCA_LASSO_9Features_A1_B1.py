import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV


root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir, "Chest new0206", "Chest new0206")
score_path = os.path.join(root_dir, "打分Chest_new0206_scores_matrix.xlsx")
out_root = os.path.join(root_dir, "PCA_LASSO_9Features_Output_A1_B1")

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
            mapping[sid] = float(score)
        if mapping:
            maps[str(col)] = mapping
    return maps


def build_feature_medians(feature_dir, score_map, allowed_suffixes):
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


def build_dataset(score_map, allowed_suffixes):
    feature_values = {}
    for feat in features:
        feature_values[feat["name"]] = build_feature_medians(feat["feature_dir"], score_map, allowed_suffixes)
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


def save_pca_results(df, out_dir):
    feature_names = [f["name"] for f in features]
    X = df[feature_names].values
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    pca = PCA(n_components=len(feature_names), random_state=42)
    pca.fit(Xz)
    comps = pd.DataFrame(
        pca.components_,
        columns=feature_names,
        index=[f"PC{i+1}" for i in range(pca.components_.shape[0])]
    )
    var_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "explained_variance_ratio": pca.explained_variance_ratio_
    })
    os.makedirs(out_dir, exist_ok=True)
    comps.to_csv(os.path.join(out_dir, "pca_loadings.csv"), encoding="utf-8-sig")
    var_df.to_csv(os.path.join(out_dir, "pca_explained_variance.csv"), index=False, encoding="utf-8-sig")

    pc1 = comps.loc["PC1"]
    pc1_sorted = pc1.reindex(pc1.abs().sort_values(ascending=False).index)
    formula_lines = ["PC1 = " + " + ".join([f"{pc1_sorted[k]:.4f}*{k}" for k in pc1_sorted.index])]
    with open(os.path.join(out_dir, "pc1_formula.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(formula_lines))

    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=180)
    im = ax.imshow(comps.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(comps.index)))
    ax.set_yticklabels(comps.index)
    for i in range(comps.shape[0]):
        for j in range(comps.shape[1]):
            ax.text(j, i, f"{comps.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pca_loading_heatmap.png"))
    plt.close(fig)


def save_lasso_results(df, out_dir):
    feature_names = [f["name"] for f in features]
    X = df[feature_names].values
    y = df["score"].values.astype(float)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    model = LassoCV(cv=5, random_state=42, n_alphas=50)
    model.fit(Xz, y)
    coefs = pd.Series(model.coef_, index=feature_names)
    coef_df = pd.DataFrame({
        "feature": coefs.index,
        "coef": coefs.values,
        "abs_coef": np.abs(coefs.values)
    }).sort_values(by="abs_coef", ascending=False)
    coef_df["selected"] = coef_df["abs_coef"] >= 0.01
    os.makedirs(out_dir, exist_ok=True)
    coef_df.to_csv(os.path.join(out_dir, "lasso_coefficients.csv"), index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=180)
    ax.bar(coef_df["feature"], coef_df["coef"], color="#1f77b4")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("LASSO Coefficient")
    ax.set_xticks(np.arange(len(coef_df["feature"])))
    ax.set_xticklabels(coef_df["feature"], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lasso_coefficients.png"))
    plt.close(fig)


def run_group(score_maps, group_name, allowed_suffixes):
    for technique, score_map in score_maps.items():
        out_dir = os.path.join(out_root, group_name, technique)
        df = build_dataset(score_map, allowed_suffixes)
        if df.empty:
            continue
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, "dataset_9features.csv"), index=False, encoding="utf-8-sig")
        save_pca_results(df, out_dir)
        save_lasso_results(df, out_dir)


def main():
    os.makedirs(out_root, exist_ok=True)
    df_scores = load_score_matrix(score_path)
    if df_scores.empty:
        raise RuntimeError("评分矩阵为空，无法进行分析")
    score_maps = build_score_maps(df_scores)
    run_group(score_maps, "A1", {"A", "1"})
    run_group(score_maps, "B1", {"B", "1"})
    print(f"✅ A1/B1 的 PCA 与 LASSO 已输出到：{out_root}")


if __name__ == "__main__":
    main()
