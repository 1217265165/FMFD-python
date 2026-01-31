"""
visualize_results.py

用途:
 - 从 feature_extraction 的输出（固定路径 run_test_features_enhanced.csv）读取数据，生成可视化：
    * 模块置信热图 (module_meta)
    * 特征相关性与分布
    * 若存在标签: RandomForest 特征重要性柱状图
    * PCA + 聚类 (DBSCAN, KMeans) 可视化
    * 每簇的特征摘要 CSV
 - 输出文件夹: ./viz_outputs/
 - 直接运行示例:


说明:
 - 此版本已固定输入路径为:

 - 改进点:
   * 修复 dendrogram 的输入（对观测矩阵直接做 linkage，先 PCA 降维）
   * 自动估计 DBSCAN eps（k-distance percentile）并打印建议值
   * 移除近似恒定特征（VarianceThreshold）
   * 计算聚类质量指标（silhouette, calinski-harabasz）
   * 兼容没有 seaborn 的环境并设置中文字体回退
   * 所有保存图像使用 bbox_inches='tight'，dpi=200
"""
import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Try to use seaborn style if seaborn is installed; otherwise fallback to matplotlib built-in
try:
    import seaborn as sns
    sns.set_style('whitegrid')
    sns.set_context('talk')
except Exception:
    plt.style.use('ggplot')
    sns = None

# ensure chinese font (Windows)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 150

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import VarianceThreshold
from scipy.cluster.hierarchy import linkage, dendrogram

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

@dataclass
class Table:
    columns: List[str]
    rows: List[Dict[str, object]]

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def shape(self) -> tuple:
        return (len(self.rows), len(self.columns))

    def column_values(self, name: str) -> np.ndarray:
        vals: List[float] = []
        for row in self.rows:
            val = row.get(name)
            if val is None or val == "":
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(val))
                except (ValueError, TypeError):
                    vals.append(np.nan)
        return np.array(vals, dtype=float)

    def matrix(self, cols: Sequence[str]) -> np.ndarray:
        return np.vstack([self.column_values(col) for col in cols]).T if cols else np.empty((len(self), 0))

    def add_column(self, name: str, values: Sequence[object]) -> None:
        if name not in self.columns:
            self.columns.append(name)
        for row, val in zip(self.rows, values):
            row[name] = val

    def to_csv(self, path: str) -> None:
        if not self.rows:
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)


def load_feature_file(path: str) -> Table:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, object]] = []
        for row in reader:
            rows.append(dict(row))
    return Table(columns=list(reader.fieldnames or []), rows=rows)

def detect_module_meta_columns(df: Table):
    return [c for c in df.columns if c.startswith('module_')]

def top_feature_correlation(df: Table, features, topn=40, out_dir='viz_outputs'):
    ensure_dir(out_dir)
    feats = [c for c in features if np.isfinite(df.column_values(c)).any()]
    if len(feats) == 0:
        print("[WARN] 没有可用的数值特征进行相关性分析")
        return []
    matrix = df.matrix(feats)
    corr = np.corrcoef(np.nan_to_num(matrix, nan=0.0), rowvar=False)
    corr_abs = np.abs(corr)
    mean_corr = np.mean(corr_abs, axis=0)
    order = np.argsort(mean_corr)[::-1]
    top = [feats[idx] for idx in order[:topn]]
    plt.figure(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(np.corrcoef(np.nan_to_num(df.matrix(top), nan=0.0), rowvar=False), cmap='vlag', center=0, annot=False)
    else:
        plt.imshow(np.corrcoef(np.nan_to_num(df.matrix(top), nan=0.0), rowvar=False), cmap='bwr', aspect='auto')
    plt.title('Top features correlation heatmap', fontsize=14)
    plt.tight_layout()
    fp = os.path.join(out_dir, 'top_feature_correlation_heatmap.png')
    plt.savefig(fp, dpi=200, bbox_inches='tight')
    plt.close()
    print("[INFO] saved", fp)
    return top

def plot_module_meta_heatmap(df: Table, module_cols, out_dir='viz_outputs'):
    ensure_dir(out_dir)
    if len(module_cols) == 0:
        print("[WARN] 无 module_meta 列（module_ 前缀），跳过模块置信热图")
        return
    mdf = df.matrix(module_cols)
    maxrows = 200
    if mdf.shape[0] > maxrows:
        rng = np.random.default_rng(42)
        idx = rng.choice(mdf.shape[0], size=maxrows, replace=False)
        mdf = mdf[idx]
    plt.figure(figsize=(12, max(4, mdf.shape[0]*0.06)))
    if sns is not None:
        sns.heatmap(mdf.T, cmap='YlGnBu', cbar_kws={'label': 'belief'}, vmin=0, vmax=1)
    else:
        plt.imshow(mdf.T, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
    plt.xlabel('sample index (subset)', fontsize=12)
    plt.ylabel('module', fontsize=12)
    plt.title('Module belief heatmap (rows: modules, cols: samples)', fontsize=14)
    plt.tight_layout()
    fp = os.path.join(out_dir, 'module_belief_heatmap.png')
    plt.savefig(fp, dpi=200, bbox_inches='tight')
    plt.close()
    print("[INFO] saved", fp)

def cluster_and_plot(df: Table, feature_cols, out_dir='viz_outputs', n_clusters=4):
    ensure_dir(out_dir)
    if len(feature_cols) == 0:
        print("[WARN] 没有用于聚类的特征列")
        return df
    X = np.nan_to_num(df.matrix(feature_cols), nan=0.0)
    if X.shape[0] == 0:
        print("[WARN] 无样本用于聚类，跳过聚类分析")
        return df
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # remove near-constant features
    try:
        vt = VarianceThreshold(threshold=1e-6)
        Xv = vt.fit_transform(Xs)
        if Xv.shape[1] < Xs.shape[1]:
            removed = Xs.shape[1] - Xv.shape[1]
            print(f"[INFO] VarianceThreshold removed {removed} near-constant features")
            Xs = Xv
    except Exception as e:
        print("[WARN] VarianceThreshold error:", e)

    print(f"[INFO] clustering: samples={Xs.shape[0]}, features={Xs.shape[1]}")
    nan_ratio = np.isnan(Xs).mean()
    print(f"[INFO] Nan ratio in features: {nan_ratio:.4f}")
    variances = np.var(Xs, axis=0)
    print(f"[INFO] features variance: min={variances.min():.3e}, mean={variances.mean():.3e}, max={variances.max():.3e}")

    # PCA for visualization
    pca_vis = PCA(n_components=2, random_state=42)
    try:
        Xp = pca_vis.fit_transform(Xs)
    except Exception:
        Xp = Xs[:, :2] if Xs.shape[1] >= 2 else np.hstack([Xs, np.zeros((Xs.shape[0], max(0,2-Xs.shape[1])))])

    # KMeans clustering with silhouette-based suggestion for k
    best_k = min(n_clusters, Xs.shape[0])
    try:
        k_max = min(8, max(3, int(np.sqrt(Xs.shape[0]))))
        k_range = list(range(2, min(k_max, Xs.shape[0]) + 1))
        best_score = -1.0
        if Xs.shape[0] >= 3 and k_range:
            for k in k_range:
                km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
                if len(set(km_tmp.labels_)) > 1:
                    try:
                        s = silhouette_score(Xs, km_tmp.labels_)
                    except Exception:
                        s = -1.0
                    if s > best_score:
                        best_score = s
                        best_k = k
            print(f"[INFO] KMeans silhouette best_k={best_k}, score={best_score:.4f}")
        else:
            print("[INFO] 样本过少，跳过 silhouette 搜索")
    except Exception as e:
        print("[WARN] KMeans silhouette search failed:", e)

    if best_k <= 1 or Xs.shape[0] <= 1:
        labels_km = np.zeros(Xs.shape[0], dtype=int)
    else:
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(Xs)
        labels_km = km.labels_

    # DBSCAN: estimate eps via k-distance heuristic
    try:
        if Xs.shape[0] > 1:
            k_db = min(5, Xs.shape[0])
            nbrs = NearestNeighbors(n_neighbors=k_db).fit(Xs)
            distances, _ = nbrs.kneighbors(Xs)
            k_distances = np.sort(distances[:, -1])
            eps_guess = float(np.percentile(k_distances, 90))
            print(f"[INFO] DBSCAN eps_guess (90th percentile) = {eps_guess:.4f}")
            db = DBSCAN(eps=eps_guess, min_samples=5).fit(Xs)
            labels_db = db.labels_
            print(f"[INFO] DBSCAN found clusters (unique labels incl -1): {np.unique(labels_db)}")
        else:
            labels_db = np.zeros(Xs.shape[0], dtype=int)
            print("[INFO] 样本过少，跳过 DBSCAN 聚类")
    except Exception as e:
        print("[WARN] DBSCAN estimation failed:", e)
        labels_db = np.array([-1]*Xs.shape[0])

    # plot PCA scatter by KMeans
    plt.figure(figsize=(8,6))
    palette = (sns.color_palette('tab10', best_k) if sns is not None else None)
    for lab in np.unique(labels_km):
        mask = labels_km == lab
        color = palette[int(lab)%len(palette)] if palette is not None else None
        plt.scatter(Xp[mask,0], Xp[mask,1], s=40, color=color, alpha=0.85, label=f'KMeans_{lab}')
    plt.title('PCA projection colored by KMeans', fontsize=14)
    plt.xlabel('PC1', fontsize=12); plt.ylabel('PC2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    fp = os.path.join(out_dir, 'pca_kmeans.png')
    plt.savefig(fp, dpi=200, bbox_inches='tight')
    plt.close()
    print("[INFO] saved", fp)

    # plot PCA scatter by DBSCAN
    plt.figure(figsize=(8,6))
    uniq = np.unique(labels_db)
    for lab in uniq:
        mask = labels_db == lab
        lab_name = f"DB_{lab}"
        if palette is not None:
            col = 'gray' if lab == -1 else palette[int(abs(lab))%len(palette)]
        else:
            col = None
        plt.scatter(Xp[mask,0], Xp[mask,1], s=40, color=col, alpha=0.85, label=lab_name)
    plt.title('PCA projection colored by DBSCAN', fontsize=14)
    plt.xlabel('PC1', fontsize=12); plt.ylabel('PC2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    fp = os.path.join(out_dir, 'pca_dbscan.png')
    plt.savefig(fp, dpi=200, bbox_inches='tight')
    plt.close()
    print("[INFO] saved", fp)

    # hierarchical dendrogram on a reduced representation (subset)
    try:
        subset_n = min(200, Xs.shape[0])
        if subset_n < 4:
            raise ValueError("Not enough samples for dendrogram")
        X_subset = Xs[:subset_n]
        n_comp = min(10, max(2, X_subset.shape[1]//2))
        pca_loc = PCA(n_components=n_comp, random_state=42)
        X_red = pca_loc.fit_transform(X_subset)
        Z = linkage(X_red, method='ward')
        plt.figure(figsize=(10, 4))
        dendrogram(Z, no_labels=True, color_threshold=None, truncate_mode='lastp', p=40)
        plt.title('Hierarchical clustering dendrogram (subset, PCA reduced)', fontsize=14)
        plt.tight_layout()
        fp = os.path.join(out_dir, 'dendrogram_subset.png')
        plt.savefig(fp, dpi=200, bbox_inches='tight')
        plt.close()
        print("[INFO] saved", fp)
    except Exception as e:
        print("[WARN] dendrogram failed:", e)

    # cluster quality metrics
    try:
        if len(set(labels_km)) > 1 and Xs.shape[0] > len(set(labels_km)):
            sil = silhouette_score(Xs, labels_km)
            ch = calinski_harabasz_score(Xs, labels_km)
            print(f"[INFO] KMeans silhouette={sil:.4f}, calinski-harabasz={ch:.4f}")
    except Exception as e:
        print("[WARN] silhouette/ch score failed:", e)

    # attach cluster labels back to df and produce cluster summary
    df_out = Table(columns=list(df.columns), rows=[dict(r) for r in df.rows])
    df_out.add_column('cluster_km', labels_km)
    df_out.add_column('cluster_db', labels_db)
    summary_rows: List[Dict[str, object]] = []
    for col in feature_cols:
        col_vals = df_out.column_values(col)
        for cluster_id in np.unique(labels_km):
            mask = labels_km == cluster_id
            mean_val = float(np.nanmean(col_vals[mask])) if np.any(mask) else 0.0
            summary_rows.append({"feature": col, "cluster_km": int(cluster_id), "mean": mean_val})
    summary_path = os.path.join(out_dir, 'cluster_kmeans_feature_mean.csv')
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "cluster_km", "mean"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print("[INFO] saved cluster summary csv ->", os.path.join(out_dir, 'cluster_kmeans_feature_mean.csv'))
    return df_out

def feature_importance_with_label(df: Table, feature_cols, label_col, out_dir='viz_outputs', topn=30):
    ensure_dir(out_dir)
    if label_col not in df.columns:
        print(f"[WARN] label_col {label_col} not in df, skip features importance")
        return None
    X = np.nan_to_num(df.matrix(feature_cols), nan=0.0)
    y = df.column_values(label_col)
    if y.dtype.kind in {'U','O','S'}:
        uniq, y_enc = np.unique(y.astype(str), return_inverse=True)
    else:
        y_enc = y
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y_enc)
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:topn]
    feat_sorted = [feature_cols[i] for i in idx]
    imps_sorted = importances[idx]
    plt.figure(figsize=(8, max(4, len(feat_sorted)*0.25)))
    if sns is not None:
        sns.barplot(x=imps_sorted, y=feat_sorted, palette='viridis')
    else:
        plt.barh(feat_sorted, imps_sorted)
    plt.xlabel('Importance', fontsize=12); plt.ylabel('Feature', fontsize=12)
    plt.title('RandomForest features importance (top {})'.format(len(feat_sorted)), fontsize=14)
    plt.tight_layout()
    fp = os.path.join(out_dir, 'feature_importance_rf.png')
    plt.savefig(fp, dpi=200, bbox_inches='tight')
    plt.close()
    print("[INFO] saved", fp)
    return Table(columns=['features', 'importance'], rows=[{"features": f, "importance": float(i)} for f, i in zip(feat_sorted, imps_sorted)])

# --------------------------
# Fixed-input entrypoint
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize detection results")
    parser.add_argument("--run_dir", default="Output/runs/run_001", help="run 目录 (默认 Output/runs/run_001)")
    parser.add_argument("--input_csv", default=None, help="特征 CSV (默认 run_dir/tables/detection_results.csv)")
    parser.add_argument("--out_dir", default=None, help="输出目录 (默认 run_dir/plots)")
    parser.add_argument("--label_col", default=None, help="标签列 (可选)")
    args = parser.parse_args()

    input_csv = args.input_csv or os.path.join(args.run_dir, "tables", "detection_results.csv")
    outdir = args.out_dir or os.path.join(args.run_dir, "plots")
    prefix = "run_test"
    n_clusters = 4
    label_col = args.label_col

    ensure_dir(outdir)
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"features file not found: {input_csv}")

    df = load_feature_file(input_csv)
    print(f"[INFO] Loaded features file: {input_csv}  rows={len(df)} cols={len(df.columns)}")
    module_cols = detect_module_meta_columns(df)
    print("[INFO] detected module columns:", module_cols)

    exclude_prefixes = ['module_', 'seq_index', 'timestamp', 'rep']
    candidate_features = [c for c in df.columns if c not in module_cols and not any(c.startswith(pref) for pref in exclude_prefixes)]
    numeric_features = [c for c in candidate_features if np.isfinite(df.column_values(c)).any()]
    print(f"[INFO] numeric features count: {len(numeric_features)}")

    top_feats = top_feature_correlation(df, numeric_features, topn=40, out_dir=outdir)
    plot_module_meta_heatmap(df, module_cols, out_dir=outdir)

    use_feats = top_feats if top_feats is not None and len(top_feats) > 0 else numeric_features
    df_with_clusters = cluster_and_plot(df, use_feats, out_dir=outdir, n_clusters=n_clusters)

    if label_col:
        fi = feature_importance_with_label(df_with_clusters, use_feats, label_col, out_dir=outdir, topn=40)
        if fi is not None:
            fi.to_csv(os.path.join(outdir, 'feature_importances_supervised.csv'))
            print("[INFO] saved supervised features importances csv")

    df_with_clusters.to_csv(os.path.join(outdir, f"{prefix}_with_clusters.csv"))
    print("[INFO] saved augmented dataframe with clusters ->", os.path.join(outdir, f"{prefix}_with_clusters.csv"))
