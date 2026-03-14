"""
Visualization Generator for Project 006: Unsupervised Discovery
Generates 5 publication-quality figures.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ASSETS = Path(__file__).parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)
DATA   = Path(__file__).parent.parent / "data"

plt.rcParams.update({'figure.dpi': 150, 'font.size': 10})

# ── Load data ──────────────────────────────────────────────────────────────────
df_sw  = pd.read_csv(DATA / "smartwatch_activity.csv")
df_wf  = pd.read_csv(DATA / "wafer_defect_patterns.csv")

feat_sw = ['steps_per_min','heart_rate_bpm','accelerometer_magnitude',
           'skin_temperature_c','calories_per_min','gyroscope_magnitude']
feat_wf = ['x_coord','y_coord','defect_density','particle_size_um',
           'electrical_fail_rate','distance_from_center']

X_sw = df_sw[feat_sw].values
X_wf = df_wf[feat_wf].values

scaler_sw = StandardScaler()
X_sw_sc = scaler_sw.fit_transform(X_sw)
scaler_wf = StandardScaler()
X_wf_sc = scaler_wf.fit_transform(X_wf)

# ── Figure 1: K-Means Elbow + Silhouette ──────────────────────────────────────
print("Generating Fig 1: Elbow + Silhouette...")
k_range = range(2, 11)
inertias, silhouettes = [], []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_sw_sc)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_sw_sc, labels, sample_size=1000))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.plot(list(k_range), inertias, 'o-', color='#1565C0', lw=2, markersize=7)
ax.axvline(6, color='#F44336', linestyle='--', lw=2, label='Optimal k=6')
ax.set_xlabel('Number of Clusters (k)'); ax.set_ylabel('Inertia (Within-cluster SSE)')
ax.set_title('Elbow Method: Finding Optimal k\n(Smartwatch Activity Data)', fontweight='bold')
ax.legend()

ax = axes[1]
ax.plot(list(k_range), silhouettes, 's-', color='#2E7D32', lw=2, markersize=7)
ax.axvline(6, color='#F44336', linestyle='--', lw=2, label='Optimal k=6')
ax.set_xlabel('Number of Clusters (k)'); ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score: Cluster Separation Quality\n(Higher = better defined clusters)', fontweight='bold')
ax.legend()

plt.suptitle('How Many Clusters? Elbow Method + Silhouette Score', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(ASSETS / "fig1_elbow_silhouette.png", bbox_inches='tight')
plt.close()
print("  fig1 saved")

# ── Figure 2: K-Means Discovered Clusters vs True Activities (2D PCA) ─────────
print("Generating Fig 2: K-Means vs True Labels...")
km6 = KMeans(n_clusters=6, random_state=42, n_init=10)
labels_km = km6.fit_predict(X_sw_sc)

pca2 = PCA(n_components=2, random_state=42)
X_sw_2d = pca2.fit_transform(X_sw_sc)

palette = ['#E91E63','#1565C0','#FF6F00','#2E7D32','#6A1B9A','#00838F']
activity_colors = {
    'Running': '#E91E63', 'Walking': '#1565C0', 'Cycling': '#FF6F00',
    'Sleeping': '#2E7D32', 'Desk Work': '#6A1B9A', 'Gym': '#00838F'
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for i, color in enumerate(palette):
    mask = labels_km == i
    ax.scatter(X_sw_2d[mask, 0], X_sw_2d[mask, 1], c=color, s=10, alpha=0.5, label=f'Cluster {i}')
centers_2d = pca2.transform(km6.cluster_centers_)
ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', s=150, marker='X', zorder=5, label='Centroids')
ax.set_title('K-Means Discovered Clusters (k=6)\n(No labels used — algorithm discovers structure)',
             fontweight='bold')
ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]:.1%})'); ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]:.1%})')
ax.legend(fontsize=8)

ax = axes[1]
for activity, color in activity_colors.items():
    mask = df_sw['true_activity'] == activity
    ax.scatter(X_sw_2d[mask, 0], X_sw_2d[mask, 1], c=color, s=10, alpha=0.5, label=activity)
ax.set_title('True Activity Labels (Ground Truth)\nFor comparison only — not used during clustering',
             fontweight='bold')
ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]:.1%})'); ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]:.1%})')
ax.legend(fontsize=8)

ari = adjusted_rand_score(df_sw['true_activity'], labels_km)
plt.suptitle(f'K-Means Discovery vs Ground Truth | Adjusted Rand Index = {ari:.3f}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(ASSETS / "fig2_kmeans_vs_true.png", bbox_inches='tight')
plt.close()
print("  fig2 saved")

# ── Figure 3: PCA Explained Variance + 3D Biplot ──────────────────────────────
print("Generating Fig 3: PCA Analysis...")
pca_full = PCA(random_state=42)
pca_full.fit(X_sw_sc)

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
bars = ax1.bar(range(1, len(feat_sw)+1), pca_full.explained_variance_ratio_,
               color='#1565C0', alpha=0.7, label='Individual')
ax1.plot(range(1, len(feat_sw)+1), cumvar, 'o-', color='#F44336', lw=2, label='Cumulative')
ax1.axhline(0.95, color='green', linestyle='--', alpha=0.7, label='95% threshold')
ax1.set_xlabel('Principal Component'); ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('PCA: Explained Variance\n(How much information each PC captures)', fontweight='bold')
ax1.legend(); ax1.set_xticks(range(1, len(feat_sw)+1))

ax2 = fig.add_subplot(122, projection='3d')
pca3 = PCA(n_components=3, random_state=42)
X_sw_3d = pca3.fit_transform(X_sw_sc)
for activity, color in activity_colors.items():
    mask = df_sw['true_activity'] == activity
    ax2.scatter(X_sw_3d[mask, 0], X_sw_3d[mask, 1], X_sw_3d[mask, 2],
                c=color, s=8, alpha=0.4, label=activity)
ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2'); ax2.set_zlabel('PC3')
ax2.set_title('3D PCA Projection\n(6 features → 3 components)', fontweight='bold')
ax2.legend(fontsize=7, loc='upper left')

plt.suptitle('Principal Component Analysis: Dimensionality Reduction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(ASSETS / "fig3_pca_analysis.png", bbox_inches='tight')
plt.close()
print("  fig3 saved")

# ── Figure 4: DBSCAN on Wafer Defect Patterns ─────────────────────────────────
print("Generating Fig 4: DBSCAN Wafer Defects...")
# Use only x, y, defect_density for spatial clustering
X_wf_spatial = df_wf[['x_coord','y_coord','defect_density']].values
scaler_sp = StandardScaler()
X_wf_sp_sc = scaler_sp.fit_transform(X_wf_spatial)

dbscan = DBSCAN(eps=0.25, min_samples=15)
labels_db = dbscan.fit_predict(X_wf_sp_sc)
n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = (labels_db == -1).sum()

palette_db = plt.cm.tab10(np.linspace(0, 1, max(n_clusters_db, 1)))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
unique_labels = sorted(set(labels_db))
for lbl in unique_labels:
    mask = labels_db == lbl
    if lbl == -1:
        ax.scatter(df_wf.loc[mask, 'x_coord'], df_wf.loc[mask, 'y_coord'],
                   c='lightgray', s=5, alpha=0.3, label=f'Noise ({mask.sum()})')
    else:
        ax.scatter(df_wf.loc[mask, 'x_coord'], df_wf.loc[mask, 'y_coord'],
                   c=[palette_db[lbl % 10]], s=10, alpha=0.6, label=f'Cluster {lbl}')
circle = plt.Circle((0, 0), 1.0, fill=False, color='black', lw=1.5, linestyle='--', alpha=0.5)
ax.add_patch(circle)
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect('equal')
ax.set_title(f'DBSCAN: Discovered {n_clusters_db} Defect Patterns\n({n_noise} noise points in gray)',
             fontweight='bold')
ax.set_xlabel('X Coordinate (normalized)'); ax.set_ylabel('Y Coordinate (normalized)')
ax.legend(fontsize=7, loc='upper right')

ax = axes[1]
true_pattern_colors = {
    'Edge Ring': '#E91E63', 'Center Spot': '#1565C0', 'Scratch': '#FF6F00',
    'Random': 'lightgray', 'Donut': '#2E7D32', 'Local Cluster': '#6A1B9A'
}
for pattern, color in true_pattern_colors.items():
    mask = df_wf['true_pattern'] == pattern
    ax.scatter(df_wf.loc[mask, 'x_coord'], df_wf.loc[mask, 'y_coord'],
               c=color, s=5, alpha=0.4, label=pattern)
circle2 = plt.Circle((0, 0), 1.0, fill=False, color='black', lw=1.5, linestyle='--', alpha=0.5)
ax.add_patch(circle2)
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect('equal')
ax.set_title('True Defect Patterns (Ground Truth)\nFor comparison only', fontweight='bold')
ax.set_xlabel('X Coordinate (normalized)'); ax.set_ylabel('Y Coordinate (normalized)')
ax.legend(fontsize=7, loc='upper right')

plt.suptitle('DBSCAN: Density-Based Wafer Defect Pattern Discovery\n(No labels, no k required)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(ASSETS / "fig4_dbscan_wafer.png", bbox_inches='tight')
plt.close()
print("  fig4 saved")

# ── Figure 5: K-Means vs DBSCAN Comparison ────────────────────────────────────
print("Generating Fig 5: Algorithm Comparison...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: K-Means on different shaped data
np.random.seed(42)
# Blobs (K-Means wins)
from sklearn.datasets import make_blobs, make_moons, make_circles
X_blobs, y_blobs = make_blobs(n_samples=400, centers=3, random_state=42)
# Moons (DBSCAN wins)
X_moons, y_moons = make_moons(n_samples=400, noise=0.08, random_state=42)
# Circles (DBSCAN wins)
X_circles, y_circles = make_circles(n_samples=400, noise=0.05, factor=0.5, random_state=42)

datasets = [(X_blobs, 'Blobs'), (X_moons, 'Moons'), (X_circles, 'Circles')]
km_colors = ['#E91E63','#1565C0','#FF6F00','#2E7D32']
db_colors = ['#E91E63','#1565C0','lightgray']

for col, (X_data, name) in enumerate(datasets):
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_data)

    # K-Means
    km = KMeans(n_clusters=2 if name != 'Blobs' else 3, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_sc)
    ax = axes[0, col]
    for lbl in np.unique(km_labels):
        mask = km_labels == lbl
        ax.scatter(X_sc[mask, 0], X_sc[mask, 1], c=km_colors[lbl], s=15, alpha=0.7)
    ax.set_title(f'K-Means on {name}', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    # DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10)
    db_labels = db.fit_predict(X_sc)
    ax = axes[1, col]
    for lbl in sorted(set(db_labels)):
        mask = db_labels == lbl
        color = 'lightgray' if lbl == -1 else km_colors[lbl % 4]
        ax.scatter(X_sc[mask, 0], X_sc[mask, 1], c=color, s=15, alpha=0.7)
    n_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    ax.set_title(f'DBSCAN on {name} ({n_db} clusters)', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

axes[0, 0].set_ylabel('K-Means', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('DBSCAN', fontsize=12, fontweight='bold')

plt.suptitle('K-Means vs DBSCAN: When to Use Which Algorithm\n'
             'K-Means: convex clusters | DBSCAN: arbitrary shapes, handles noise',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(ASSETS / "fig5_kmeans_vs_dbscan.png", bbox_inches='tight')
plt.close()
print("  fig5 saved")

print("\nAll 5 figures generated successfully.")
