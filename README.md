# Post 006 — Unsupervised Discovery: K-Means, DBSCAN, PCA

**AI Engineering Lab Series** | Era 1: Classic Machine Learning

---

## Overview

This project demonstrates the three foundational unsupervised learning techniques — **K-Means**, **DBSCAN**, and **PCA** — applied to two real-world discovery problems. The key insight: when you have no labels, unsupervised learning finds hidden structure in your data.

| Concept | Description |
|---|---|
| **K-Means** | Partitions data into k clusters by minimizing within-cluster variance (inertia) |
| **Elbow Method** | Finds optimal k by plotting inertia vs k and looking for the "elbow" |
| **Silhouette Score** | Measures cluster quality: how similar each point is to its own cluster vs others |
| **DBSCAN** | Density-based clustering — finds arbitrary shapes, handles noise natively |
| **eps and min_samples** | DBSCAN parameters: neighborhood radius and minimum points to form a core point |
| **PCA** | Reduces dimensionality while preserving maximum variance |
| **Explained Variance** | How much information each principal component captures |

---

## Datasets

### Dataset A: Smartwatch Activity Clustering
Discovers daily activities from wearable sensor data — no labels provided.

| Feature | Description |
|---|---|
| `steps_per_min` | Step cadence (steps/min) — high for running, near zero for sleeping |
| `heart_rate_bpm` | Heart rate (bpm) — elevated during exercise |
| `accelerometer_magnitude` | Movement intensity — high for gym, low for desk work |
| `skin_temperature_c` | Skin temperature (°C) — rises during exercise |
| `calories_per_min` | Caloric burn rate — highest during running and gym |
| `gyroscope_magnitude` | Rotational motion — high for running, near zero for sleeping |
| `true_activity` | Ground truth label (kept for evaluation only, not used in clustering) |

- **Rows:** 4,000 | **Hidden clusters:** 6 (Running, Walking, Cycling, Sleeping, Desk Work, Gym)

### Dataset B: Wafer Defect Pattern Clustering (Post-Silicon Validation)
Discovers spatial defect patterns on silicon wafers — no labels provided.

| Feature | Description |
|---|---|
| `x_coord` | Normalized X position on wafer (-1 to +1) |
| `y_coord` | Normalized Y position on wafer (-1 to +1) |
| `defect_density` | Local defect density (0-1) |
| `particle_size_um` | Average particle size (micrometers) |
| `electrical_fail_rate` | Fraction of dies failing electrical test at this location |
| `distance_from_center` | Distance from wafer center (0-1) |
| `true_pattern` | Ground truth pattern (for evaluation only) |

- **Rows:** 8,000 | **Hidden patterns:** 6 (Edge Ring, Center Spot, Scratch, Random, Donut, Local Cluster)

---

## Project Structure

```
006_unsupervised_discovery/
├── data/
│   ├── smartwatch_activity.csv
│   └── wafer_defect_patterns.csv
├── notebooks/
│   ├── 01_unsupervised_smartwatch.ipynb
│   └── 02_unsupervised_wafer.ipynb
├── src/
│   ├── data_generator.py
│   └── generate_visuals.py
├── assets/
│   ├── fig1_elbow_silhouette.png
│   ├── fig2_kmeans_vs_true.png
│   ├── fig3_pca_analysis.png
│   ├── fig4_dbscan_wafer.png
│   └── fig5_kmeans_vs_dbscan.png
├── PRD.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Key Visualizations

| Figure | Description |
|---|---|
| `fig1` | Elbow method + Silhouette score side-by-side — finding optimal k=6 |
| `fig2` | K-Means discovered clusters vs true activity labels (2D PCA projection) |
| `fig3` | PCA explained variance bar chart + 3D projection of all 6 activities |
| `fig4` | DBSCAN discovering wafer defect patterns vs true patterns (spatial map) |
| `fig5` | K-Means vs DBSCAN on blobs/moons/circles — when to use which algorithm |

---

## Quick Start

```bash
git clone https://github.com/AIML-Engineering-Lab/006_unsupervised_discovery.git
cd 006_unsupervised_discovery
pip install -r requirements.txt
python src/data_generator.py
jupyter notebook notebooks/
```

---

*Part of the [AI Engineering Lab](https://github.com/AIML-Engineering-Lab) series.*
