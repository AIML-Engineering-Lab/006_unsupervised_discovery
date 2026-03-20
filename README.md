# Unsupervised Discovery: K-Means, DBSCAN & PCA

---

## Overview

This project demonstrates **K-Means Clustering**, **DBSCAN**, and **PCA Dimensionality Reduction** applied to two real-world unsupervised learning problems. The key insight: when you have no labels, unsupervised learning finds hidden structure in your data.

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

- **Rows:** 4,000 | **Hidden clusters:** 4 (Running, Gym, Cycling, Sedentary)

### Dataset B: Wafer Defect Pattern Discovery
Discovers spatial defect patterns on semiconductor wafers — no labels provided.

| Feature | Description |
|---|---|
| `x_coord` | Normalized X position on wafer (-1 to +1) |
| `y_coord` | Normalized Y position on wafer (-1 to +1) |
| `defect_density` | Local defect density (0-1) |
| `particle_size_um` | Average particle size (micrometers) |
| `electrical_fail_rate` | Fraction of dies failing electrical test at this location |
| `distance_from_center` | Distance from wafer center (0-1) |
| `true_pattern` | Ground truth pattern (for evaluation only) |

- **Rows:** 8,000 | **Hidden patterns:** 5 (Random, Edge Ring, Center, Scratch, None)

---

## Repository Structure

```
006_unsupervised_discovery/
├── assets/
│   ├── proj1_smartwatch_eda.png                   # Smartwatch: feature distributions + correlations
│   ├── proj1_smartwatch_pca_analysis.png          # Smartwatch: PCA projection + variance
│   ├── proj1_smartwatch_pca_analysis_fig.png      # Smartwatch: extended PCA with component loadings
│   ├── proj1_smartwatch_kmeans_k_selection.png    # Smartwatch: k selection (inertia + silhouette)
│   ├── proj1_smartwatch_elbow_silhouette.png      # Smartwatch: elbow method + silhouette analysis
│   ├── proj1_smartwatch_kmeans_vs_true.png        # Smartwatch: K-Means clusters vs true labels
│   ├── proj1_smartwatch_clustering_comparison.png # Smartwatch: multi-algorithm comparison
│   ├── proj1_smartwatch_3d_pca_clusters.png       # Smartwatch: 3D PCA projection
│   ├── proj1_smartwatch_model_heatmap.png         # Smartwatch: model performance heatmap
│   ├── proj1_smartwatch_flowchart.png             # Smartwatch: AI-generated pipeline flowchart
│   ├── proj2_wafer_eda.png                        # Wafer: feature distributions + spatial scatter
│   ├── proj2_wafer_pca.png                        # Wafer: PCA projection by true pattern
│   ├── proj2_wafer_clustering_comparison.png      # Wafer: multi-algorithm comparison
│   ├── proj2_wafer_dbscan.png                     # Wafer: DBSCAN density-based clustering
│   ├── proj2_wafer_kmeans_vs_dbscan.png           # Wafer: K-Means vs DBSCAN comparison
│   ├── proj2_wafer_3d_defect_map.png              # Wafer: 3D defect spatial map
│   ├── proj2_wafer_3d_dbscan_landscape.png        # Wafer: 3D DBSCAN landscape
│   └── proj2_wafer_flowchart.png                  # Wafer: AI-generated pipeline flowchart
├── data/
│   ├── smartwatch_activity.csv                    # 4,000 smartwatch records (6 features + label)
│   └── wafer_defect_patterns.csv                  # 8,000 wafer inspections (6 features + label)
├── deploy/
│   ├── Dockerfile                                 # Container image for FastAPI server
│   └── docker-compose.yml                         # Single-command deployment
├── docs/
│   ├── Unsupervised_Discovery_Report.html         # Full report with embedded visualizations
│   └── Unsupervised_Discovery_Report.pdf          # Print-ready A4 report
├── models/
│   ├── kmeans_smartwatch.pkl                      # Trained KMeans pipeline (smartwatch k=4)
│   └── kmeans_wafer.pkl                           # Trained KMeans pipeline (wafer k=5)
├── notebooks/
│   ├── 01_unsupervised_smartwatch.ipynb           # Smartwatch EDA, clustering, PCA analysis
│   └── 02_unsupervised_wafer_defects.ipynb        # Wafer EDA, DBSCAN, K-Means comparison
├── src/
│   ├── train.py                                   # Train both KMeans models (DATASETS dict)
│   ├── predict.py                                 # Inference: load model, predict on DataFrame
│   ├── api.py                                     # FastAPI endpoint (/health, /info, /predict)
│   └── data_generator.py                          # Generate synthetic datasets
├── tests/
│   └── test_model.py                              # 4 tests: existence + prediction per model
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Core runtime |
| scikit-learn | ≥ 1.3 | KMeans, DBSCAN, PCA, StandardScaler, silhouette_score |
| pandas | ≥ 2.0 | Data manipulation |
| numpy | ≥ 1.24 | Numerical operations |
| matplotlib / seaborn | ≥ 3.7 / 0.12 | Visualizations |
| FastAPI | latest | REST API serving |
| joblib | built-in | Model serialization |

---

## Quick Start

```bash
git clone https://github.com/AIML-Engineering-Lab/006_unsupervised_discovery.git
cd 006_unsupervised_discovery
pip install -r requirements.txt

# Train both models
python src/train.py

# Run inference
python src/predict.py

# Run tests
python tests/test_model.py

# Explore notebooks
jupyter notebook notebooks/
```
