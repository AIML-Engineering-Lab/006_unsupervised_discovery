"""
Data Generator for Project 006: Unsupervised Discovery
Generates two synthetic datasets:
  A) Smartwatch Activity Clustering (general, intuitive)
  B) Wafer Defect Pattern Clustering (Posiva)
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def generate_smartwatch_data(n=4000, seed=42):
    """
    Smartwatch Activity Clustering
    No labels — the algorithm discovers the activities.
    True hidden clusters: Running, Walking, Cycling, Sleeping, Desk Work, Gym
    Features: steps_per_min, heart_rate_bpm, accelerometer_magnitude,
              skin_temperature_c, calories_per_min, gyroscope_magnitude
    """
    rng = np.random.default_rng(seed)

    activities = {
        'Running':    {'n': 600, 'steps': (160, 15), 'hr': (155, 15), 'accel': (3.8, 0.5),
                       'temp': (36.8, 0.3), 'cal': (12, 2), 'gyro': (2.5, 0.5)},
        'Walking':    {'n': 800, 'steps': (95, 12),  'hr': (95, 10),  'accel': (1.5, 0.3),
                       'temp': (36.4, 0.2), 'cal': (5, 1),  'gyro': (0.8, 0.2)},
        'Cycling':    {'n': 500, 'steps': (20, 8),   'hr': (130, 12), 'accel': (2.2, 0.4),
                       'temp': (36.6, 0.3), 'cal': (9, 1.5), 'gyro': (1.8, 0.4)},
        'Sleeping':   {'n': 700, 'steps': (0, 1),    'hr': (55, 6),   'accel': (0.1, 0.05),
                       'temp': (36.0, 0.2), 'cal': (1, 0.2), 'gyro': (0.05, 0.02)},
        'Desk Work':  {'n': 800, 'steps': (2, 2),    'hr': (72, 8),   'accel': (0.3, 0.1),
                       'temp': (36.2, 0.2), 'cal': (1.5, 0.3), 'gyro': (0.15, 0.05)},
        'Gym':        {'n': 600, 'steps': (80, 20),  'hr': (145, 18), 'accel': (4.5, 0.8),
                       'temp': (37.0, 0.4), 'cal': (14, 3), 'gyro': (3.2, 0.7)},
    }

    rows = []
    for activity, params in activities.items():
        n_act = params['n']
        rows.append(pd.DataFrame({
            'steps_per_min':          rng.normal(*params['steps'], n_act).clip(0, 220),
            'heart_rate_bpm':         rng.normal(*params['hr'], n_act).clip(40, 200),
            'accelerometer_magnitude': rng.normal(*params['accel'], n_act).clip(0, 8),
            'skin_temperature_c':     rng.normal(*params['temp'], n_act).clip(34, 39),
            'calories_per_min':       rng.normal(*params['cal'], n_act).clip(0.5, 25),
            'gyroscope_magnitude':    rng.normal(*params['gyro'], n_act).clip(0, 6),
            'true_activity': activity  # kept for evaluation only, not used in clustering
        }))

    df = pd.concat(rows, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.round(3)
    df.to_csv(DATA_DIR / "smartwatch_activity.csv", index=False)
    print(f"Smartwatch dataset: {len(df)} rows, {df['true_activity'].nunique()} hidden activities")
    return df


def generate_wafer_defect_data(n=8000, seed=99):
    """
    Wafer Defect Pattern Clustering (Post-Silicon Validation)
    No labels — the algorithm discovers defect patterns.
    True hidden clusters: Edge Ring, Center Spot, Scratch, Random, Donut, Local Cluster
    Features: x_coord (normalized), y_coord (normalized), defect_density,
              particle_size_um, electrical_fail_rate, distance_from_center
    """
    rng = np.random.default_rng(seed)

    patterns = {
        'Edge Ring':      {'n': 1200},
        'Center Spot':    {'n': 800},
        'Scratch':        {'n': 600},
        'Random':         {'n': 2500},
        'Donut':          {'n': 1000},
        'Local Cluster':  {'n': 1900},
    }

    rows = []

    # Edge Ring: high defects near the wafer edge (r > 0.85)
    n_er = patterns['Edge Ring']['n']
    r_er = rng.uniform(0.85, 1.0, n_er)
    theta_er = rng.uniform(0, 2*np.pi, n_er)
    rows.append(pd.DataFrame({
        'x_coord': (r_er * np.cos(theta_er)).round(3),
        'y_coord': (r_er * np.sin(theta_er)).round(3),
        'defect_density': rng.normal(0.75, 0.1, n_er).clip(0.4, 1.0),
        'particle_size_um': rng.normal(0.8, 0.2, n_er).clip(0.2, 2.0),
        'electrical_fail_rate': rng.normal(0.65, 0.1, n_er).clip(0.3, 1.0),
        'distance_from_center': r_er.round(3),
        'true_pattern': 'Edge Ring'
    }))

    # Center Spot: high defects near center (r < 0.25)
    n_cs = patterns['Center Spot']['n']
    r_cs = rng.uniform(0, 0.25, n_cs)
    theta_cs = rng.uniform(0, 2*np.pi, n_cs)
    rows.append(pd.DataFrame({
        'x_coord': (r_cs * np.cos(theta_cs)).round(3),
        'y_coord': (r_cs * np.sin(theta_cs)).round(3),
        'defect_density': rng.normal(0.80, 0.1, n_cs).clip(0.5, 1.0),
        'particle_size_um': rng.normal(1.5, 0.4, n_cs).clip(0.5, 3.0),
        'electrical_fail_rate': rng.normal(0.70, 0.1, n_cs).clip(0.4, 1.0),
        'distance_from_center': r_cs.round(3),
        'true_pattern': 'Center Spot'
    }))

    # Scratch: linear pattern across the wafer
    n_sc = patterns['Scratch']['n']
    x_sc = rng.uniform(-0.9, 0.9, n_sc)
    y_sc = 0.3 * x_sc + rng.normal(0, 0.05, n_sc)
    rows.append(pd.DataFrame({
        'x_coord': x_sc.round(3),
        'y_coord': y_sc.round(3),
        'defect_density': rng.normal(0.60, 0.1, n_sc).clip(0.3, 0.9),
        'particle_size_um': rng.normal(2.5, 0.5, n_sc).clip(1.0, 5.0),
        'electrical_fail_rate': rng.normal(0.55, 0.1, n_sc).clip(0.3, 0.9),
        'distance_from_center': np.sqrt(x_sc**2 + y_sc**2).round(3),
        'true_pattern': 'Scratch'
    }))

    # Random: uniformly distributed, low defect density
    n_rd = patterns['Random']['n']
    r_rd = rng.uniform(0, 1.0, n_rd)
    theta_rd = rng.uniform(0, 2*np.pi, n_rd)
    rows.append(pd.DataFrame({
        'x_coord': (r_rd * np.cos(theta_rd)).round(3),
        'y_coord': (r_rd * np.sin(theta_rd)).round(3),
        'defect_density': rng.exponential(0.08, n_rd).clip(0, 0.3),
        'particle_size_um': rng.normal(0.5, 0.15, n_rd).clip(0.1, 1.2),
        'electrical_fail_rate': rng.exponential(0.05, n_rd).clip(0, 0.2),
        'distance_from_center': r_rd.round(3),
        'true_pattern': 'Random'
    }))

    # Donut: ring between r=0.4 and r=0.7
    n_dn = patterns['Donut']['n']
    r_dn = rng.uniform(0.4, 0.7, n_dn)
    theta_dn = rng.uniform(0, 2*np.pi, n_dn)
    rows.append(pd.DataFrame({
        'x_coord': (r_dn * np.cos(theta_dn)).round(3),
        'y_coord': (r_dn * np.sin(theta_dn)).round(3),
        'defect_density': rng.normal(0.55, 0.1, n_dn).clip(0.3, 0.8),
        'particle_size_um': rng.normal(1.0, 0.3, n_dn).clip(0.3, 2.5),
        'electrical_fail_rate': rng.normal(0.45, 0.1, n_dn).clip(0.2, 0.8),
        'distance_from_center': r_dn.round(3),
        'true_pattern': 'Donut'
    }))

    # Local Cluster: concentrated in one quadrant
    n_lc = patterns['Local Cluster']['n']
    x_lc = rng.normal(0.55, 0.12, n_lc).clip(0.2, 0.9)
    y_lc = rng.normal(0.55, 0.12, n_lc).clip(0.2, 0.9)
    rows.append(pd.DataFrame({
        'x_coord': x_lc.round(3),
        'y_coord': y_lc.round(3),
        'defect_density': rng.normal(0.65, 0.12, n_lc).clip(0.3, 0.95),
        'particle_size_um': rng.normal(1.2, 0.3, n_lc).clip(0.4, 3.0),
        'electrical_fail_rate': rng.normal(0.58, 0.1, n_lc).clip(0.3, 0.9),
        'distance_from_center': np.sqrt(x_lc**2 + y_lc**2).round(3),
        'true_pattern': 'Local Cluster'
    }))

    df = pd.concat(rows, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.to_csv(DATA_DIR / "wafer_defect_patterns.csv", index=False)
    print(f"Wafer Defect dataset: {len(df)} rows, {df['true_pattern'].nunique()} hidden patterns")
    return df


if __name__ == "__main__":
    generate_smartwatch_data()
    generate_wafer_defect_data()
    print("Both datasets generated successfully.")
