# Voronoi of Latent Behavioral Space

This pipeline processes behavioral data through a three-phase workflow to create a Voronoi diagram visualization of latent behavioral patterns. The pipeline uses spectral embedding for dimensionality reduction, DBSCAN clustering to identify behavioral archetypes, and Voronoi tessellation to map the behavioral space.

## Setup

1. **Install Dependencies**

   First, ensure you have Python 3.x installed. Then install all required packages:

   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - pandas
   - numpy
   - scikit-learn
   - scipy
   - matplotlib
   - plotly
   - networkx

2. **Prepare Your Data**

   Ensure you have a CSV file named `Contact Information (Responses) 8:9.csv` in the project root directory. This file should contain behavioral data with columns for sleep times, wake times, energy levels, and other behavioral features.

## Pipeline Execution

The pipeline must be run in sequential order. Each phase depends on outputs from the previous phase.

### Phase 1: Data Preparation and Feature Engineering

**Script:** `phase1.py`

**What it does:**
- Loads the behavioral data from CSV
- Performs advanced feature engineering (temporal features, rolling statistics, etc.)
- Normalizes all features using Min-Max scaling
- Saves the processed feature matrix and feature names for subsequent phases

**Run:**
```zsh
python3 phase1.py
```

**Outputs:**
- `s_total_features.csv` - Normalized feature matrix
- `feature_names.csv` - List of feature names

---

### Phase 2: Spectral Embedding and Clustering

**Script:** `phase2.py`

**What it does:**
- Loads the normalized features from Phase 1
- Applies Spectral Embedding to reduce dimensionality to 2D (creates the latent behavioral space)
- Performs DBSCAN clustering to identify behavioral archetypes
- Calculates cluster centroids (used as Voronoi sites)
- Visualizes the clustered data with centroids
- Saves outputs needed for Voronoi construction

**Run:**
```zsh
python3 phase2.py
```

**Outputs:**
- `voronoi_sites.csv` - Cluster centroids (x, y coordinates) for Voronoi diagram
- `manifold_points.csv` - All data points in 2D space with cluster assignments
- `cluster_labels.csv` - Cluster assignments for each data point
- Visualization plot showing DBSCAN clusters and centroids

**Note:** The Spectral Embedding method uses under-the-hood computations that can be visualized using helper scripts:
- `adjacency_visual.py` - Visualizes the k-nearest neighbor graph that Spectral Embedding uses internally
- `points_with_radius.py` - Visualizes the DBSCAN density circles (eps radius) used for clustering

---

### Phase 3: Voronoi Diagram Visualization

**Script:** `phase3.py`

**What it does:**
- Loads Voronoi sites and manifold points from Phase 2
- Constructs the Voronoi diagram from cluster centroids
- Clips Voronoi regions to the convex hull of all data points
- Creates a color-coded visualization of the behavioral space map
- Labels each behavioral archetype

**Run:**
```zsh
python3 phase3.py
```

**Outputs:**
- `latent_space_map_final.png` - Final Voronoi diagram visualization of the latent behavioral space

---

## Helper Scripts for Visualization

These scripts are not part of the main pipeline sequence but provide additional visualizations to understand the methods:

### `adjacency_visual.py`

**Purpose:** Visualizes the k-nearest neighbor graph used internally by Spectral Embedding in Phase 2.

This script shows the adjacency matrix as a network graph, demonstrating how Spectral Embedding connects similar data points before computing the low-dimensional embedding.

**Run:**
```zsh
python3 adjacency_visual.py
```

**Requirements:** Must have `s_total_features.csv` from Phase 1.

**Output:** `slide4_network_graph.png`

---

### `points_with_radius.py`

**Purpose:** Visualizes the DBSCAN density clustering method used in Phase 2.

This script draws circles around each point with radius equal to the DBSCAN `eps` parameter, showing how density-based clustering groups nearby points together.

**Run:**
```zsh
python3 points_with_radius.py
```

**Requirements:** Must have `manifold_points.csv` from Phase 2.

**Output:** `slide6_dbscan_radius.png`

---

### `archetype_barcharts.py`

**Purpose:** Creates bar charts showing the defining features of each behavioral archetype (cluster).

This visualization helps interpret what makes each cluster unique by displaying the top 5 highest-valued features for each archetype.

**Run:**
```zsh
python3 archetype_barcharts.py
```

**Requirements:** Must have `s_total_features.csv`, `feature_names.csv`, and `cluster_labels.csv` from Phases 1 and 2.

**Output:** `slide9_archetype_fingerprints.png`

---

### `region_construction.py`

**Purpose:** Visualizes the geometric construction of the Voronoi diagram before final color-coding.

This script shows the Voronoi diagram construction lines, convex hull boundary, and sites (centroids) to illustrate how the final map is built geometrically.

**Run:**
```zsh
python3 region_construction.py
```

**Requirements:** Must have `voronoi_sites.csv` and `manifold_points.csv` from Phase 2.

**Output:** `slide7_geometry.png`

---

## Complete Workflow Summary

```zsh
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the main pipeline
python3 phase1.py
python3 phase2.py
python3 phase3.py

# 3. (Optional) Generate additional visualizations
python3 adjacency_visual.py
python3 points_with_radius.py
python3 archetype_barcharts.py
python3 region_construction.py
```

## Output Files

### Main Pipeline Outputs:
- `s_total_features.csv` - Normalized feature matrix
- `feature_names.csv` - Feature names
- `voronoi_sites.csv` - Cluster centroids for Voronoi diagram
- `manifold_points.csv` - 2D embedded points with cluster labels
- `cluster_labels.csv` - Cluster assignments
- `latent_space_map_final.png` - Final Voronoi diagram

### Visualization Outputs:
- `slide4_network_graph.png` - K-nearest neighbor graph
- `slide6_dbscan_radius.png` - DBSCAN density visualization
- `slide7_geometry.png` - Voronoi construction visualization
- `slide9_archetype_fingerprints.png` - Behavioral archetype bar charts

## Troubleshooting

- **FileNotFoundError**: Make sure you've run previous phases in order. Each phase requires outputs from earlier phases.
- **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **Empty clusters**: If DBSCAN produces too few or too many clusters, you may need to adjust the `eps` and `min_samples` parameters in `phase2.py`

## Notes

- The pipeline is designed to work with behavioral data that includes temporal patterns (sleep, wake times, daily activities)
- Parameters in Phase 2 (Spectral Embedding and DBSCAN) can be tuned based on your specific dataset characteristics
- The final Voronoi diagram represents behavioral archetypes as regions in the latent space, where each region contains points most similar to its centroid archetype
