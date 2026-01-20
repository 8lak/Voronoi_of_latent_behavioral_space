import numpy as np
import pandas as pd
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# --- 1. Load Data & Perform Manifold Learning ---
S_TOTAL_FILE = "data_in_csv/s_total_features.csv"
FEATURE_NAMES_FILE = "data_in_csv/feature_names.csv" # You'll need this file from Phase 1

S_total = np.loadtxt(S_TOTAL_FILE, delimiter=",")
feature_names = np.loadtxt(FEATURE_NAMES_FILE, dtype=str, delimiter=',')

print("Running Spectral Embedding...")
se = SpectralEmbedding(n_components=2, n_neighbors=10, random_state=42)
manifold_2d = se.fit_transform(S_total)

# --- 2. Perform Clustering ---
print("Running DBSCAN to discover clusters...")
dbscan = DBSCAN(eps=0.02, min_samples=10)
clusters = dbscan.fit_predict(manifold_2d)

# --- 3. Consolidate All Data into a Master DataFrame ---
# This is the solution to your technical problem.
# We create a single, tidy DataFrame with one row per data point (day).
master_df = pd.DataFrame(manifold_2d, columns=['x', 'y'])
master_df['cluster_id'] = clusters

# For centroid interpretation, let's also bring in the high-dimensional data
df_s_total = pd.DataFrame(S_total, columns=feature_names)
master_df = pd.concat([master_df, df_s_total], axis=1)

print("\nData consolidation complete. Master DataFrame head:")
print(master_df.head())

# --- 4. Calculate Centroids (The "pandas-native" way) ---
# Now that we have the master_df, calculating centroids is a one-liner.
# We group by the cluster ID and calculate the mean for all other columns.
# We will drop the noise points (-1) before calculating.
all_centroids = master_df[master_df['cluster_id'] != -1].groupby('cluster_id').mean()

all_centroids.to_csv("data_in_csv/feature_means.csv",index=False)

# Separate the 2D centroids (for Voronoi sites) from the high-D ones (for interpretation)
voronoi_sites = all_centroids[['x', 'y']]
high_dim_centroids = all_centroids[feature_names]

print("\n--- 2D Centroids (Voronoi Sites) ---")
print(voronoi_sites)

# --- 5. Interpret the High-Dimensional Centroids ---
print("\n--- Behavioral Archetype Analysis ---")
for cluster_id, centroid_series in high_dim_centroids.iterrows():
    print(f"\n--- Archetype for Cluster {cluster_id} ---")
    print("Top 5 Defining HIGH Features:")
    print(centroid_series.nlargest(5))
    print("\nTop 5 Defining LOW Features:")
    print(centroid_series.nsmallest(5))

# --- 6. Visualize the Results with Centroids ---
unique_cluster_ids = sorted(master_df['cluster_id'].unique())
cluster_ids = [cid for cid in unique_cluster_ids if cid != -1]
color_denominator = max(1, len(cluster_ids) - 1)
cluster_colors = {
    cid: plt.cm.Spectral(i / color_denominator)
    for i, cid in enumerate(cluster_ids)
}

plt.figure(figsize=(10, 6))
# Plot the clustered points
for cluster_id, group in master_df.groupby('cluster_id'):
    color = 'black' if cluster_id == -1 else cluster_colors.get(cluster_id, plt.cm.Spectral(0.0))
    marker_size = 30 if cluster_id == -1 else 200
    plt.scatter(group['x'], group['y'], s=marker_size, c=[color], label=f'Cluster {cluster_id}', edgecolors='k')

# Plot the Voronoi sites (2D centroids)
plt.scatter(voronoi_sites['x'], voronoi_sites['y'], s=250, c='red', marker='X', edgecolors='k', label='Centroids')

plt.title(f'DBSCAN Clustering with Centroids (Estimated clusters: {len(voronoi_sites)})')
plt.xlabel("Eigenvector 1")
plt.ylabel("Eigenvector 2")
plt.grid(True)
plt.legend()
plt.show()

# --- 7. Save Final Outputs for Phase IV ---
VORONOI_SITES_FILE = "data_in_csv/voronoi_sites.csv"
MANIFOLD_POINTS_FILE = "data_in_csv/manifold_points.csv"

voronoi_sites.to_csv(VORONOI_SITES_FILE,index=True)
master_df[['x', 'y', 'cluster_id']].to_csv(MANIFOLD_POINTS_FILE)
print(f"\nOutputs for Phase IV saved to '{VORONOI_SITES_FILE}' and '{MANIFOLD_POINTS_FILE}'.")
