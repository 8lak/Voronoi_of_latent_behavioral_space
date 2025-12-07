import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

# 1. Load Data
points_df = pd.read_csv("manifold_points.csv")
all_points = points_df[['x', 'y']].values
cluster_ids = points_df['cluster_id'].values

# 2. Setup Plot
fig, ax = plt.subplots(figsize=(10, 10))
EPS_RADIUS = 0.03  # The radius used in your analysis

# 3. Draw the EPS Circles (The "Density" visual)
# We draw these semi-transparently to show how they merge into groups
print("Drawing density circles...")
for i, point in enumerate(all_points):
    cluster = cluster_ids[i]
    
    # Color logic: Gray for noise, Spectral color for clusters
    if cluster == -1:
        color = 'gray'
        alpha = 0.1 # Faint for noise
    else:
        # Use the same colormap as your final plot
        unique_clusters = np.unique(cluster_ids)
        # Filter out -1 for color mapping
        valid_clusters = unique_clusters[unique_clusters != -1]
        color = plt.cm.Spectral(cluster / (len(valid_clusters) - 1))
        alpha = 0.2 # Semi-transparent for clusters to show overlap

    # Create the circle patch
    circle = Circle((point[0], point[1]), EPS_RADIUS, color=color, alpha=alpha, linewidth=0)
    ax.add_patch(circle)

# 4. Draw the actual points on top
unique_clusters = np.unique(cluster_ids)
for cluster_id in unique_clusters:
    mask = (cluster_ids == cluster_id)
    if cluster_id == -1:
        color = 'black'
        label = 'Noise'
        marker = 'x'
    else:
        valid_clusters = unique_clusters[unique_clusters != -1]
        color = plt.cm.Spectral(cluster_id / (len(valid_clusters)-1))
        label = f'Cluster {cluster_id}'
        marker = 'o'
    
    ax.scatter(all_points[mask, 0], all_points[mask, 1], c=[color], 
               s=50, edgecolors='k', zorder=10) # zorder ensures points are on top of circles

# 5. Framing and Aesthetics
ax.set_title(f"Density Clustering (eps={EPS_RADIUS})", fontsize=16)
ax.set_xlabel("Eigenvector 1")
ax.set_ylabel("Eigenvector 2")
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_aspect('equal')

# Zoom to fit data + radius
x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
margin = EPS_RADIUS * 1.5
ax.set_xlim(x_min - margin, x_max + margin)
ax.set_ylim(y_min - margin, y_max + margin)

plt.tight_layout()
plt.savefig("slide6_dbscan_radius.png", dpi=300)
plt.show()