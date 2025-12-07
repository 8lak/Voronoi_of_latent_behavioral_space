import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull

# 1. Load Data
# We need the points to draw the hull, and sites for Voronoi
points_df = pd.read_csv("manifold_points.csv")
all_points = points_df[['x', 'y']].values

# Load sites (handling the cluster_id index correctly)
sites_df = pd.read_csv("voronoi_sites.csv", index_col=0)
voronoi_sites = sites_df[['x', 'y']].values

# 2. Compute Geometry
vor = Voronoi(voronoi_sites)
hull = ConvexHull(all_points)

# 3. Setup Plot
fig, ax = plt.subplots(figsize=(10, 10))

# -- Layer 1: Faint Data Points --
# We keep the original points very faint in the background to show context
ax.scatter(all_points[:, 0], all_points[:, 1], c='black', s=30, alpha=0.5, label='Original Data')

# -- Layer 2: The Voronoi Diagram (Construction Lines) --
# We use the built-in helper to draw the "infinite" lines, which looks cool for construction
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=2, line_alpha=0.6, point_size=0)

# -- Layer 3: The Convex Hull (The Boundary) --
for simplex in hull.simplices:
    ax.plot(all_points[simplex, 0], all_points[simplex, 1], 'k--', linewidth=2, label='Convex Hull' if simplex[0] == hull.simplices[0][0] else "")

# -- Layer 4: The Sites (Centroids) --
# We plot these as bold red X's to signify they are the "Mathematical Generators" of the regions
ax.scatter(voronoi_sites[:, 0], voronoi_sites[:, 1], s=200, c='red', marker='X', edgecolors='k', zorder=10, label='Centroids (Sites)')

# 4. Framing
ax.set_title("Geometric Construction", fontsize=16)
ax.set_xlabel("Eigenvector 1")
ax.set_ylabel("Eigenvector 2")
ax.grid(True)
ax.set_aspect('equal')

# Consistent Zoom (matches the previous slide)
x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
# Calculate margins based on data range
x_margin = (x_max - x_min) * 0.1
y_margin = (y_max - y_min) * 0.1
ax.set_xlim(x_min - x_margin, x_max + x_margin)
ax.set_ylim(y_min - y_margin, y_max + y_margin)

# Custom Legend to avoid duplicates
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

plt.tight_layout()
plt.savefig("slide7_geometry.png", dpi=300)
plt.show()