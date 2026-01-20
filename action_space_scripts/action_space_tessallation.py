import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# --- Helper function to handle infinite Voronoi regions ---
# This is a standard utility to make Voronoi regions plottable as finite polygons.
def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        # sort region vertices
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.asarray(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


# --- 1. Load Data ---
VORONOI_SITES_FILE = "data_in_csv/action_space/action_space_voronoi_sites.csv"
MANIFOLD_POINTS_FILE = "data_in_csv/action_space/action_manifold_points.csv"

print("--- Starting Phase V: Final Visualization & Reporting ---")
sites_df = pd.read_csv(VORONOI_SITES_FILE,index_col=0)
voronoi_sites = sites_df[['x', 'y']].values
points_df = pd.read_csv(MANIFOLD_POINTS_FILE)
all_points = points_df[['x', 'y']].values
print("Successfully loaded data.")

# --- 2. Re-compute Geometric Structures ---
vor = Voronoi(voronoi_sites)
hull = ConvexHull(all_points)

# --- 3. Generate the Final Plot ---
print("Generating the final, report-ready plot...")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# -- Layer 1: The Clipped, Shaded Voronoi Regions --
# Create the Convex Hull polygon which will act as our clipping mask
hull_poly = Polygon(all_points[hull.vertices], facecolor='none', edgecolor='none')
ax.add_patch(hull_poly)

# Get the finite Voronoi polygons
regions, vertices = voronoi_finite_polygons_2d(vor)
cluster_colors = [plt.cm.Spectral(i / (len(voronoi_sites)-1)) for i in range(len(voronoi_sites))]

# Plot each region as a colored, clipped polygon
for i, region_indices in enumerate(regions):
    # Find which original site this region belongs to
    
    polygon = Polygon(vertices[region_indices], facecolor=cluster_colors[i], alpha=0.6)
    # This is the key step: clip the polygon to the hull
    polygon.set_clip_path(hull_poly)
    ax.add_patch(polygon)

# -- Layer 2: The Cleaned-up Centroids --
# Render centroids as simple black points, as requested
ax.scatter(voronoi_sites[:, 0], voronoi_sites[:, 1], s=150, c='black', label='Behavioral Archetypes')

print("Adding labels to centroids...")
for cluster_id, row in sites_df.iterrows():
    ax.text(
        x=row['x'],
        y=row['y'] + 0.005, # Small vertical offset for clarity
        s=str(cluster_id),  # Ensure the ID is a string
        fontsize=12,
        fontweight='bold',
        ha='center',        # Center the text horizontally
        va='bottom'         # Vertically align to the bottom of the text
    )
# -- Layer 3: The Final Boundary --
# Plot the clean black outline of the Convex Hull on top of everything
ax.plot(all_points[hull.vertices, 0], all_points[hull.vertices, 1], 'k-', linewidth=3)
# Close the loop
ax.plot([all_points[hull.vertices[-1], 0], all_points[hull.vertices[0], 0]],
        [all_points[hull.vertices[-1], 1], all_points[hull.vertices[0], 1]], 'k-', linewidth=3)

# -- Final Plotting Touches --
ax.set_title('Map of the Latent Behavioral Space', fontsize=18)
ax.set_xlabel('Eigenvector 1', fontsize=12)
ax.set_ylabel('Eigenvector 2', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True)
ax.set_aspect('equal', adjustable='box')

# --- ADD THESE LINES TO FIX THE "BABY GRAPH" ---
# Calculate the min/max of your data's bounding box
x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

# Set the limits with a small aesthetic margin (e.g., 10% of the data's range)
x_margin = (x_max - x_min) * 0.1
y_margin = (y_max - y_min) * 0.1
ax.set_xlim(x_min - x_margin, x_max + x_margin)
ax.set_ylim(y_min - y_margin, y_max + y_margin)
# -----------------------------------------------

plt.tight_layout()
plt.savefig("images/latent_action_space_map_final.png", dpi=300)
plt.show() 