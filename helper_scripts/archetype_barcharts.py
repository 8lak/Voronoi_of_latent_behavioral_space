import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load Data using Pandas (Robust Method) ---
print("Loading data for charts...")
try:
    # Load Feature Names
    # header=None means the first row is data, not labels (unless you saved headers)
    # We flatten it to make sure it's a simple list of strings
    feature_names_df = pd.read_csv("data_in_csv/action_space/action_space_names.csv", header=None)
    feature_names = feature_names_df.values.flatten()

    # Load S_total (Feature Matrix)
    # We load it without a header, then assign the feature names manually
    df_features = pd.read_csv("data_in_csv/action_space/A_total_features.csv", header=None)
    # IMPORTANT: Ensure the number of columns matches the number of feature names
    if df_features.shape[1] == len(feature_names):
        df_features.columns = feature_names
    else:
        print(f"Warning: Mismatch between features ({df_features.shape[1]}) and names ({len(feature_names)})")

    # Load Cluster Labels
    df_clusters = pd.read_csv("data_in_csv/action_space/action_manifold_points.csv")

except FileNotFoundError:
    print("Error: Required CSV files not found. Please run previous phases first.")
    exit()

# --- 2. Organize Data ---
# Combine features and clusters into one DataFrame
df = df_features.copy()
df['cluster'] = df_clusters['cluster_id']

# Explicitly convert cluster column to integers to prevent the string error
# errors='coerce' turns non-numbers into NaN, fillna(-1) turns NaN to -1 (noise)
df['cluster'] = pd.to_numeric(df['cluster'], errors='coerce').fillna(-1).astype(int)

# Calculate centroids (average day per cluster)
# We exclude the noise points (-1)
centroids = df[df['cluster'] != -1].groupby('cluster').mean()

# --- 3. Setup Plotting ---
unique_clusters = centroids.index.sort_values().tolist()
n_clusters = len(unique_clusters)

# Create a grid (2 rows of 3 columns covers 6 clusters perfectly)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

# --- 4. Generate a Bar Chart for Each Cluster ---
for i, cluster_id in enumerate(unique_clusters):
    if i >= len(axes): break # Safety check in case you have > 6 clusters
    
    ax = axes[i]
    centroid = centroids.loc[cluster_id]
    
    # Get the Top 5 "Defining High" features
    top_5 = centroid.nlargest(5).sort_values(ascending=True)
    
    # Determine the color for this cluster (matching the main map)
    map_color = plt.cm.Spectral(cluster_id / (n_clusters - 1))
    
    # Plot Horizontal Bars
    bars = ax.barh(top_5.index, top_5.values, color=map_color, edgecolor='black', alpha=0.8)
    
    # Styling
    ax.set_title(f"Archetype {cluster_id}", fontsize=14, fontweight='bold', color='black')
    ax.set_xlim(0, 1.0) # Data is MinMax scaled (0-1)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Remove clutter
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Add value labels on the bars for clarity
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', va='center', fontsize=9, color='gray')

# --- 5. Final Layout ---
fig.suptitle("Behavioral Archetype Fingerprints (Top 5 Features)", fontsize=20, y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.90) # Make room for the super title
plt.savefig("images/action_space_fingerprints.png", dpi=300)
print("Chart generated: slide9_archetype_fingerprints.png")
plt.show()