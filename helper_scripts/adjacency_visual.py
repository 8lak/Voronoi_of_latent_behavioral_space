import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import kneighbors_graph

# 1. Load your data
S_total = np.loadtxt("s_total_features.csv", delimiter=",")

print(S_total.shape)
# 2. Compute the Graph (The Adjacency Matrix)
# This is exactly what SpectralEmbedding does internally
k = 10
A = kneighbors_graph(S_total, n_neighbors=k, mode='connectivity', include_self=False)

print(A)
# 3. Create a NetworkX Graph object from the matrix
G = nx.from_scipy_sparse_array(A)
print(G)

# 4. Visualization
plt.figure(figsize=(10, 8))

# Use a spring layout - this simulates the "springs" physics we talked about!
# It naturally pulls connected nodes together.
pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue', edgecolors='black')

# Draw edges (with some transparency to look nicer)
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')

# Draw labels (optional, can remove if too cluttered)
# nx.draw_networkx_labels(G, pos, font_size=8)

plt.title(f"The Nearest-Neighbor Graph (k={k})", fontsize=15)
plt.axis('off') # Turn off the x/y axis box
plt.tight_layout()
plt.savefig("slide4_network_graph.png", dpi=300)
plt.show()