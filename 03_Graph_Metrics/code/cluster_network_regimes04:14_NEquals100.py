import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --------------------------------------------------------
# LOAD PRECOMPUTED DATA (FAST)
# --------------------------------------------------------

clustering_vals = np.load("/Users/bass/LaurenOh/results/clustering_vals.npy")
path_lengths = np.load("/Users/bass/LaurenOh/results/path_lengths.npy")

# --------------------------------------------------------
# CLEAN
# --------------------------------------------------------

mask = np.isfinite(clustering_vals) & np.isfinite(path_lengths)

clustering_clean = clustering_vals[mask]
path_clean = path_lengths[mask]

print("Valid subjects:", len(clustering_clean))

# --------------------------------------------------------
# CLUSTER
# --------------------------------------------------------

X = np.column_stack([clustering_clean, path_clean])

kmeans = KMeans(n_clusters=2, random_state=0)
labels = kmeans.fit_predict(X)

print("\nCluster sizes:", np.bincount(labels))
print("Cluster centers:")
print(kmeans.cluster_centers_)

# --------------------------------------------------------
# PLOT
# --------------------------------------------------------

plt.figure()

for i in range(2):
    plt.scatter(
        clustering_clean[labels == i],
        path_clean[labels == i],
        label=f"Cluster {i}",
        alpha=0.7
    )

centers = kmeans.cluster_centers_

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    s=200,
    marker="X",
    label="Centers"
)

plt.xlabel("Clustering")
plt.ylabel("Path Length")
plt.title("Network Regimes")
plt.legend()
plt.show()