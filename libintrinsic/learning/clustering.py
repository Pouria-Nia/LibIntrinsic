import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..geometry.mesh import TriMesh
from ..spectral.laplacian import compute_mass_matrix, compute_stiffness_matrix
from ..numerics.solvers import compute_eigen_decomposition
from ..spectral.descriptors import compute_shape_dna


def extract_features_batch(meshes, k_eigenvalues=20):
    """
    Converts a list of meshes into a (N_meshes, k_eigenvalues) feature matrix.
    """
    features = []
    valid_indices = []

    print(f"--- Extracting ShapeDNA (Top {k_eigenvalues} Eigenvalues) ---")
    total = len(meshes)
    
    for i, mesh in enumerate(meshes):
        try:
            # 1. Compute Spectral Basics
            M = compute_mass_matrix(mesh)
            L = compute_stiffness_matrix(mesh)

            # 2. Calculate Total Surface Area (Trace of Mass Matrix)
            # This is the stable number we need for normalization
            total_area = M.sum()

            # 3. Solve Eigenvalues
            # Request a few extra to ensure we have enough after filtering
            evals, _ = compute_eigen_decomposition(L, M, k=k_eigenvalues + 5)

            # 4. Compute Fingerprint (ShapeDNA)
            # PASS THE AREA HERE
            dna = compute_shape_dna(evals, area=total_area, normalize=True)

            # Keep only the requested number of features
            limit = min(len(dna), k_eigenvalues)
            if limit < k_eigenvalues:
                padded = np.zeros(k_eigenvalues)
                padded[:limit] = dna[:limit]
                features.append(padded)
            else:
                features.append(dna[:k_eigenvalues])
                
            valid_indices.append(i)

        except Exception as e:
            print(f"Skipping mesh {i} due to error: {e}")
            
    return np.array(features), valid_indices


def auto_cluster_meshes(meshes, max_k=10):
    """
    Automatically groupos meshes into clusters.
    It determines the optimal number of clusters (k) using the Silhoutte Score.

    Args:
        meshes: List of TriMesh objects.
        max_k: Maximum number of clusters to try.
    
    Returns:
        best_labels: Array of cluster IDs for each mesh.
        best_k: The optimal number of clusters found.
        valid_indices: Indices of the original list that were successfully proccessed.
    """
    # 1. Get the "Barcodes" for every mesh
    X, valid_indices = extract_features_batch(meshes)

    if len(X) < 2:
        print("Not enough meshes to cluster.")
        return np.zeros(len(X)), 1, valid_indices
    
    # 2. Find optimal 'k' (Elbow Method / Silhouette)
    best_score = -1
    best_k = 2
    best_model = None

    # We can't have more clusters than samples
    limit = min(max_k, len(X))

    print(f"--- Optimizing Cluster Count (Trying k=2 to {limit})---")

    for k in range(2, limit):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Silhouette score measures how distinct the clusters are (-1 to 1)
        score = silhouette_score(X, labels)
        print(f"   k={k} -> Silhouette Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_model = labels

        print(f"✅ Best Clustering found at k={best_k} (Score: {best_score:.4f})")

        return best_model, best_k, valid_indices
