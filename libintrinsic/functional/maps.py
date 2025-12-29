import numpy as np
from scipy.spatial import cKDTree


def match_shapes_by_descriptors(desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
    """
    Finds the point-to-point correspondence between two shapes based on feature similarity.
    
    This uses a Nearest Neighbor search in the Descriptor Space (not 3D space).
    If vertex i on Shape 1 has a similar HKS to vertex j on Shape 2, they are matched.
    
    Args:
        desc1: (N1, k) Descriptors of Shape 1 (e.g., HKS).
        desc2: (N2, k) Descriptors of Shape 2.
        
    Returns:
        matches: (N1,) array of indices. matches[i] = j means vertex i on Shape 1 
                 maps to vertex j on Shape 2.
    """
    # 1. Build a fast lookup tree for Shape 2's descriptors
    # We are asking: "For every point in Shape 1, which point in Shape 2 is closest?"
    print(f"   -> Building KD-Tree for {desc2.shape[0]} target vertices...")
    tree = cKDTree(desc2)
    
    # 2. Query the tree
    # k=1 means "Find the single closest match"
    print(f"   -> Querying matches for {desc1.shape[0]} source vertices...")
    distances, matches = tree.query(desc1, k=1)
    
    return matches


def shape_distance(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """
    Calculates the geometric distance between two global signatures.
    Lower values = More similar shapes.
    
    Args:
        sig1, sig2: Global signatures (Global HKS or ShapeDNA).
        
    Returns:
        float: The Euclidean distance (L2 Norm).
    """
    return float(np.linalg.norm(sig1 - sig2))