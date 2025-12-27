import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def compute_eigen_decomposition(L, M, k: int = 10):
    """
    Solves the Generalized Eigenvalue Problem: L * phi = lambda * M * phi.
    
    It finds the 'k' smallest eigenvalues and corresponding eigenvectors.
    This corresponds to the low-frequency harmonics of the shape.
    
    Args:
        L (sp.csc_matrix): Stiffness matrix (N, N).
        M (sp.dia_matrix): Mass matrix (N, N).
        k (int): Number of eigenvalues to compute.
        
    Returns:
        evals (np.ndarray): (k,) array of eigenvalues.
        evecs (np.ndarray): (N, k) array of eigenvectors.
    """
    # Robustness: Ensure k is less than the size of the mesh
    n = L.shape[0]
    if k >= n:
        k = n - 1

    # We use 'sigma' (shift) to find eigenvalues near 0 efficiently.
    # L is positive semi-definite, so eigenvalues are >= 0.
    # A small negative sigma puts us "close" to 0 but avoids singularity if L has a null space.
    sigma = -1e-8
    
    try:
        # 'LM' = Largest Magnitude. 
        # Since we are in shift-invert mode (sigma is set), "Largest Magnitude" of the 
        # inverted operator corresponds to eigenvalues closest to sigma (closest to 0).
        evals, evecs = sla.eigsh(L, M=M, k=k, sigma=sigma, which='LM')
        
    except RuntimeError:
        # Fallback if shift-invert fails (rare, but happens on degenerate meshes)
        print("Warning: Shift-invert solver failed. Trying standard mode (slower).")
        evals, evecs = sla.eigsh(L, M=M, k=k, which='SM')

    # Sort them just in case (smallest to largest)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    # Sometimes very small negative values appear due to precision (e.g. -1e-12).
    # We clamp them to absolute 0.
    evals[evals < 1e-9] = 0.0
    
    return evals, evecs