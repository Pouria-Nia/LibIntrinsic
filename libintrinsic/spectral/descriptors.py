import numpy as np


def compute_hks(evals: np.ndarray, evecs: np.ndarray, time_scales: np.ndarray) -> np.ndarray:
    """
    Computes the Heat Kernel Signature (HKS) for a shape.
    
    HKS describes the local geometry of a vertex by measuring heat diffusion
    over different time scales.
    - Small t: Describes local curvature (bumps, noise).
    - Large t: Describes global structure (limbs, overall shape).
    
    Formula: HKS(x, t) = sum( exp(-lambda * t) * phi(x)^2 )
    
    Args:
        evals: (k,) Eigenvalues.
        evecs: (N, k) Eigenvectors.
        time_scales: (T,) array of time steps to evaluate.
        
    Returns:
        HKS: (N, T) array where row i is the signature of vertex i.
    """
    # 1. Square the eigenvectors (phi^2)
    # This makes the descriptor sign-invariant (fixes the flip issue)
    phi_squared = evecs ** 2
    
    # 2. Prepare storage (N vertices x T time scales)
    hks = np.zeros((evecs.shape[0], len(time_scales)))
    
    # 3. Compute Heat Kernel for each time scale
    # We broadcast the calculation: sum( weight * phi^2 )
    for i, t in enumerate(time_scales):
        # Calculate spectral weights: exp(-lambda * t)
        weights = np.exp(-evals * t)
        
        # Weighted sum across all k modes
        # dot product: (N, k) @ (k,) -> (N,)
        hks[:, i] = phi_squared.dot(weights)
        
    return hks


def compute_global_hks(hks: np.ndarray) -> np.ndarray:
    """
    Collapses local vertex signatures into a single global shape fingerprint.

    Args:
        hks: (N, T) array of local Heat Kernel Signatures.

    Returns:
        (T,) vector representing the average heat dissipation of the whole shape.
    """
    # Averaging across the vertex axis (axis 0)
    return np.mean(hks, axis=0)


def compute_shape_dna(evals: np.ndarray, area: float = 1.0, normalize: bool = True) -> np.ndarray:
    """
    Computes the ShapeDNA descriptor (the first k eigenvalues).
    
    Args:
        evals: (k,) array of eigenvalues.
        area:  The total surface area of the mesh (trace of Mass Matrix).
               Required for robust normalization via Weyl's Law.
        normalize: If True, multiplies by Area to make the signature scale-invariant.
                   
    Returns:
        (k,) vector of normalized eigenvalues.
    """
    dna = evals.copy()
    
    # Remove the first eigenvalue if it's 0 (it carries no shape info, just says 'I exist')
    if len(dna) > 0 and abs(dna[0]) < 1e-5:
        dna = dna[1:]
        
    if normalize:
        # Scale-Invariant Normalization (Weyl's Law)
        # We multiply by Area instead of dividing by eigenvalues.
        # This is robust to disconnected meshes where evals[1] might be 0.
        dna = dna * area
            
    return dna
