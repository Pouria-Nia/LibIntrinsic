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