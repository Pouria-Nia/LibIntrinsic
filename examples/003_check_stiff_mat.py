import numpy as np

from libintrinsic.datasets.loaders import get_sphere
from libintrinsic.spectral.laplacian import compute_mass_matrix, compute_stiffness_matrix


def main():
    mesh = get_sphere(radius=1.0, subdivisions=2)
    
    # 1. Compute Matrices
    M = compute_mass_matrix(mesh)
    L = compute_stiffness_matrix(mesh)
    
    print(f"Stiffness Matrix (L) shape: {L.shape}")
    print(f"Non-zeros in L: {L.nnz}")
    
    # 2. Verify: Row Sums should be zero (L * 1 = 0)
    # Create a vector of ones
    ones = np.ones(mesh.n_vertices)
    
    # Multiply L by ones
    result = L @ ones
    
    # Check if result is close to zero
    error = np.linalg.norm(result)
    print(f"L * 1 Error (Should be ~0): {error:.10f}")
    
    if error < 1e-8:
        print("✅ Laplacian Check Passed: Constant functions have zero energy.")
    else:
        print("❌ Laplacian Check Failed.")


if __name__ == "__main__":
    main()