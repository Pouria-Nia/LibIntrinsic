import numpy as np

from libintrinsic.datasets.loaders import get_sphere
from libintrinsic.spectral.laplacian import compute_mass_matrix, compute_stiffness_matrix
from libintrinsic.numerics.solvers import compute_eigen_decomposition


def main():
    print("--- Spectral Analysis Verification ---")
    
    # 1. Load a high-res sphere (needs resolution to match theory)
    # Using subdivisions=4 gives enough vertices (~2500) for decent accuracy
    mesh = get_sphere(radius=1.0, subdivisions=4)
    print(f"Mesh: {mesh.n_vertices} vertices")
    
    # 2. Build Operators
    print("Building operators...")
    M = compute_mass_matrix(mesh)
    L = compute_stiffness_matrix(mesh)
    
    # 3. Solve Eigenproblem
    print("Solving for first 15 eigenvalues...")
    k = 15
    evals, evecs = compute_eigen_decomposition(L, M, k=k)
    
    # 4. Print Results
    print("\nCalculated Eigenvalues:")
    print(np.round(evals, 3))
    
    print("\nTheoretical Sphere Eigenvalues:")
    print("[0.    2.    2.    2.    6.    6.    6.    6.    6.    12...]")
    
    # Check the first non-zero group (should be approx 2.0)
    # Note: Discretization error is normal. You might get 2.05 or 2.1.
    first_mode = evals[1:4].mean()
    error = abs(first_mode - 2.0)
    
    print(f"\nMean of first non-zero mode (Target 2.0): {first_mode:.4f}")
    
    if error < 0.2:
        print("✅ SUCCESS: The spectral engine is accurate!")
    else:
        print("❌ WARNING: Results are drifting too far from theory.")


if __name__ == "__main__":
    main()