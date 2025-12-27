import numpy as np
import trimesh

from libintrinsic.datasets.loaders import get_sphere
from libintrinsic.spectral.laplacian import compute_mass_matrix, compute_stiffness_matrix
from libintrinsic.numerics.solvers import compute_eigen_decomposition
from libintrinsic.spectral.descriptors import compute_hks


def main():
    # 1. Load Data
    mesh_data = get_sphere(radius=1.0, subdivisions=3)
    print(f"Mesh: {mesh_data}")

    # 2. Compute Spectral Basis
    M = compute_mass_matrix(mesh_data)
    L = compute_stiffness_matrix(mesh_data)
    evals, evecs = compute_eigen_decomposition(L, M, k=50)

    # 3. Compute HKS
    # We look at heat after t=0.1 (short time) and t=10.0 (long time)
    ts = np.array([0.1, 1.0, 10.0])
    hks = compute_hks(evals, evecs, ts)
    
    print(f"HKS Shape: {hks.shape} (Vertices x TimeScales)")
    
    # 4. Analysis
    # On a perfect sphere, every point is geometrically identical.
    # Therefore, the HKS should be CONSTANT across all vertices for a fixed time.
    
    print("\n--- Checking Spherical Symmetry ---")
    for i, t in enumerate(ts):
        signature_at_t = hks[:, i]
        min_val = signature_at_t.min()
        max_val = signature_at_t.max()
        std_dev = signature_at_t.std()
        
        print(f"Time {t:4.1f}: Val ~ {min_val:.4f} (StdDev: {std_dev:.6f})")
        
        if std_dev < 0.01:
             print("   -> Symmetry holds (Heat spreads evenly)")
        else:
             print("   -> Symmetry broken (Mesh irregularity detected)")

    # 5. Optional: Visualize if you have a display
    # We assign the HKS values (at t=1.0) to the vertex colors
    try:
        # Normalize for visualization (0 to 255)
        # Note: We need to convert our 'TriMesh' back to a 'trimesh' object for viewing
        # or just use the raw trimesh if you want.
        
        # Simple text fallback if no GUI
        print("\nVisualizing logic: HKS allows us to color the mesh by curvature features.")
        print("Since this is a sphere, the color would be uniform.")
        
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()