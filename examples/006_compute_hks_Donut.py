import numpy as np
import trimesh

from libintrinsic.geometry.mesh import TriMesh  # Update to TriMesh if you renamed it
from libintrinsic.spectral.laplacian import compute_mass_matrix, compute_stiffness_matrix
from libintrinsic.numerics.solvers import compute_eigen_decomposition
from libintrinsic.spectral.descriptors import compute_hks


def scalar_to_color(values):
    """
    Simple helper to convert a 1D array of values into Red-Blue colors.
    Low values -> Blue
    High values -> Red
    """
    # Normalize 0 to 1
    v_min, v_max = values.min(), values.max()
    norm = (values - v_min) / (v_max - v_min + 1e-8)
    
    # Create RGBA array (N, 4)
    colors = np.zeros((len(values), 4), dtype=np.uint8)
    
    # Red channel (increases with value)
    colors[:, 0] = (255 * norm).astype(np.uint8)
    # Blue channel (decreases with value)
    colors[:, 2] = (255 * (1 - norm)).astype(np.uint8)
    # Alpha (Opacity)
    colors[:, 3] = 255
    
    return colors


def main():
    print("--- 1. Generating Torus (Donut) ---")
    # We use a Torus because it has interesting curvature!
    # A sphere is too boring (uniform color).
    original_mesh = trimesh.creation.torus(
        major_radius=1.0, 
        minor_radius=0.4, 
        major_sections=64, 
        minor_sections=64
    )
    
    # Wrap in our library's container
    my_mesh = TriMesh(original_mesh.vertices, original_mesh.faces)
    print(f"Mesh: {my_mesh.n_vertices} vertices")

    print("--- 2. Computing Spectral Fingerprint ---")
    M = compute_mass_matrix(my_mesh)
    L = compute_stiffness_matrix(my_mesh)
    evals, evecs = compute_eigen_decomposition(L, M, k=50)

    print("--- 3. Computing HKS (Heat Kernel Signature) ---")
    # We pick a time scale t=5.0 to see global structure
    t_scales = np.array([5.0]) 
    hks = compute_hks(evals, evecs, t_scales)
    
    # Extract the single column of data
    signal = hks[:, 0]

    print("--- 4. Visualizing ---")
    print("Opening 3D Window... (Close it to end script)")
    
    # Convert HKS values to Colors
    vertex_colors = scalar_to_color(signal)
    
    # Apply to the original trimesh object for viewing
    original_mesh.visual.vertex_colors = vertex_colors
    
    # Show it!
    original_mesh.show(smooth=False)


if __name__ == "__main__":
    main()