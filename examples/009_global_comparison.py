import numpy as np
import trimesh

from libintrinsic.geometry.mesh import TriMesh
from libintrinsic.spectral.laplacian import compute_mass_matrix, compute_stiffness_matrix
from libintrinsic.numerics.solvers import compute_eigen_decomposition
from libintrinsic.spectral.descriptors import compute_hks, compute_global_hks, compute_shape_dna
from libintrinsic.functional.maps import shape_distance


def get_signature(mesh_obj):
    """Full pipeline to get a global signature."""
    mesh = TriMesh(mesh_obj.vertices, mesh_obj.faces)
    M = compute_mass_matrix(mesh)
    L = compute_stiffness_matrix(mesh)
    evals, evecs = compute_eigen_decomposition(L, M, k=50)
    
    # Get Global HKS
    ts = np.geomspace(0.1, 10.0, 10) # 10 time scales
    hks_local = compute_hks(evals, evecs, ts)
    hks_global = compute_global_hks(hks_local)
    
    # Get ShapeDNA
    dna = compute_shape_dna(evals)
    
    return hks_global, dna


def main():
    print("--- Global Shape Comparison (Shape Retrieval) ---")

    # 1. Create three shapes
    print("Generating Sphere A, Sphere B (Translated), and Torus C...")
    s_a = trimesh.creation.icosphere(subdivisions=3)
    
    s_b = s_a.copy()
    s_b.apply_translation([10, 0, 0]) # Move it away
    
    t_c = trimesh.creation.torus(major_radius=1.0, minor_radius=0.4, 
                                 major_sections=32, minor_sections=32)

    # 2. Extract Signatures
    print("Computing signatures...")
    hks_a, dna_a = get_signature(s_a)
    hks_b, dna_b = get_signature(s_b)
    hks_c, dna_c = get_signature(t_c)

    # 3. Compare Sphere A to Sphere B (Should be almost 0)
    dist_ab = shape_distance(dna_a, dna_b)
    
    # 4. Compare Sphere A to Torus C (Should be large)
    dist_ac = shape_distance(dna_a, dna_c)

    print("\n--- Results (ShapeDNA Distance) ---")
    print(f"Distance (Sphere A -> Sphere B): {dist_ab:.6f}")
    print(f"Distance (Sphere A -> Torus C) : {dist_ac:.6f}")

    if dist_ab < dist_ac:
        print("\n✅ SUCCESS: The computer recognized that two spheres are more similar")
        print("   than a sphere and a torus, even though they were in different places!")
    else:
        print("\n❌ FAILURE: The signatures are not distinguishing the shapes.")


if __name__ == "__main__":
    main()