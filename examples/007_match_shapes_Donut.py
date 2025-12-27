import numpy as np
import trimesh

from libintrinsic.geometry.mesh import TriMesh
from libintrinsic.spectral.laplacian import compute_mass_matrix, compute_stiffness_matrix
from libintrinsic.numerics.solvers import compute_eigen_decomposition
from libintrinsic.spectral.descriptors import compute_hks
from libintrinsic.functional.maps import match_shapes_by_descriptors


def process_shape(trimesh_obj):
    """Helper to run the full spectral pipeline on a mesh."""
    mesh = TriMesh(trimesh_obj.vertices, trimesh_obj.faces)
    M = compute_mass_matrix(mesh)
    L = compute_stiffness_matrix(mesh)
    evals, evecs = compute_eigen_decomposition(L, M, k=30)
    # Use multiple time scales for a robust fingerprint
    ts = np.array([0.1, 1.0, 5.0, 10.0])
    hks = compute_hks(evals, evecs, ts)
    return mesh, hks


def main():
    print("--- Shape Matching Experiment ---")
    
    # 1. Create Source Shape (A Donut)
    print("1. Generatng Shape A (Source)...")
    shape_a_viz = trimesh.creation.torus(major_radius=1.0, minor_radius=0.4, major_sections=32, minor_sections=32)
    mesh_a, hks_a = process_shape(shape_a_viz)
    
    # 2. Create Target Shape (The SAME Donut, but moved and rotated)
    # This proves we are matching geometry, not just XYZ position.
    print("2. Generating Shape B (Target - Rotated & Moved)...")
    shape_b_viz = shape_a_viz.copy()
    
    # Apply a random rotation and translation
    matrix = trimesh.transformations.random_rotation_matrix()
    matrix[0:3, 3] = [5.0, 5.0, 5.0] # Move it far away
    shape_b_viz.apply_transform(matrix)
    
    mesh_b, hks_b = process_shape(shape_b_viz)

    # 3. Match them!
    print("3. Matching shapes based on HKS...")
    # For every point on A, find the twin on B
    matches = match_shapes_by_descriptors(hks_a, hks_b)
    
    # 4. Verify Accuracy
    # Since we know it's a copy, vertex i on A should map to vertex i on B.
    # The 'matches' array should look like [0, 1, 2, 3, ...]
    
    correct_matches = np.arange(mesh_a.n_vertices)
    # Calculate how many we got right
    num_correct = np.sum(matches == correct_matches)
    accuracy = (num_correct / mesh_a.n_vertices) * 100
    
    print(f"\n--- Results ---")
    print(f"Total Vertices: {mesh_a.n_vertices}")
    print(f"Correct Matches: {num_correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    if accuracy > 90:
        print("✅ SUCCESS: The shapes were matched blindly using spectral fingerprints!")
    else:
        print("❌ FAILURE: The descriptors were not distinct enough.")

    # 5. Visual Proof (Optional)
    # We draw lines between a few random matching points
    print("\nVisualizing correspondence (showing 50 random links)...")
    
    # Pick 50 random indices
    indices = np.random.choice(mesh_a.n_vertices, 50, replace=False)
    
    # Get coordinates
    points_a = shape_a_viz.vertices[indices]
    points_b = shape_b_viz.vertices[matches[indices]]
    
    # Create lines connecting them
    # We use trimesh.load_path to draw segments
    lines = np.stack((points_a, points_b), axis=1)
    path = trimesh.load_path(lines)
    
    # Show both shapes and the connecting lines
    scene = trimesh.Scene([shape_a_viz, shape_b_viz, path])
    scene.show()


if __name__ == "__main__":
    main()