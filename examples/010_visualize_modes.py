import os
import sys
import numpy as np
import trimesh
import matplotlib.pyplot as plt

# --- 1. SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from libintrinsic.geometry.mesh import TriMesh
from libintrinsic.spectral.laplacian import compute_mass_matrix, compute_stiffness_matrix
from libintrinsic.numerics.solvers import compute_eigen_decomposition

# --- FILE CONFIGURATION ---
# We assume the file is in the 'data' folder. 
# If it is somewhere else, change this line!
INPUT_FILE = os.path.join(parent_dir, "data", "vase-1.obj")

def robust_normalize_to_colors(vector, colormap_name='coolwarm'):
    # Robust normalization to ignore outliers (top/bottom 2%)
    v_min = np.percentile(vector, 2)
    v_max = np.percentile(vector, 98)
    clipped = np.clip(vector, v_min, v_max)
    
    if v_max - v_min < 1e-9:
        normalized = np.zeros_like(clipped)
    else:
        normalized = (clipped - v_min) / (v_max - v_min)
    
    cmap = plt.get_cmap(colormap_name)
    return (cmap(normalized)[:, :3] * 255).astype(np.uint8)

def main():
    print(f"--- Visualizing Spectral Modes on {os.path.basename(INPUT_FILE)} ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ Error: File not found at: {INPUT_FILE}")
        print("Please check where your 'db_5.obj' is located and update line 18.")
        return

    # 1. Load Mesh
    tm_orig = trimesh.load(INPUT_FILE)
    if isinstance(tm_orig, trimesh.Scene):
        tm_orig = trimesh.util.concatenate(tuple(tm_orig.geometry.values()))
    
    print(f"Loaded Mesh: {len(tm_orig.vertices)} vertices")

    # 2. Fix Geometry (Standard Safety Procedure)
    # Even if it's an OBJ, we weld just to be safe.
    print("Ensuring mesh is watertight...")
    tm_orig.merge_vertices()
    
    # 3. Compute Spectral Geometry
    mesh = TriMesh(tm_orig.vertices, tm_orig.faces)
    
    print("Computing LBO (Heat Flow)...")
    M = compute_mass_matrix(mesh)
    L = compute_stiffness_matrix(mesh)
    
    # Request 6 modes so we can see the nice complex ones
    k_modes = 6
    print(f"Solving for {k_modes} vibration modes...")
    
    try:
        evals, evecs = compute_eigen_decomposition(L, M, k=k_modes)
        print("Solution found.")
    except Exception as e:
        print(f"Solver Error: {e}")
        return

    # 4. Visualize
    modes_to_show = [1, 2, 3, 4] # We skip 0 because it's just one color
    
    print("\n--- Starting Visualization ---")
    print("1. A window will open.")
    print("2. Rotate the object to a nice angle.")
    print("3. Take a SCREENSHOT for your LinkedIn post.")
    print("4. Close the window to see the next mode.")

    for mode_idx in modes_to_show:
        if mode_idx < evecs.shape[1]:
            print(f"\n>> Showing Mode {mode_idx} (Frequency: {evals[mode_idx]:.4f})")
            
            field = evecs[:, mode_idx]
            
            # Apply the "Red/Blue" thermal coloring
            tm_orig.visual.vertex_colors = robust_normalize_to_colors(field, 'coolwarm')
            
            scene = trimesh.Scene(tm_orig)
            scene.show(caption=f"Mode {mode_idx} - ShapeDNA")

if __name__ == "__main__":
    main()