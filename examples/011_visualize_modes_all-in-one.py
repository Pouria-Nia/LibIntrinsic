import os
import sys
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import copy

# --- 1. SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from libintrinsic.geometry.mesh import TriMesh
from libintrinsic.spectral.laplacian import compute_mass_matrix, compute_stiffness_matrix
from libintrinsic.numerics.solvers import compute_eigen_decomposition

# --- CONFIGURATION ---
INPUT_FILE = os.path.join(parent_dir, "data", "vase-1.obj") 
# Try: 'vase-1.obj', 'db_5.obj', or your violin file

def robust_normalize_to_colors(vector, colormap_name='turbo'):
    """
    'turbo' or 'jet' are high-contrast colormaps that look 
    very 'engineering/scientific' for LinkedIn.
    """
    # Clip outliers (top/bottom 2%) for better contrast
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
    print(f"--- Creating Spectral Gallery for {os.path.basename(INPUT_FILE)} ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: File not found at: {INPUT_FILE}")
        return

    # 1. Load & Clean
    tm_orig = trimesh.load(INPUT_FILE)
    if isinstance(tm_orig, trimesh.Scene):
        tm_orig = trimesh.util.concatenate(tuple(tm_orig.geometry.values()))
    
    tm_orig.merge_vertices() # Essential for "Flow"
    
    # Calculate spacing (Width of object * 1.5)
    bbox = tm_orig.bounds
    width = bbox[1][0] - bbox[0][0]
    spacing = width * 1.5

    # 2. Compute Math
    print("Computing Eigenvalues (ShapeDNA)...")
    mesh = TriMesh(tm_orig.vertices, tm_orig.faces)
    M = compute_mass_matrix(mesh)
    L = compute_stiffness_matrix(mesh)
    
    # Get 10 modes just to be safe
    evals, evecs = compute_eigen_decomposition(L, M, k=10)

    # 3. Create the "Gallery" Scene
    scene = trimesh.Scene()
    
    # We will show Modes 1, 2, 3, 4, 5 side-by-side
    modes_to_show = [1, 2, 3, 4, 5]
    
    print("\nProcessing Visuals...")
    for i, mode_idx in enumerate(modes_to_show):
        if mode_idx >= evecs.shape[1]:
            continue

        print(f" - Coloring Mode {mode_idx}...")
        
        # Clone the mesh so we don't overwrite previous ones
        mesh_copy = tm_orig.copy()
        
        # Color it
        field = evecs[:, mode_idx]
        mesh_copy.visual.vertex_colors = robust_normalize_to_colors(field, 'turbo')
        
        # Move it to the right
        translation = [i * spacing, 0, 0]
        mesh_copy.apply_translation(translation)
        
        # Add to scene
        scene.add_geometry(mesh_copy)

    # 4. Show Everything
    print("\n✅ opening Window...")
    print(">> Tip: Press 'w' inside the window to toggle Wireframe (looks cool!)")
    print(">> Tip: Press 'a' to toggle Axes.")
    
    scene.show(
        caption="Spectral Geometry Gallery (Modes 1-5)", 
        smooth=False # Set False to see the sharp mesh, True for smooth look
    )

if __name__ == "__main__":
    main()