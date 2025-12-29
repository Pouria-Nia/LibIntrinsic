import argparse
import json
import os
import sys
import numpy as np
import glob
import trimesh
import traceback

# Ensure we can import libintrinsic from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from libintrinsic.learning.clustering import auto_cluster_meshes
from libintrinsic.geometry.mesh import TriMesh
from libintrinsic.spectral.laplacian import compute_mass_matrix, compute_stiffness_matrix
from libintrinsic.numerics.solvers import compute_eigen_decomposition
from libintrinsic.spectral.descriptors import compute_shape_dna
from libintrinsic.functional.maps import shape_distance

# ------------------------------
# Helpers
# ------------------------------
def load_mesh_from_path(path):
    """Load a mesh robustly and convert to TriMesh."""
    try:
        # force='mesh' prevents loading generic scenes
        tm = trimesh.load(path, force='mesh') 
        
        if isinstance(tm, trimesh.Scene):
            if len(tm.geometry) > 0:
                tm = trimesh.util.concatenate(tuple(tm.geometry.values()))
            else:
                return None
        
        # --- SAFETY FIX ---
        # Ensures Rhino meshes are connected so heat can flow
        tm.merge_vertices()
        # ------------------
        
        try:
            components = tm.split(only_watertight=False)
            if len(components) > 1:
                components.sort(key=lambda m: len(m.vertices), reverse=True)
                tm = components[0]
        except Exception as e:
            # If split fails, just use what we have
            pass
        
        return TriMesh(tm.vertices, tm.faces)
    except Exception as e:
        print("ERROR (load):", e)
        traceback.print_exc()
        return None

def normalize_mesh(mesh: TriMesh):
    """Center and scale mesh to unit bounding box."""
    verts = np.array(mesh.vertices, dtype=float).copy()
    center = verts.mean(axis=0)
    verts -= center
    bbox_size = (verts.max(axis=0) - verts.min(axis=0)).max()
    if bbox_size > 0:
        verts /= bbox_size
    return TriMesh(verts, mesh.faces)

def get_dna(mesh, k=50):
    """Compute ShapeDNA of fixed length k."""
    try:
        mesh = normalize_mesh(mesh)
        M = compute_mass_matrix(mesh)
        L = compute_stiffness_matrix(mesh)
        total_area = M.sum()
        evals, _ = compute_eigen_decomposition(L, M, k=k)
        
        # Pad if too short
        if len(evals) < k:
            evals = np.pad(evals, (0, k - len(evals)), mode='edge') # 'edge' padding is safer than 0
            
        return compute_shape_dna(evals, area=total_area, normalize=True)
    except Exception as e:
        print("ERROR (DNA):", e)
        # traceback.print_exc() # Optional: hide trace to keep CLI clean
        return None

# ------------------------------
# Command: Search
# ------------------------------
def run_search(args):
    print(f"--- Spectral Search Engine ---")
    print(f"Query: {os.path.basename(args.query)}")
    print(f"Database: {args.database}")

    query_mesh = load_mesh_from_path(args.query)
    if not query_mesh:
        print("Error: Could not load query mesh.")
        return
        
    # Note: args.k here comes from the --k (eigenvalues) argument
    query_feat = get_dna(query_mesh, k=args.k)
    
    if query_feat is None:
        print("Error: Could not compute descriptor for query.")
        return

    # Load database
    if os.path.isdir(args.database):
        files = glob.glob(os.path.join(args.database, "*.off")) + \
                glob.glob(os.path.join(args.database, "*.obj"))
    else:
        files = [args.database]

    results = []
    print(f"Scanning {len(files)} candidates...")

    for fpath in files:
        if os.path.abspath(fpath) == os.path.abspath(args.query):
            continue

        target_mesh = load_mesh_from_path(fpath)
        if target_mesh:
            target_feat = get_dna(target_mesh, k=args.k)
            
            if target_feat is not None:
                if len(target_feat) != len(query_feat):
                    print(f"Skipping {fpath}: descriptor size mismatch")
                    continue
                
                dist = np.linalg.norm(query_feat - target_feat)
                results.append((dist, fpath))

    results.sort(key=lambda x: x[0])
    
    # args.top_k determines how many results to return
    top_k = results[:args.top_k]

    output_data = {
        "query": args.query,
        "results": [{"rank": i+1, "file": r[1], "distance": r[0]} for i, r in enumerate(top_k)]
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"✅ Success! Top {len(top_k)} matches saved to {args.output}")

# ------------------------------
# Command: Cluster
# ------------------------------
def run_clustering(args):
    print(f"--- Spectral Clustering CLI ---")
    if os.path.isdir(args.input):
        files = glob.glob(os.path.join(args.input, "*.off")) + \
                glob.glob(os.path.join(args.input, "*.obj"))
    else:
        files = [args.input]

    meshes = []
    valid_filenames = []
    for f in files:
        m = load_mesh_from_path(f)
        if m:
            meshes.append(m)
            valid_filenames.append(os.path.abspath(f))

    if len(meshes) < 2:
        print("Not enough meshes for clustering.")
        return

    labels, k_found, _ = auto_cluster_meshes(meshes, max_k=args.k)
    results = {"data": {}}
    for i, fpath in enumerate(valid_filenames):
        results["data"][fpath] = int(labels[i])

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"✅ Found {k_found} clusters.")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Search Command
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--query", "-q", required=True)
    search_parser.add_argument("--database", "-d", required=True)
    search_parser.add_argument("--output", "-o", required=True)
    search_parser.add_argument("--top_k", "-k", type=int, default=3, help="Number of matches to return")
    search_parser.add_argument("--k", type=int, default=50, help="Number of eigenvalues (ShapeDNA Size)")

    # Cluster Command
    cluster_parser = subparsers.add_parser("cluster")
    cluster_parser.add_argument("--input", "-i", required=True)
    cluster_parser.add_argument("--output", "-o", required=True)
    cluster_parser.add_argument("--k", type=int, default=3, help="Max clusters")

    args = parser.parse_args()

    if args.command == "search":
        run_search(args)
    elif args.command == "cluster":
        run_clustering(args)
    else:
        parser.print_help()