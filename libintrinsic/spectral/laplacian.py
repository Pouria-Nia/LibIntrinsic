import numpy as np
import scipy.sparse as sp
from ..geometry.mesh import TriMesh

def compute_mass_matrix(mesh: TriMesh) -> sp.dia_matrix:
    """
    Computes the Lumped Mass Matrix (Diagonal) for a triangle mesh.

    The mass matrix represents the local area associated with each vertex.
    For the lumped mass matrix, 1/3 of the area of each triangle is
    assigned to each of its three vertices.

    Args:
        mesh: The TriMesh object containing V and F.

    Returns:
        M: A (N, N) diagonal sparse matrix where M[i, i]
           is the area associated with vertex i.
    """
    vertices = mesh.vertices
    faces = mesh.faces

    # --- 1. Vectorized Edge Calculation ---
    # Get the 3 vertices for every face: shape (M, 3, 3)
    # v0, v1, v2 are (M, 3) arrays of coordinates
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Calculate two edge vectors per face
    e1 = v1 - v0
    e2 = v2 - v0

    # --- 2. Face Area Calculation ---
    # The area of a triangle is 0.5 * length of the cross product of two edges
    # np.cross(e1, e2) gives the normal vector weighted by 2*Area
    cross_prod = np.cross(e1, e2)

    # Calculate magnitude (norm) of the cross products
    # axis=1 computes norm across the (x,y,z) dimension
    norms = np.linalg.norm(cross_prod, axis=1)

    faces_areas = 0.5 * norms  # shape (M,)

    # --- 3. Distribute Area to Vertices ---
    # Each vertex gets 1/3 of the face's area.
    # We need to accumulate these values into the correct vertex indices.

    # We want a vector 'mass' of shape (N,)
    n_vertices = mesh.n_vertices
    mass = np.zeros(n_vertices, dtype=np.float64)

    # We use np.add.at for unbuffered summation (like a fast 'scatter_add')
    # For every face, add area/3 to its three vertices
    area_per_vertex = faces_areas / 3.0

    # Add to first column of indices (v0)
    np.add.at(mass, faces[:, 0], area_per_vertex)
    # Add to first column of indices (v1)
    np.add.at(mass, faces[:, 1], area_per_vertex)
    # Add to first column of indices (v2)
    np.add.at(mass, faces[:, 2], area_per_vertex)

    # --- 4. Construct Sparse Matrix ---
    # Create a diagonal matrix
    M = sp.diags(mass, offsets=0, shape=(n_vertices, n_vertices), format='dia')

    return M
    
