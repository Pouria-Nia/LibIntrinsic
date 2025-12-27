import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
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
    

def compute_stiffness_matrix(mesh: TriMesh) -> sp.csc_matrix:
    """
    Computes the Stiffness Matrix (L) using Cotangent Weights.

    The matrix is constructed such that:
    L[i, j] = -0.5 * (cot(alpha) + cot(beta)) for edge (i, j)
    L[i, j] = -sum(L[i, j]) for all j != i.

    Args:
        mesh: The mesh container.

    Returns:
        L: The (N, N) sparse stiffness matrix.
    """
    v = mesh.vertices
    f = mesh.faces

    # 1. Get vertices for every face: shape (M, 3)
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]

    # 2. Compute vectors for the three edges of each triangle
    # e0: vector opposite to vertex 0 (v1 -> v2)
    # e1: vector opposite to vertex 1 (v2 -> v0)
    # e2: vector opposite to vertex 2 (v0 -> v1)
    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    # 3. Compute Cotangents of the angles opposite to these edges
    # Using formula: cot(theta) = (u . v) / |u x v|
    # Note: Double Area = |u x v|. We compute it once per face.

    # Double area of each triangle (norm of cross product)
    # Using e0 and e1 covers the whole triangle area
    n = np.cross(e0, e1)
    dbl_area = np.linalg.norm(n, axis=1)
    dbl_area = np.maximum(dbl_area, 1e-10)

    # Dot products of the vectors forming the angles
    # Angle at v0 is formed by vectors (v1-v0) and (v2-v0), i.e., e2 and -e1
    cot0 = np.sum(e2 * (-e1), axis=1) / dbl_area
    cot1 = np.sum(e0 * (-e2), axis=1) / dbl_area
    cot2 = np.sum(e1 * (-e0), axis=1) / dbl_area

    # 4. Construct the Sparse Matrix
    # We flatten the cotangents to map them to the correct edges
    term = 0.5 * np.concatenate([cot0, cot1, cot2])

    # Map to edge indices (i, j)
    # cot0 is opposite e0 (edge v1-v2)
    ii = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    jj = np.concatenate([f[:, 2], f[:, 0], f[:, 1]])

    # Create the matrix (automatically sums duplicate entries for shared edges)
    n_v = mesh.n_vertices
    W = coo_matrix((term, (ii, jj)), shape=(n_v, n_v))

    # MAke symmetric (since we only computed one way, e.g. v1-v2, but v2->v1 exists too)
    W = W + W.T

    # 5. Build L (Laplacian)
    # L = D - W, where D is diagonal sum of W
    diagonal = np.array(W.sum(axis=1)).flatten()
    L = sp.diags(diagonal, 0, shape=(n_v, n_v)) - W

    return L.tocsc()