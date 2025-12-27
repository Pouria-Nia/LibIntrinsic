import trimesh
from ..geometry.mesh import TriMesh

def get_sphere(radius: float = 1.0, subdivisions: int = 2) -> TriMesh:
    """
    Generates an icosphere mesh for testing spectral properties.

    An icosphere is preferred over a UV-sphere because its triangulation
    is more uniform, which is better for Laplacian calculations.

    Args:
        radius: The radius of the sphere.
        subdivisions: Level of detail (higher = more vertices)

    Returns:
        TriMesh: The generated mesh container. 
    """
    # Create the mesh using trimesh's built-in generator
    mesh_data = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)

    # Wrap it in our clean TriMesh class
    return TriMesh(vertices=mesh_data.vertices, faces=mesh_data.faces)