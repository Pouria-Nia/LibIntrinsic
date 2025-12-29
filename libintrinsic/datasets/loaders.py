import os
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


def load_shrec(file_id, dataset_root=None):
    """
    Loads a SHREC mesh and KEEPS ONLY THE LARGEST CONNECTED COMPONENT.
    This fixes the 'floating leaves' problem that breaks spectral analysis.
    """
    if dataset_root is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        dataset_root = os.path.join(project_root, "data", "shrec2012")

    for ext in ['.off', '.obj']:
        path = os.path.join(dataset_root, f"{file_id}{ext}")
        if os.path.exists(path):
            try:
                tm = trimesh.load(path)
                
                # Handle Scenes
                if isinstance(tm, trimesh.Scene):
                    if len(tm.geometry) > 0:
                        tm = trimesh.util.concatenate(tuple(tm.geometry.values()))
                    else:
                        return None

                # --- THE FIX: Keep only the largest connected component ---
                # This requires 'networkx' installed
                components = tm.split(only_watertight=False)
                if len(components) > 1:
                    # Sort by vertex count and keep the big one
                    components.sort(key=lambda m: len(m.vertices), reverse=True)
                    tm = components[0]
                # ---------------------------------------------------------

                return TriMesh(tm.vertices, tm.faces)
            except Exception as e:
                print(f"Error loading {file_id}: {e}")
                return None
                
    return None