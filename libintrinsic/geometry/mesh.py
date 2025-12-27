import numpy as np
from numpy.typing import NDArray

class TriMesh:
    """
    A container for 3D triangular meshes.

    Attributes:
        vertices (NDArray[np.float64]): (N, 3) array of vertex coordinates.
        faces (NDArray[np.int32]): (M, 3) array of triangular face indices.
    """

    def __init__(self, vertices: NDArray[np.float64], faces: NDArray[np.int32]) -> None:
        """
        Initialize the mesh with vertices and faces.

        Args:
            Vertices: List or array of shape (N, 3).
            faces: List or array of shape (M, 3).

        Raises:
            ValueError: If input shapes are incorrect or faces are not triangles.
        """
        # Convert inputs to strictly typed numpy arrays
        self.vertices = np.asanyarray(vertices, dtype=np.float64)
        self.faces = np.asanyarray(faces, dtype=np.int32)

        self._validate()

    def _validate(self) -> None:
        """Checks the integrity of the mesh data."""
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"Vertices must be (N, 3). Got {self.vertices.shape}.")
            
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"Faces must be (M, 3). Only triangles are supported. Got {self.faces.shape}.")

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]
            
    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]
            
    def __repr__(self) -> str:
        return f"<IntrinsicMesh V={self.n_vertices} F={self.n_faces}>"