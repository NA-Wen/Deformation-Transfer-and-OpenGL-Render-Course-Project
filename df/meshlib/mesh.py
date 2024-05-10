"""
Classes to load and handle mesh data.
"""

import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pywavefront



@dataclass
class Mesh:
    """
    First simple data structure holding only the vertices and faces in a numpy array

    @param vertices th positions of triangle corners (x,y,z)
    @param faces the triangles (Triple of vertices indices)
    """
    vertices: np.ndarray
    faces: np.ndarray

    @classmethod
    def from_pywavefront(cls, obj: pywavefront.Wavefront) -> "Mesh":
        """
        Load a mesh from a pywavefront object
        :param obj:
        :return:
        """
        assert obj.mesh_list
        return cls(
            vertices=np.array(obj.vertices),
            faces=np.array(obj.mesh_list[0].faces)
        )

    @classmethod
    def load_obj(cls, file: str, **kwargs) -> "Mesh":
        """
        Load a mesh from a .obj file
        :param file:
        :param kwargs:
        :return:
        """
        assert os.path.isfile(file), f"Mesh file is missing: {file}"
        kwargs.setdefault("encoding", "UTF-8")
        return cls.from_pywavefront(pywavefront.Wavefront(file, collect_faces=True, **kwargs))

    @classmethod
    def load_npz(cls, file: str, **kwargs) -> "Mesh":
        """
        Load a mesh from a numpy file .npz
        :param file:
        :param kwargs:
        :return:
        """
        assert os.path.isfile(file), f"Mesh file is missing: {file}"
        data = np.load(file)
        return cls(data["vertices"], data["faces"])

    @classmethod
    def load(cls, file: str, **kwargs) -> "Mesh":
        if file.endswith(".obj") or file.endswith(".pose"):
            return cls.load_obj(file, **kwargs)
        elif file.endswith(".npz"):
            return cls.load_npz(file, **kwargs)
        raise ValueError("Invalid file format")

    def get_centroids(self) -> np.ndarray:
        return self.vertices[self.faces[:, :3]].mean(axis=1)

    def scale(self, factor: float):
        """
        Scale the mesh
        :param factor:
        :return:
        """
        self.vertices *= factor
        return self

    

    def span_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the triangle span components of each surface with the offset v1
        :return:
            Tuple of the three triangle spans
        """
        v1, v2, v3 = self.vertices[self.faces.T][:3]
        # print(v1.shape,v2.shape,v3.shape)
        a = v2 - v1
        b = v3 - v1
        tmp = np.cross(a, b)
        c = (tmp.T / np.sqrt(np.linalg.norm(tmp, axis=1))).T
        return a, b, c

    @property
    def span(self) -> np.ndarray:
        """
        Calculates the triangle spans of each surface with the offset v1.
        V actually 
        The span components are ordered in columns.
        :return:
            triangles Nx3x3
        """
        a, b, c = self.span_components()
        return np.transpose((a, b, c), (1, 2, 0))

    @property
    def v1(self):
        return self.vertices[self.faces[:, 0]]

    def get_dimension(self) -> int:
        return self.faces.shape[1]

    def is_fourth_dimension(self) -> bool:
        return self.get_dimension() == 4

    def to_fourth_dimension(self, copy=True) -> "Mesh":
        if self.is_fourth_dimension():
            if copy:
                return Mesh(np.copy(self.vertices), np.copy(self.faces))
            else:
                return self

        assert self.vertices.shape[1] == 3, f"Some strange error occurred! vertices.shape = {self.vertices.shape}"
        a, b, c = self.span_components()
        v4 = self.v1 + c
        new_vertices = np.concatenate((self.vertices, v4), axis=0)
        v4_indices = np.arange(len(self.vertices), len(self.vertices) + len(c))
        new_faces = np.concatenate((self.faces, v4_indices.reshape((-1, 1))), axis=1)
        return Mesh(new_vertices, new_faces)

    def is_third_dimension(self) -> bool:
        return self.faces.shape[1] == 3

    def to_third_dimension(self, copy=True) -> "Mesh":
        if self.is_third_dimension():
            if copy:
                return Mesh(np.copy(self.vertices), np.copy(self.faces))
            else:
                return self

        assert self.vertices.shape[1] == 3, f"Some strange error occurred! vertices.shape = {self.vertices.shape}"
        new_faces = self.faces[:, :3]
        new_vertices = self.vertices[:np.max(new_faces) + 1]
        return Mesh(new_vertices, new_faces)

    def transpose(self, shape=(0, 1, 2)):
        shape = np.asarray(shape)
        assert shape.shape == (3,)
        return Mesh(
            vertices=self.vertices[:, shape],
            faces=self.faces
        )

    def normals(self) -> np.ndarray:
        v1, v2, v3 = self.vertices[self.faces.T][:3]
        vns = np.cross(v2 - v1, v3 - v1)
        return (vns.T / np.linalg.norm(vns, axis=1)).T


