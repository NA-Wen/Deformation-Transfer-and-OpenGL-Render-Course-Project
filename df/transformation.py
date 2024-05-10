import numpy as np
import tqdm
from scipy import sparse
import meshlib
from scipy.sparse.linalg import lsqr
from scipy.linalg import lstsq

class LinearSolver:
    """A linear solver for minimum the L2 distance of (Ax-b)
    
    Keyword arguments:
    argument -- method name, A, b
    Return: x
    """
    def __init__(self, method: str, A, b) -> None:
        self.name = method
        self.A = A
        self.b = b
        if self.name == "lsqr":
            self.x = self.lsqr_solver()
        elif self.name == "lstsq":
            self.x = self.lstsq_solver()
        elif self.name == "qr":
            self.x = self.qr_solver()
        else:
            self.x = None
    
    def lsqr_solver(self):
        x_1, istop, itn, normr = lsqr(self.A, self.b[:,0])[:4]
        x_2, istop, itn, normr = lsqr(self.A, self.b[:,1])[:4]
        x_3, istop, itn, normr = lsqr(self.A, self.b[:,2])[:4]
        x = np.stack((x_1, x_2, x_3), axis=-1)
        return x
    
    def lstsq_solver(self):
        x = np.ones((3016,3))
        b_res = self.b - self.A.dot(x)
        x_res = lstsq(self.A, b_res)[0]
        return x + x_res
    
    def qr_solver(self):
        LU = (self.A.T @ self.A)
        x = np.linalg.solve(LU, self.A.T@self.b)
        return x



class TransformMatrix:
    @classmethod
    def expand(self, f: np.ndarray, inv: np.ndarray, size: int):
        i0, i1, i2, i3 = f
        col = np.array([i0, i0, i0, i1, i1, i1, i2, i2, i2, i3, i3, i3])
        data = np.concatenate([-inv.sum(axis=0), *inv])
        return sparse.coo_matrix((data, (np.array([0, 1, 2] * 4), col)), shape=(3, size), dtype=float)
    @classmethod
    def construct(self, faces: np.ndarray, invVs: np.ndarray, size:int):
        # f: 4*1, inv: 3*3
        assert len(faces) == len(invVs)
        assert faces.shape[1] == 4
        assert invVs.shape[1] == 3

        a =[(face,inv) for face, inv in zip(faces, invVs)]
        return sparse.vstack([
            self.expand(face, inv, size) for face, inv in zip(faces, invVs)
        ], dtype=float)

class Transformation:
    def __init__(
            self,
            source,
            target
    ):
        self.source = source.to_third_dimension(copy=False)
        self.target = target.to_third_dimension(copy=False)

        # 错切变换
        # shear_xy = 0.1
        # shear_matrix = np.array([[1, shear_xy, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 1]])
        # self.target.vertices = np.dot(self.target.vertices, shear_matrix.T)

        # rotation transformation of t_0
        theta = np.pi/4
        rotation_matrix = np.array([[np.cos(theta), np.sin(theta), 0],
                         [-np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])
        print(self.target.vertices[:1])
        self.target.vertices = target.vertices @ rotation_matrix.T
        
        # inverse rotation of t_0
        R = self.kabsch_algorithm(self.source.vertices, self.target.vertices)
        self.target.vertices = self.target.vertices @ np.linalg.inv(R).T
        
        self._Am = self._compute_mapping_matrix(self.target)

    def _compute_centroid(self, points):
        return np.mean(points, axis=0)
    
    def kabsch_algorithm(self, P, Q):
        """Calculating the optimal rotation matrix from P to Q"""
        assert P.shape == Q.shape
        
        centroid_P = self._compute_centroid(P)
        centroid_Q = self._compute_centroid(Q)
        
        P_prime = P - centroid_P
        Q_prime = Q - centroid_Q
        
        A = P_prime.T @ Q_prime
        U, _, Vt = np.linalg.svd(A)

        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        return R
    
    def _compute_mapping_matrix(cls, target):
        target = target.to_fourth_dimension(copy=False)
        inv_target_span = np.linalg.inv(target.span)
        Am = TransformMatrix.construct(target.faces, inv_target_span,len(target.vertices))
        return Am.tocsc()


    def __call__(self, pose):
        # rotation for pose (s_1) mesh
        # theta = np.pi/2
        # rotation_matrix = np.array([[np.cos(theta), np.sin(theta), 0],
        #                  [-np.sin(theta), np.cos(theta), 0],
        #                  [0, 0, 1]])
        # pose.vertices = pose.vertices @ rotation_matrix.T
        # with open(r"mesh1/s1_rotation.obj", "w") as f:
        #     for vertex in pose.vertices:
        #         f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        #     for face in pose.faces:
        #         f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        s = (pose.span @ np.linalg.inv(self.source.span)).transpose(0, 2, 1)
        Bm = np.concatenate(s)

        Astack = [self._Am]
        Bstack = [Bm]

        A: sparse.spmatrix = sparse.vstack(Astack, format="csc")
        A.eliminate_zeros()
        b = np.concatenate(Bstack)

        assert A.shape[0] == b.shape[0]
        assert b.shape[1] == 3

        A = np.array(A.toarray())
        b = np.array(b)
        
        # initial a linear solver 
        solver = LinearSolver("qr", A, b)
        x = solver.x

        vertices = x
        vertices[:,0] += self.target.vertices[0,0]-vertices[0,0]
        vertices[:,1] += self.target.vertices[0,1]-vertices[0,1]
        vertices[:,2] += self.target.vertices[0,2]-vertices[0,2]

        result = meshlib.Mesh(vertices=vertices[:len(self.target.vertices)], faces=self.target.faces)
        return result


if __name__ == "__main__":
    original_source = meshlib.Mesh.load(r"mesh/s0.obj")
    original_pose = meshlib.Mesh.load(r"mesh/s1.obj")
    original_target = meshlib.Mesh.load(r"mesh/t0.obj")
    print(original_target.vertices.shape)


    transf = Transformation(original_source, original_target)
    result = transf(original_pose)

    with open(r"result.obj", "w") as f:
        for vertex in result.vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in result.faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
