import torch
import numpy as np
import meshio
from sklearn.decomposition import PCA
import scipy.sparse
import scipy.io
from pytorch3d.structures import Meshes
from scipy.sparse.linalg import eigsh
from pytorch3d.ops import laplacian
import torch

##Modification of a code from pytorch3d
def laplacian(edges):
    """
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] = deg(i)       , if i == j
    L[i, j] = -1  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    V = torch.max(edges) + 1

    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=edges.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = -1 if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, -1.0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, -1.0, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = deg(i).
    idx = torch.arange(V, device=edges.device)
    idx = torch.stack([idx, idx], dim=0)
    L += torch.sparse.FloatTensor(idx, deg, (V, V))

    return L




'''
edges=scipy.io.loadmat('edges.mat')["edges"]
a=np.ones(edges.shape[1])
adjacency=scipy.sparse.coo_matrix((a, (edges[0,:], edges[1,:])),shape=(edges.max()+1, edges.max()+1))
laplacian,diag=scipy.sparse.csgraph.laplacian(adjacency, normed=False,return_diag=True)
w,v=eigsh(laplacian,k=512,which='LM',sigma=0,return_eigenvectors=True)
scipy.io.savemat('v.mat', {'v': v})
'''
edges=scipy.io.loadmat('edges.mat')["edges"]
points=torch.zeros(600,edges.max()+1,3)
edges=edges.T
edges=torch.tensor(edges)
L=laplacian(edges)
print(L)
print(L[0,0])
w,v=torch.lobpcg(L,k=512,largest=False)
scipy.io.savemat('v.mat', {'v': v})


