import torch
import numpy as np
import meshio
from sklearn.decomposition import PCA
import scipy.sparse
import scipy.io
from pytorch3d.structures import Meshes
from scipy.sparse.linalg import eigsh
edges=scipy.io.loadmat('edges.mat')["edges"]
a=np.ones(edges.shape[1])
adjacency=scipy.sparse.coo_matrix((a, (edges[0,:], edges[1,:])),shape=(edges.max()+1, edges.max()+1))
laplacian,diag=scipy.sparse.csgraph.laplacian(adjacency, normed=False,return_diag=True)
w,v=eigsh(laplacian,k=512,which='SM',sigma=0,return_eigenvectors=True)


