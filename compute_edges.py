import torch
import meshio
import scipy.sparse
import scipy.io
from pytorch3d.structures import Meshes
from pygem import FFD
points=meshio.read("data/Stanford_Bunny.stl").points

triangles=meshio.read("data/Stanford_Bunny.stl").cells_dict['triangle']


mesh=Meshes(verts=[torch.tensor(points)],faces=[torch.tensor(triangles)])
edges=mesh.edges_packed()
e0, e1 = edges.unbind(1)
idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)
idx=idx.numpy()
scipy.io.savemat('edges.mat', {'edges': idx})
