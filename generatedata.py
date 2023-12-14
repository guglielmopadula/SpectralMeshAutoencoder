from pygem import FFD
import meshio
import numpy as np
from tqdm import trange
np.random.seed(1)
points=meshio.read("data/Stanford_Bunny.stl").points
points[:,2]=points[:,2]-np.min(points[:,2])+0.0000001
points[:,0]=points[:,0]-np.min(points[:,0])+0.2
points[:,1]=points[:,1]-np.min(points[:,1])+0.2
points=0.9*points/np.max(points)
triangles=meshio.read("data/Stanford_Bunny.stl").cells_dict['triangle']
for i in trange(600):
    ffd = FFD([3,3,3])
    ffd.array_mu_x=ffd.array_mu_x+0.2*np.random.rand(3,3,3)
    ffd.array_mu_y=ffd.array_mu_y+0.2*np.random.rand(3,3,3)
    tmp=0.2*np.random.rand(3,3,3)
    tmp[:,:,0]=0
    ffd.array_mu_z=ffd.array_mu_z+tmp

    def_points=ffd(points)
    meshio.write_points_cells("data/bunny_dense_test_"+str(i)+".ply",def_points,{"triangle":triangles})

np.random.seed(0)
points=meshio.read("data/Stanford_Bunny_red.stl").points
points[:,2]=points[:,2]-np.min(points[:,2])+0.0000001
points[:,0]=points[:,0]-np.min(points[:,0])+0.2
points[:,1]=points[:,1]-np.min(points[:,1])+0.2
points=0.9*points/np.max(points)
triangles=meshio.read("data/Stanford_Bunny_red.stl").cells_dict['triangle']
for i in trange(600):
    ffd = FFD([3,3,3])
    ffd.array_mu_x=ffd.array_mu_x+0.2*np.random.rand(3,3,3)
    ffd.array_mu_y=ffd.array_mu_y+0.2*np.random.rand(3,3,3)
    tmp=0.2*np.random.rand(3,3,3)
    tmp[:,:,0]=0
    ffd.array_mu_z=ffd.array_mu_z+tmp

    def_points=ffd(points)
    meshio.write_points_cells("data/bunny_red_"+str(i)+".ply",def_points,{"triangle":triangles})

np.random.seed(1)
points=meshio.read("data/Stanford_Bunny_red.stl").points
points[:,2]=points[:,2]-np.min(points[:,2])+0.0000001
points[:,0]=points[:,0]-np.min(points[:,0])+0.2
points[:,1]=points[:,1]-np.min(points[:,1])+0.2
points=0.9*points/np.max(points)
triangles=meshio.read("data/Stanford_Bunny_red.stl").cells_dict['triangle']
for i in trange(600):
    ffd = FFD([3,3,3])
    ffd.array_mu_x=ffd.array_mu_x+0.2*np.random.rand(3,3,3)
    ffd.array_mu_y=ffd.array_mu_y+0.2*np.random.rand(3,3,3)
    tmp=0.2*np.random.rand(3,3,3)
    tmp[:,:,0]=0
    ffd.array_mu_z=ffd.array_mu_z+tmp

    def_points=ffd(points)
    meshio.write_points_cells("data/bunny_coarse_test_"+str(i)+".ply",def_points,{"triangle":triangles})

