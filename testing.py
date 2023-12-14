import torch
from sklearn.linear_model import LinearRegression,Ridge
import meshio
import numpy as np
from spae import SpectralAutoencoderFNN

points_coarse=meshio.read("data/Stanford_Bunny_red.stl").points
triangles_coarse=meshio.read("data/Stanford_Bunny_red.stl").cells_dict['triangle']
all_points_coarse=np.zeros((600,points_coarse.shape[0],points_coarse.shape[1]))
for i in range(600):
    all_points_coarse[i]=meshio.read("data/bunny_coarse_test_"+str(i)+".ply").points

points_dense=meshio.read("data/Stanford_Bunny.stl").points
triangles_dense=meshio.read("data/Stanford_Bunny.stl").cells_dict['triangle']
all_points_dense=np.zeros((600,points_dense.shape[0],points_dense.shape[1]))
for i in range(600):
    all_points_dense[i]=meshio.read("data/bunny_dense_test_"+str(i)+".ply").points

v_dense=np.load('v.npy')
v_coarse=np.load('v_red.npy')

lat_coarse=((all_points_coarse.transpose(0,2,1))@v_coarse).reshape(600,-1)
lat_dense=((all_points_dense.transpose(0,2,1))@v_dense).reshape(600,-1)


LinReg_dc=Ridge(alpha=0.01)
LinReg_dc.fit(lat_dense,lat_coarse)

LinReg_cd=Ridge(alpha=0.01)
LinReg_cd.fit(lat_coarse,lat_dense)

m_cd,b_cd=LinReg_cd.coef_,LinReg_cd.intercept_
m_dc,b_dc=LinReg_dc.coef_,LinReg_dc.intercept_
print(np.linalg.det(m_cd))
print(np.linalg.det(m_dc))
print(np.linalg.norm(lat_coarse-lat_dense@m_dc.T-b_dc))
print(np.linalg.norm(lat_dense-lat_coarse@m_cd.T-b_cd))

m_cd=torch.tensor(m_cd)
m_dc=torch.tensor(m_dc)
b_cd=torch.tensor(b_cd)
b_dc=torch.tensor(b_dc)

model=SpectralAutoencoderFNN(5)
model=torch.load("model.pt")
model.eval()
lat_dense=torch.tensor(lat_dense)
lat_coarse=torch.tensor(lat_coarse)

lat_dense_c=lat_dense@m_dc.T+b_dc
lat_dense_c_r=model(model.inverse(lat_dense_c))
print(torch.linalg.norm(lat_dense_c_r-lat_dense_c)/torch.linalg.norm(lat_dense_c))
lat_dense_r=lat_dense_c_r@m_cd.T+b_cd
print(torch.linalg.norm(lat_dense-lat_dense_r)/torch.linalg.norm(lat_dense))
lat_dense_r=lat_dense_r.detach().numpy()
lat_dense_r=lat_dense_r.reshape(600,3,-1)
dense_rec=lat_dense_r@v_dense.T
dense_rec=dense_rec.transpose(0,2,1)

print(np.linalg.norm(dense_rec-all_points_dense)/np.linalg.norm(all_points_dense))
for i in range(600):
    points=dense_rec[i]
    points=points-np.min(points)
    points=points/np.max(points)
    points=points-np.mean(points)
    meshio.write_points_cells("data/bunny_rec_"+str(i)+".ply",points,{"triangle":triangles_dense})
