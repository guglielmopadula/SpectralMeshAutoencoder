import torch
import numpy as np
import meshio
from sklearn.decomposition import PCA
import scipy.sparse
import scipy.io
from sklearn.linear_model import LinearRegression
from tqdm import trange
torch.set_default_dtype(torch.float64)
points_red=meshio.read("data/Stanford_Bunny_red.stl").points
triangles_red=meshio.read("data/Stanford_Bunny_red.stl").cells_dict['triangle']
all_points_red=np.zeros((600,points_red.shape[0],points_red.shape[1]))
for i in range(600):
    all_points_red[i]=meshio.read("data/bunny_red_"+str(i)+".ply").points
all_points_red=all_points_red.reshape(all_points_red.shape[0],-1)
np.save('points_red.npy',all_points_red)
data_red=np.load('points_red.npy')
v_red=np.load('v_red.npy')
data_red=data_red.reshape(600,-1,3)
data_t_red=data_red.transpose(0,2,1)
data_rec_red=data_t_red@v_red@v_red.T
data_rec_red=data_rec_red.transpose(0,2,1)
data_rec_red=data_rec_red.reshape(600,-1)
data_red=data_red.reshape(600,-1)
loss_red=np.linalg.norm(data_red-data_rec_red)/np.linalg.norm(data_red)
print("PCA loss is",loss_red)
print("data var is",np.var(data_red))
print("data_rec var is",np.var(data_rec_red))
data_red=data_red.reshape(600,-1,3)
data_t_red=data_red.transpose(0,2,1)
data_l_red=data_t_red@v_red
data_l_red=data_l_red.reshape(600,-1)
data_l_red=torch.tensor(data_l_red)
data_red=torch.tensor(data_red)
data_red=data_red.reshape(600,-1)
triangles_red=torch.tensor(triangles_red)
v_red=torch.tensor(v_red)
from spae import SpectralAutoencoderFNN
model=SpectralAutoencoderFNN(5)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number of parameters",params)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs=500
for epoch in trange(num_epochs):
    optimizer.zero_grad()
    recon=model(model.inverse(data_l_red)) 
    loss = torch.linalg.norm(recon-data_l_red)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        loss_tmp=torch.mean(torch.linalg.norm(model(model.inverse(data_l_red))-data_l_red,axis=1)/torch.linalg.norm(data_l_red,axis=1))
        print(loss_tmp)

torch.save(model, "model.pt")


