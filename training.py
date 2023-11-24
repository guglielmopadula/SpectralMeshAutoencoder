import torch
import numpy as np
import meshio
from sklearn.decomposition import PCA
import scipy.sparse
import scipy.io
from pytorch3d.ops import laplacian
from sklearn.manifold import spectral_embedding
from pytorch3d.structures import Meshes
points=meshio.read("data/Stanford_Bunny.stl").points

triangles=meshio.read("data/Stanford_Bunny.stl").cells_dict['triangle']
all_points=np.zeros((600,points.shape[0],points.shape[1]))
for i in range(600):
    all_points[i]=meshio.read("data/bunny_"+str(i)+".ply").points

all_points=all_points.reshape(all_points.shape[0],-1)




np.save('points.npy',all_points)


data=np.load('points.npy')

v=scipy.io.loadmat('v.mat')["v"]

data=data.reshape(600,-1,3)
data_t=data.transpose(0,2,1)
data_rec=data_t@v@v.T
data_rec=data_rec.transpose(0,2,1)
data_rec=data_rec.reshape(600,-1)
data=data.reshape(600,-1)
loss=np.linalg.norm(data-data_rec)/np.linalg.norm(data)
print("PCA loss is",loss)
print("data var is",np.var(data))
print("data_rec var is",np.var(data_rec))

data=data.reshape(600,-1,3)
data_t=data.transpose(0,2,1)
data_l=data_t@v

data_l=data_l.reshape(600,-1)
pca=PCA(n_components=5)
pca.fit(data_l)
print("PCA loss of spectral is",np.linalg.norm(pca.inverse_transform(pca.transform(data_l))-data_l)/np.linalg.norm(data_l))

data_l=torch.tensor(data_l,dtype=torch.float32)
data=torch.tensor(data,dtype=torch.float32)
data=data.reshape(600,-1)
triangles=torch.tensor(triangles)

v=torch.tensor(v,dtype=torch.float32)






from spae import SpectralAutoencoderFNN
model=SpectralAutoencoderFNN(5,v)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number of parameters",params)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs=500
for epoch in range(num_epochs):
    optimizer.zero_grad()
    recon=model(model.inverse(data_l)) 
    loss = torch.linalg.norm(recon-data_l)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        loss_tmp=torch.mean(torch.linalg.norm(model(model.inverse(data_l))-data_l,axis=1)/torch.linalg.norm(data_l,axis=1))
        print(loss_tmp)


model.eval()
with torch.no_grad():
    recon=model(model.inverse(data_l))
    print(torch.mean(torch.linalg.norm(recon-data_l,axis=1)/torch.linalg.norm(data_l,axis=1)))
    recon=recon.reshape(600,3,-1)
    recon=recon@v.t()
    recon=torch.transpose(recon,2,1)
    recon=recon.reshape(600,-1)
    loss_tmp=torch.mean(torch.linalg.norm(recon-data,axis=1)/torch.linalg.norm(data,axis=1))
    print("PCA loss is",loss_tmp)
    for i in range(600):
        meshio.write("data/bunny_rec_"+str(i)+".ply", meshio.Mesh(points=recon[i].detach().numpy().reshape(-1,3),cells={}))

    latent_data=model.inverse(data_l).detach().numpy()
    np.save('latent_data.npy',latent_data)
