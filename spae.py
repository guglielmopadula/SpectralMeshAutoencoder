import torch
from torch import nn

    

class LearnablePooling(nn.Module):
    def __init__(self, input_size,output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.matrix=nn.Parameter(torch.randn(self.input_size,self.output_size))

    def forward(self, x):
        return x@self.matrix
    

class BasicLayer(nn.Module):
    def __init__(self, in_channels,out_channels,input_size,output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.conv = nn.Conv1d(self.in_channels,self.out_channels,padding=1,kernel_size=3)
        self.pool = LearnablePooling(self.input_size,self.output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        return x
    
class SimpleLayerEnc(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.batchnorm1=nn.BatchNorm1d(self.in_channels)
        self.batchnorm2=nn.BatchNorm1d(self.out_channels)
        self.conv1 = nn.Conv1d(self.in_channels,self.in_channels,padding=1,kernel_size=3)
        self.conv2 = nn.Conv1d(self.in_channels,self.out_channels,stride=2,kernel_size=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.batchnorm2(x)
        x=self.relu(x)
        return x

class LBR(nn.Module):
    def __init__(self, input_size,output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batchnorm1=nn.BatchNorm1d(self.output_size)
        self.lin = nn.Linear(self.input_size,self.output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x=self.lin(x)
        x=self.batchnorm1(x)
        x=self.relu(x)
        return x




class SimpleLayerDec(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.batchnorm1=nn.BatchNorm1d(self.out_channels)
        self.batchnorm2=nn.BatchNorm1d(self.out_channels)
        self.conv2 = nn.ConvTranspose1d(self.out_channels,self.out_channels,padding=1,kernel_size=3)
        self.conv1 = nn.ConvTranspose1d(self.in_channels,self.out_channels,stride=2,kernel_size=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.batchnorm2(x)
        x=self.relu(x)
        return x


'''
class SpectralAutoencoderCNN(nn.Module):
    def __init__(self,latent_size,v):
        super().__init__()
        self.latent_size = latent_size
        self.laplacian=LaplacianReduction(v)
        self.enc1=SimpleLayerEnc(3,100) 
        self.enc2=SimpleLayerEnc(100,100)
        self.enc3=SimpleLayerEnc(100,100)
        self.enc4=SimpleLayerEnc(100,100)   
        self.enc5=SimpleLayerEnc(100,100)
        self.enc6=SimpleLayerEnc(100,100)
        self.enc7=nn.Conv1d(100,1,stride=2,kernel_size=2)
        self.encflat=nn.Flatten()
        self.decflat=nn.Unflatten(1,(1,4))
        self.dec7=nn.ConvTranspose1d(1,100,stride=2,kernel_size=2)
        self.dec6=SimpleLayerDec(100,100)
        self.dec5=SimpleLayerDec(100,100)
        self.dec4=SimpleLayerDec(100,100)
        self.dec3=SimpleLayerDec(100,100)
        self.dec2=SimpleLayerDec(100,100)
        self.dec1=SimpleLayerDec(100,3)

    
    def inverse(self, x):
        x=x.reshape(x.shape[0],-1,3)
        x=torch.transpose(x,1,2)
        x = self.laplacian(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        x = self.enc7(x)
        x = self.encflat(x)
        return x

    def forward(self, x):
        x = self.decflat(x)
        x = self.dec7(x)
        x = self.dec6(x)
        x = self.dec5(x)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        x=self.laplacian.inverse(x)
        x=torch.transpose(x,1,2)
        x=x.reshape(x.shape[0],-1)  
        return x
'''    
class SpectralAutoencoderFNN(nn.Module):
    def __init__(self,latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.enc1=LBR(512*3,500) 
        self.enc2=LBR(500,500)
        self.enc3=LBR(500,500)
        self.enc4=nn.Linear(500,self.latent_size)
        self.dec4=LBR(self.latent_size,500)
        self.dec3=LBR(500,500)
        self.dec2=LBR(500,500)
        self.dec1=nn.Linear(500,3*512)

    
    def inverse(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        return x

    def forward(self, x):
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        return x

