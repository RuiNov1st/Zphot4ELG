import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time

class catalog_MLP(nn.Module):
    def __init__(self,features_num):
        super(catalog_MLP,self).__init__()
        self.layers = nn.ModuleList()
        self.activ = getattr(nn,'GELU')()
        
        self.layers.append(nn.Linear(features_num,512))
        for _ in range(2-1):
            self.layers.append(nn.Linear(512,512))
        
        self.dropout = nn.Dropout(0.1)
        self.num_layers = 2
        
    def forward(self,X):
        for i in range(self.num_layers):
            X = self.layers[i](X)
            X = self.activ(X)
            X = self.dropout(X)
        
        return X

class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4,with_kernel_5=True):
        super(Inception,self).__init__()
        # path1: 1*1 conv
        self.p1_1 = nn.Conv2d(in_channels,c1,kernel_size=1)
        # path2: 1*1 -> 3*3
        self.p2_1 = nn.Conv2d(in_channels,c2[0],kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)
        # path3: 1*1 -> 5*5
        # adding path3 or not:
        self.with_kernel_5 = with_kernel_5
        if self.with_kernel_5:
            self.p3_1 = nn.Conv2d(in_channels,c3[0],kernel_size=1)
            self.p3_2 = nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)
        # path4: 1*1 -> pooling:
        self.p4_1 = nn.Conv2d(in_channels,c4,kernel_size=1)
        self.pooling = nn.AvgPool2d(kernel_size=2,stride=1)
        
    def forward(self,X):
        p1 = F.relu(self.p1_1(X))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(X))))
        p4 = F.pad(self.pooling(F.relu(self.p4_1(X))),(1,0,1,0))
        if self.with_kernel_5:
            p3 = F.relu(self.p3_2(F.relu(self.p3_1(X))))
            output = torch.cat([p1,p2,p3,p4],dim=1)
        else:
            output = torch.cat([p1,p2,p4],dim=1)

        return output


class image_CNN(nn.Module):
    def __init__(self,channels1,channels2):
        super(image_CNN,self).__init__()
        self.channels1 = channels1
        self.channels2 = channels2
    
        self.conv01 = nn.Conv2d(len(channels1),64,kernel_size=5,padding=2)
        self.conv0p1 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.i01 = Inception(64,64,(48,64),(48,64),64,True)
        self.i11 = Inception(256,92,(64,92),(64,92),64,True)
        self.i1p1 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.i21 = Inception(340,128,(92,128),(92,128),92,True)
        self.i31 = Inception(476,128,(92,128),(92,128),92,True)
        self.i3p1 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.i41 = Inception(476,128,(92,128),(None,None),92,False)
        
    
        self.convseq1 = nn.Sequential(
            nn.Conv2d(348,96,kernel_size=3,stride=1,padding=0), # 6*6
            nn.Conv2d(96,96,kernel_size=3,stride=1,padding=0), # 4*4
            nn.Conv2d(96,96,kernel_size=3,stride=1,padding=0), # 2*2
            nn.AdaptiveAvgPool2d((1,1)) # 1*1
        )
        self.fc00 = nn.Linear(96+1,1024)
        
        # part2
        self.conv02 = nn.Conv2d(len(channels2),64,kernel_size=5,padding=2)
        self.conv0p2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.i02 = Inception(64,64,(48,64),(48,64),64,True)
        self.i12 = Inception(256,92,(64,92),(64,92),64,True)
        self.i1p2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.i22 = Inception(340,128,(92,128),(92,128),92,True)
        self.i32 = Inception(476,128,(92,128),(92,128),92,True)
        self.i3p2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.i42 = Inception(476,128,(92,128),(None,None),92,False) # 348*8*8
        
       
        self.convseq2 = nn.Sequential(
            nn.Conv2d(348,96,kernel_size=3,stride=1,padding=0), # 6*6
            nn.Conv2d(96,96,kernel_size=3,stride=1,padding=0), # 4*4
            nn.Conv2d(96,96,kernel_size=3,stride=1,padding=0), # 2*2
            nn.AdaptiveAvgPool2d((1,1)) # 1,1
        )

        self.fc01 = nn.Linear(96+1,1024)
        
        self.fc0 = nn.Linear(2048,4096)
        self.fc1 = nn.Linear(4096,1024)

    
    def forward(self,X,ebv):
        # split X into two parts:
        X_1 = X[:,self.channels1,:,:]
        X_2 = X[:,self.channels2,:,:]
        ebv = ebv.view(ebv.size(0),1)

        X_1 = F.relu(self.conv01(X_1))
        X_1 = self.conv0p1(X_1) 
        X_1 = self.i01(X_1)
        X_1 = self.i1p1(self.i11(X_1))
        X_1 = self.i21(X_1)
        X_1 = self.i3p1(self.i31(X_1))
        X_1 = self.i41(X_1)
        X_1 = self.convseq1(X_1)
        X_1 = X_1.view(X_1.size(0), -1) # (N,96)
        
        X_1 = torch.cat([X_1,ebv],dim=1) # (N,97)
        X_1 = self.fc00(X_1) # 1024
        # X_1 = nn.Flatten()(X_1) # 348*8*8

        # part2:
        X_2 = F.relu(self.conv02(X_2))
        X_2 = self.conv0p2(X_2) 
        X_2 = self.i02(X_2)
        X_2 = self.i1p2(self.i12(X_2))
        X_2 = self.i22(X_2)
        X_2 = self.i3p2(self.i32(X_2))
        X_2 = self.i42(X_2)
        X_2 = self.convseq1(X_2)
        X_2 = X_2.view(X_2.size(0), -1) # (N,96)
        X_2 = torch.cat([X_2,ebv],dim=1) # (N,97)
        X_2 = self.fc01(X_2) # 1024
        
        
        # X_2 = nn.Flatten()(X_2) # 348*8*8

        # concat two parts together:
        X = torch.cat([X_1,X_2],dim=1)
        X = F.relu(self.fc0(X))
        X = nn.Dropout(p=0.3)(X) # dropout
        X = F.relu(self.fc1(X)) 


        return X


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.1)  


class EstimateModel(nn.Module):
    def __init__(self,Nbins):
        super(EstimateModel,self).__init__()
        self.fc1 = nn.Linear(512+1024,1024) # 1024+2048
        self.fc2 = nn.Linear(1024,Nbins)
        
    def forward(self,output):
        rep = F.relu(self.fc1(output))
        output = self.fc2(rep)

        return output,rep



def model_load_weights(pretrained_dict,prefix,current_dict):
    new_dict = {k[len(prefix)+1:]:v for k,v in pretrained_dict.items() if k.startswith(prefix)}
    assert (new_dict.keys() == current_dict.keys())
    current_dict.update(new_dict)
    return current_dict


class Multimodal_model(nn.Module):
    def __init__(self,channels1,channels2,features_num,Nbins):
        super(Multimodal_model,self).__init__()
        self.channels1 = channels1
        self.channels2 = channels2
        self.features_num = features_num
        self.Nbins = Nbins
        self.image_encoder = image_CNN(self.channels1,self.channels2)
        self.catalog_encoder = catalog_MLP(features_num)
        self.estimate_model = EstimateModel(self.Nbins)

    def load_weights(self,finetune=False,image_weights=None,catalog_weights=None,freeze=False):
        if finetune:
            # update image encoder weight
            image_dict = torch.load(image_weights)
            image_dict = model_load_weights(image_dict,'image_cnn',self.image_encoder.state_dict())
            self.image_encoder.load_state_dict(image_dict)
            # update catalog encoder weight
            catalog_dict = torch.load(catalog_weights)
            catalog_dict = model_load_weights(catalog_dict,'catalog_mlp',self.catalog_encoder.state_dict())
            self.catalog_encoder.load_state_dict(catalog_dict)
        if freeze:
            # freeze weights
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.catalog_encoder.parameters():
                param.requires_grad = False

    def forward(self,images,values):
        
        image_value = self.image_encoder(images,values[:,-1]) # ebv
        catalog_value = self.catalog_encoder(values) 

        output = torch.cat([image_value,catalog_value],dim=1)
        output,rep = self.estimate_model(output)

        return output,rep



