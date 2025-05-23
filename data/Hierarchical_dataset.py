"""
Description: design for Hierarchial classification.

"""
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import sys 
from sklearn.preprocessing import StandardScaler
from data.load_utils import load_data,load_catalog,read_config,compute_mean_std,read_split_data,read_split_catalog
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from data.z_compute import compute_z,check_z_dist,z_encode,setzbins_4hierarchical
from data.mag_compute import check_dataset_magnitude_dist
from data.filter import data_filter

def image_Augmentation():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90, fill=(0,))  # Random rotation with fill mode set to 0
    ])
    

def data_split(indices,data,config):
    # split indices:
    # split train and test
    indices_train,indices_test = train_test_split(indices,shuffle=True,test_size=config['Data']['TEST_SIZE'], random_state=42) 
    # split train and valid
    indices_train,indices_valid = train_test_split(indices_train,shuffle=True,test_size=config['Data']['VALIDATION_SIZE'],random_state=42)

    # split data
    train_data,valid_data,test_data = [],[],[]
    for d in data:
        train_data.append(d[indices_train])
        valid_data.append(d[indices_valid])
        test_data.append(d[indices_test])
    
    print(f"training set size:{len(indices_train)} \t validation set size:{len(indices_valid)} \t test set size:{len(indices_test)}")

    return train_data,valid_data,test_data



class AstroDataset_Hierarchical(Dataset):
    def __init__(self, images,labels,catalog_data,indices,classification_bin,estimation_bins,augmentation=None,transform=None):
        self.images = images
        self.labels = labels
        self.catalog_data = catalog_data
        self.indices = indices
        self.classification_bin = classification_bin
        self.estimation_bins = estimation_bins
        self.augmentation  =  augmentation
        self.transform = transform


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        # augmentation:
        if self.augmentation:
            image = self.augmentation(image)
        # normalize and totensor:0-1
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        catalog_data = self.catalog_data[idx]
        indice = self.indices[idx]

        # z encode:
        classification_encode = z_encode(self.classification_bin,label,len(self.classification_bin)-1)
        # according to classification level encode result choose suitable estimation level bins to encode:
        estimation_encode = z_encode(self.estimation_bins[classification_encode],label,len(self.estimation_bins[classification_encode])-1)
        label_encode = np.array([classification_encode,estimation_encode])
        
        return image.to(torch.float32), label.to(torch.float32),label_encode,torch.Tensor(catalog_data),indice


def make_dataset(config):
    if config['Data']['Already_split']:
        data_list,label_arr = read_split_data(config['Data']['PATH'],config['Data']['DATA_TYPE'])
        train_data,valid_data,test_data = data_list[0],data_list[1],data_list[2]
        catalog_data_list,indices,catalog = read_split_catalog(config['Data']['PATH'],config['Data']['CATALOG_COLUMN'])

        train_data.append(catalog_data_list[0])
        train_data.append(indices[0])  # [images,labels,catalog_data,indices]
        valid_data.append(catalog_data_list[1])
        valid_data.append(indices[1])
        test_data.append(catalog_data_list[2])
        test_data.append(indices[2])
    
    # check z:
    z_min,z_max = check_z_dist([label_arr,train_data[1],valid_data[1],test_data[1]],config['Experiment']['Run_name'])
    # set zbins:
    classification_bin,estimation_bins,zbins_midpoints,classification_Nbins,estimation_Nbins = setzbins_4hierarchical(config)
    
    # check magnitude:
    check_dataset_magnitude_dist(catalog,train_data[3],valid_data[3],test_data[3],config['Experiment']['Run_name'])

    # make dataset:
    train_dataset = AstroDataset_Hierarchical(images=train_data[0],labels=train_data[1],catalog_data=train_data[2],indices=train_data[3],classification_bin=classification_bin,estimation_bins=estimation_bins,augmentation=image_Augmentation(),transform=None)
    valid_dataset = AstroDataset_Hierarchical(images=valid_data[0],labels=valid_data[1],catalog_data=valid_data[2],indices=valid_data[3],classification_bin=classification_bin,estimation_bins=estimation_bins,augmentation=None,transform=None)
    test_dataset = AstroDataset_Hierarchical(images=test_data[0],labels=test_data[1],catalog_data=test_data[2],indices=test_data[3],classification_bin=classification_bin,estimation_bins=estimation_bins,augmentation=None,transform=None)

    train_loader = DataLoader(train_dataset,batch_size=config['Train']['BATCH_SIZE'],shuffle=True,num_workers=4,pin_memory=True)
    valid_loader = DataLoader(valid_dataset,batch_size=config['Train']['BATCH_SIZE'],shuffle=True,num_workers=4,pin_memory=True)
    test_loader = DataLoader(test_dataset,batch_size=config['Train']['BATCH_SIZE'],shuffle=True,num_workers=4,pin_memory=True)

    # dataset_info:
    channels = int(train_data[0].shape[1])
    features_num = int(len(config['Data']['CATALOG_COLUMN']))
    
    return train_loader,valid_loader,test_loader,catalog,zbins_midpoints,channels,features_num,classification_Nbins,estimation_Nbins


if __name__ =='__main__':
    config = read_config("/data/home/wsr/Workspace/dl/Algorithm/Henghes22/config/Hierarchical_config.yaml")
    train_loader,valid_loader,test_loader,catalog,zbins_midpoints,channels,features_num,classification_Nbins,estimation_Nbins = make_dataset(config)
    print(channels,features_num,classification_Nbins,estimation_Nbins)

    


