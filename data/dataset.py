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
from data.z_compute import compute_z,check_z_dist
from data.mag_compute import check_dataset_magnitude_dist
from data.filter import data_filter
import random

def image_Augmentation():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90, fill=(0,))  # Random rotation with fill mode set to 0

    ])
    
    
# def image_Transform(mean=None,std=None):
#     return transforms.Compose([
#             transforms.ToTensor(), # 0-1 
#             transforms.Normalize(mean=mean,std=std)
#         ])
    
def image_Transform():
    return transforms.Compose([
        # crop to 32*32
        transforms.CenterCrop(32),
        transforms.Resize(64)        # Resize back to 64x64
    ])


# ensure dataloader's reproducibility :https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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



class AstroDataset(Dataset):
    def __init__(self, images,labels,labels_encode,catalog_data,indices,augmentation=None,transform=None):
        self.images = images
        self.labels = labels
        self.labels_encode = labels_encode
        self.catalog_data = catalog_data
        self.indices = indices
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
            if image.shape[0]==10: # 只对WISE部分crop
                image_1 = image[:7,:,:]
                image_2 = image[7:,:,:]
                image_2 = self.transform(image_2)

                image = torch.cat([image_1,image_2],dim=0)

        label = self.labels[idx]
        label_encode = self.labels_encode[idx]
        
        catalog_data = self.catalog_data[idx]
        indice = self.indices[idx]

        # set z flag:
        z_flag = 0 if label<=0.5 else 1
        return image.to(torch.float32), label.to(torch.float32),label_encode,torch.Tensor(catalog_data),indice,z_flag
       



def make_dataset(config):
    # load image and label
    images,labels = load_data(config['Data']['IMG_PATH'],
        config['Data']['LABEL_PATH'],
        config['Data']['DATA_TYPE']
    )

    # reshape images：
    images = torch.tensor(images).permute(0,3,1,2) # n,channel,width,height

    # convert labels type:
    labels = torch.Tensor(labels).to(torch.float32) # to float32
    
    # load catalog:
    catalog_data,catalog = load_catalog(config['Data']['CATALOG_PATH'],config['Data']['CATALOG_COLUMN'])

    # StandardScaler:
    # scaler = StandardScaler()
    # catalog_data = scaler.fit_transform(catalog_data)
    # if config['Data']['DATA_FILTER']:
    # [images,labels,catalog_data],catalog = data_filter([images,labels,catalog_data],catalog)

    # indices: # re-index after filter
    indices = np.arange(len(images))

    # print shape:
    print(images.shape,labels.shape,catalog_data.shape,indices.shape)

    # data split:
    train_data,valid_data,test_data = data_split(indices,[images,labels,catalog_data,indices],config)

    # # compute mean and std: only compute training set and apply it to validation and test set
    # image_mean,image_std =  compute_mean_std(train_data[0])
    # print(image_mean,image_std)

    # check z:
    z_min,z_max = check_z_dist([labels,train_data[1],valid_data[1],test_data[1]],config['Experiment']['Run_name'])
    # encode:
    label_encode,zbins_midpoint,Nbins = compute_z([train_data[1],valid_data[1],test_data[1]],z_min,z_max,config)
    if config['Model']['MODEL_TYPE'] == 'Regression': # regression
        Nbins = 1
    
    # check magnitude:
    check_dataset_magnitude_dist(catalog,train_data[3],valid_data[3],test_data[3],config['Experiment']['Run_name'])

    # make dataset:
    # image_transform = image_Transform(image_mean,image_std)
    train_dataset = AstroDataset(images=train_data[0],labels=train_data[1],labels_encode = label_encode[0], catalog_data=train_data[2],indices=train_data[3],augmentation=image_Augmentation(),transform=None)
    valid_dataset = AstroDataset(images=valid_data[0],labels=valid_data[1],labels_encode = label_encode[1],catalog_data=valid_data[2],indices=valid_data[3],augmentation=None,transform=None)
    test_dataset = AstroDataset(images=test_data[0],labels=test_data[1],labels_encode = label_encode[2],catalog_data=test_data[2],indices=test_data[3],augmentation=None,transform=None)

    # sampler for DDP:
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    # dataloader:
    # train_loader = DataLoader(train_dataset,batch_size=config['Train']['BATCH_SIZE'],sampler=train_sampler)
    train_loader = DataLoader(train_dataset,batch_size=config['Train']['BATCH_SIZE'],shuffle=True,num_workers=4,pin_memory=True,worker_init_fn=worker_init_fn)

    valid_loader = DataLoader(valid_dataset,batch_size=config['Train']['BATCH_SIZE'],shuffle=True,num_workers=4,pin_memory=True,worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset,batch_size=config['Train']['BATCH_SIZE'],shuffle=True,num_workers=4,pin_memory=True,worker_init_fn=worker_init_fn)

    # dataset_info:
    channels = int(images.shape[1])
    features_num = int(len(config['Data']['CATALOG_COLUMN']))
    
    return train_loader,valid_loader,test_loader,catalog,zbins_midpoint,Nbins,channels,features_num

def make_dataset_v2(config):
    """
    already have splited train, valid and test dataset
    """
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


        z_min,z_max = check_z_dist([label_arr,train_data[1],valid_data[1],test_data[1]],config['Experiment']['Run_name'])
        label_encode,zbins_midpoint,Nbins = compute_z([train_data[1],valid_data[1],test_data[1]],z_min,z_max,config)
        
        if config['Model']['MODEL_TYPE'] == 'Regression': # regression
            Nbins = 1

        # check magnitude:
        check_dataset_magnitude_dist(catalog,train_data[3],valid_data[3],test_data[3],config['Experiment']['Run_name'])

        # make dataset:
        train_dataset = AstroDataset(images=train_data[0],labels=train_data[1],labels_encode = label_encode[0], catalog_data=train_data[2],indices=train_data[3],augmentation=image_Augmentation(),transform=None)
        valid_dataset = AstroDataset(images=valid_data[0],labels=valid_data[1],labels_encode = label_encode[1],catalog_data=valid_data[2],indices=valid_data[3],augmentation=None,transform=None)
        test_dataset = AstroDataset(images=test_data[0],labels=test_data[1],labels_encode = label_encode[2],catalog_data=test_data[2],indices=test_data[3],augmentation=None,transform=None)

        
        train_loader = DataLoader(train_dataset,batch_size=config['Train']['BATCH_SIZE'],shuffle=True,num_workers=4,pin_memory=True,worker_init_fn=worker_init_fn)
        valid_loader = DataLoader(valid_dataset,batch_size=config['Train']['BATCH_SIZE'],shuffle=True,num_workers=4,pin_memory=True,worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_dataset,batch_size=config['Train']['BATCH_SIZE'],shuffle=True,num_workers=4,pin_memory=True,worker_init_fn=worker_init_fn)

        # dataset_info:
        channels = int(train_data[0].shape[1])
        features_num = int(len(config['Data']['CATALOG_COLUMN']))

        return train_loader,valid_loader,test_loader,catalog,zbins_midpoint,Nbins,channels,features_num


   

if __name__ =='__main__':
    config = read_config()
    train_dataset, val_dataset, test_dataset = make_dataset(config)
    


