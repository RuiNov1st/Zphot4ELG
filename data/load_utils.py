import torch
import numpy as np
import yaml
import pandas as pd
import os

def load_data(img_path,label_path,data_type):
    """
    载入数据
    """
    images = np.load(img_path)
    labels = np.load(label_path)

    if data_type == 'DESI':
        images = images[:,:,:,:4] # g,r,i,z
    elif data_type == 'WISE_COLOR-WISE':
        images = images[:,:,:,[0,1,2,3,6,7,8]] # g,r,i,z,g-r,r-i,i-z 
    elif data_type == 'DESI_COLOR':
        images = images[:,:,:,:7] # g,r,i,z,g-r,r-i,i-z 
    elif data_type == 'WISE':
        images = images[:,:,:,:6] # g,r,i,z,w1,w2
    elif data_type == 'WISE_COLOR':
        images = images[:,:,:,:11] # g,r,i,z,w1,w2,g-r,r-i,i-z,z-w1,w1-w2

    return images,labels

def load_catalog(catalog_path,colsname=['MAG_R']):
    df = pd.read_csv(catalog_path)
    # 读取其中所需要的列：
    arr = df[colsname[0]].values
    arr = arr.reshape(-1,1)
    if len(colsname)>1:
        for i in range(1,len(colsname)):
            arr = np.concatenate([arr,df[colsname[i]].values.reshape(-1,1)],axis=1)
    return arr,df


def image_channel_select(images,data_type):
    if data_type == 'DESI':
        images = images[:,:,:,:4] # g,r,i,z
    elif data_type == 'WISE_COLOR-WISE':
        images = images[:,:,:,[0,1,2,3,6,7,8]] # g,r,i,z,g-r,r-i,i-z 
    elif data_type == 'DESI_COLOR':
        images = images[:,:,:,:7] # g,r,i,z,g-r,r-i,i-z 
    elif data_type == 'WISE':
        images = images[:,:,:,:6] # g,r,i,z,w1,w2
    elif data_type == 'WISE_COLOR':
        images = images[:,:,:,:11] # g,r,i,z,w1,w2,g-r,r-i,i-z,z-w1,w1-w2
    
    return images


def read_split_data(path,data_type):
    """
    read data that already split
    """
    data_list = []
    data_pattern = ['images1','labels1']
    dataset_name = ['train','valid','test']
    labels_list = [] # for check_z_dist
    for n in range(len(dataset_name)):
        data_list_p = []
        for d in range(len(data_pattern)):
            tmp_data = np.load(os.path.join(path,f'{data_pattern[d]}_{dataset_name[n]}.npy'))
            if 'images' in data_pattern[d]:
                tmp_data = image_channel_select(tmp_data,data_type)
                tmp_data = torch.tensor(tmp_data).permute(0,3,1,2) # n,channel,width,height
            
            if 'labels' in data_pattern[d]:
                tmp_data = torch.Tensor(tmp_data).to(torch.float32) # to float32
                labels_list.append(tmp_data)
                
            
            data_list_p.append(tmp_data) # [images,labels]
        
        data_list.append(data_list_p) # [train,valid,test]
    
    label_arr = labels_list[0]
    for i in range(1,len(labels_list)):
        label_arr = np.concatenate([label_arr,labels_list[i]],axis=0)

    label_arr = torch.Tensor(label_arr).to(torch.float32)

   
    
    return data_list,label_arr


def read_split_catalog(path,colsname=['MAG_R']):
    df_list = []
    arr_list = []
    indices = []
    dataset_name = ['train','valid','test']
    length = 0
    for n in range(len(dataset_name)):
        # catalog:
        tmp_df = pd.read_csv(os.path.join(path,f'{dataset_name[n]}_df.csv'))
        df_list.append(tmp_df)
        # catalog data:
        tmp_arr = tmp_df[colsname[0]].values
        tmp_arr = tmp_arr.reshape(-1,1)
        if len(colsname)>1:
            for i in range(1,len(colsname)):
                tmp_arr = np.concatenate([tmp_arr,tmp_df[colsname[i]].values.reshape(-1,1)],axis=1)
        arr_list.append(tmp_arr)

        # index:
        indices.append([i for i in range(length,length+len(tmp_df))])
        length += len(tmp_df)

    print(f"training:{len(indices[0])} valid:{len(indices[1])} test:{len(indices[2])}")
    df = pd.concat([i for i in df_list]).reset_index(drop=True)
    
    df.to_csv(os.path.join(path,'df.csv'),index=False)

    return arr_list,indices,df




def read_config():
    """
    读取配置
    """
    with open("../config.yaml") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


# compute mean & std for normalize:
def compute_mean_std(images):
    # required images shape: (n,channel,width,height)
    mean = images.mean(dim=[0, 2, 3])
    std = images.std(dim=[0,2,3])

    return mean,std


