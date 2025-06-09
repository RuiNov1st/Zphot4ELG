"""
Description: split dataset advanced for Hierarchical classification
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
import matplotlib.pyplot as plt

def split_dataset(df,data,data_pattern,output_path):
    df['zspec_bin'] = pd.cut(df['z_spec'],bins=np.arange(0.,4.25,0.25)) # setting by myself
    bins_count = df['zspec_bin'].value_counts()

    train_list = []
    valid_list = []
    test_list = []
    for idx in bins_count.index:
        bin_data = df[df['zspec_bin']==idx]
        if len(bin_data)>=5:
            train_idx,test_idx = train_test_split(bin_data.index,test_size=0.2,shuffle=True,random_state=42)
            train_idx,valid_idx = train_test_split(train_idx,test_size=0.06,shuffle=True,random_state=42) # train:valid:test = 6:0.06:2
        else:
            train_idx = bin_data.index
            valid_idx,test_idx = [],[]
        if len(train_idx)!=0:
            train_list.extend(train_idx.tolist()) 
        if len(valid_idx)!=0:
            valid_list.extend(valid_idx.tolist()) 
        if len(test_idx)!=0:
            test_list.extend(test_idx.tolist()) 
    
    # df:
    train_df = df.iloc[train_list]
    valid_df = df.iloc[valid_list]
    test_df = df.iloc[test_list]

    # data:
    train_new_data = []
    valid_new_data = []
    test_new_data = []
    for d in data:
        train_new_data.append(d[train_list])
        valid_new_data.append(d[valid_list])
        test_new_data.append(d[test_list])

    # save:
    train_df.to_csv(os.path.join(output_path,'train_df.csv'),index=False)
    valid_df.to_csv(os.path.join(output_path,'valid_df.csv'),index=False)
    test_df.to_csv(os.path.join(output_path,'test_df.csv'),index=False)


    for d in range(len(data_pattern)):
        np.save(os.path.join(output_path,f'{data_pattern[d]}_train.npy'),train_new_data[d])
    for d in range(len(data_pattern)):
        np.save(os.path.join(output_path,f'{data_pattern[d]}_valid.npy'),valid_new_data[d])
    for d in range(len(data_pattern)):
        np.save(os.path.join(output_path,f'{data_pattern[d]}_test.npy'),test_new_data[d])


    # check:
    print(f"train:{len(train_list)} valid:{len(valid_list)} test:{len(test_list)}")
    sns.kdeplot(train_df['z_spec'], fill=True,color='r', label = 'train')
    sns.kdeplot(valid_df['z_spec'],  fill=True, color='g', label = 'valid')
    sns.kdeplot(test_df['z_spec'],  fill=True,color='b', label = 'test')
    plt.xlabel('z_spec')
    plt.legend()
    plt.savefig(os.path.join(output_path,'dataset_dist.png'))

    

   

    
