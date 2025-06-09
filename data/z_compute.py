import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import torch

def set_zbins(z_min,z_max,Nbins):
    """
    set redshift bin
    """
    bin_edges = np.linspace(z_min,z_max,Nbins+1) # Nbins+1个值
    bin_width = np.mean(np.diff(bin_edges))

    return bin_edges,bin_width

def z_encode(bin_edges,labels,Nbins):
    """
    Assign the redshift values into predefined bins.
    """
    label_encode = np.digitize(labels,bin_edges,right = True) -1  # 0-based. [0,Nbins-1]. digitize函数：用bins给labels编码，right=True表示在bins是升序的情况下是右等号的，即bin[i-1]<x<=bin[i]。https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
    label_encode = np.clip(label_encode,0,Nbins-1) # force clip
    return label_encode


def compute_zbins_midpoint(bin_edges):
    """
    Calculate the median value for each redshift bin.
    """
    zbins_midpoint = (bin_edges[1:]+bin_edges[:-1])/2 # 计算每个bin对应的中值，共Nbins个值
    return zbins_midpoint

def check_z_dist(labels,model_name):
    """
    Inspect the dataset: plot the redshift distribution of the dataset.
    labels:[all_labels,train_labels,valid_labels,test_labels]
    """
    z_min = torch.min(labels[0])
    z_max = torch.max(labels[0])
    print(f"z_min:{z_min} z_max:{z_max}")

    # check redshfit distribution in dataset
    name = ['all','train','valid','test']
    fig,ax = plt.subplots(4,1)
    plt.tight_layout()
    for i in range(len(name)):
        ax[i].hist(labels[i],bins=50)
        ax[i].set_title(f'{name[i]}_z_distribution')
    plt.savefig(f'./output/{model_name}/{model_name}_z_distribution.png')

    # kstest:
    print("ks test for train and valid",sep=',')
    print(ks_2samp(labels[1],labels[2])) # train vs valid
    print("ks test for train and test",sep=',')
    print(ks_2samp(labels[1],labels[3])) # train vs test
    print("ks test for valid and test",sep=',')
    print(ks_2samp(labels[2],labels[3])) # valid vs test
    
    return z_min,z_max



def compute_z(labels,z_min,z_max,config):
    """
    labels:[train_labels,valid_labels,test_labels]
    """
    # z value to z encode:
    if config['Data']['Z_USE_DEFAULT']: # Use the predefined default values.
        z_min = config['Data']['Z_MIN']
        z_max = config['Data']['Z_MAX']
    
    # set zbins
    if config['Data']['BIN_DEFAULT']:
        bin_edges = np.array(config['Data']['BIN_EDGES'],dtype=np.float32)
        bin_width = np.mean(np.diff(bin_edges))
        Nbins = len(bin_edges)-1
    else:
        bin_width = config['Data']['BIN_WIDTH_DEFAULT'] # zbins_gap in default
        Nbins = np.min([int(np.ceil((z_max-z_min)/bin_width)),config['Data']['NBINS_MAX']]) # Take the minimum between Nbins and the default value.
        bin_edges,bin_width = set_zbins(z_min,z_max,Nbins) # set zbins

    print(f"There are {len(bin_edges)-1} bins and bin width is {bin_width}")
    print(bin_edges)
    
    # compute zbin midpoint:
    zbins_midpoint = compute_zbins_midpoint(bin_edges)

    # save midpoint:
    np.save(f"./output/{config['Experiment']['Run_name']}/{config['Experiment']['Run_name']}_zbins_midpoint.npy",zbins_midpoint)
    # save zbins:
    np.save(f"./output/{config['Experiment']['Run_name']}/{config['Experiment']['Run_name']}_zbins.npy",bin_edges)

    # encode z labels:
    encode_labels = []
    for l in range(len(labels)):
        encode_labels.append(z_encode(bin_edges,labels[l],Nbins))
    
    return encode_labels,zbins_midpoint,Nbins





        

