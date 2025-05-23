import numpy as np
import matplotlib.pyplot as plt
# from keras.utils.vis_utils import plot_model
import gc
''' modules for scatter density'''
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib import cm
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import yaml
import torch
import wandb
import random
import os
from PIL import Image
os.environ["WANDB_API_KEY"] = "your wandb api key"

def read_config(config_path="./config.yaml"):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def write_config(config):
    """
    Write the config file of each experiment to disk for record-keeping and future reference.
    """
    with open(f"./output/{config['Experiment']['Run_name']}/config.yaml", 'w',encoding='utf-8') as yaml_file:
        yaml.dump(config, yaml_file,allow_unicode=True)
    
    print("write config.yaml successfully!")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_gpu(device_list=None):
    if torch.cuda.is_available():
        if device_list is not None:
            device_ids = device_list # set device list by yourself
        else: 
            device_ids = list(range(torch.cuda.device_count()))
        print(f"Using GPU. CUDA NUMBER:{len(device_ids)}")
        return device_ids
    else:
        device = torch.device("cpu")
        print("using cpu")
    return device

    
def set_wandb(config,resume_run=False):
    """
    wandb setup
    """
    wandb.login()
    # Resume run
    if resume_run:
        run = wandb.init(
                project=config['Experiment']['Project_name'], 
                id=config['Experiment']['Run_id'], 
                resume="must")
    # new run
    else:
        run = wandb.init(
            project = config['Experiment']['Project_name'],
            config = config,
            notes = config['Experiment']['Description'],
            name = config['Experiment']['Run_name'],
            group = config['Experiment']['Group']
        )
        # write run id into config
        config['Experiment']['Run_id'] = wandb.run.id
    return run,config


def upload_wandb(config,run,bias,nmad,outlier):
    model_name = config['Experiment']['Run_name']
    # metrics:
    wandb.define_metric("Bias",summary = "best")
    wandb.define_metric("NMAD",summary = "best")
    wandb.define_metric("Outlier fraction",summary="best")

    log_dict = {
        "Bias":bias,
        "NMAD":nmad,
        "Outlier fraction":outlier
    }
    wandb.log(log_dict,commit=False)

    # images:
    images_dict = {
        "train_metrics_monitor":f'./output/{model_name}/{model_name}_loss_acc.png',
        "pdf_check":f'./output/{model_name}/check_pdf_{model_name}.png',
        "z plot":f'./output/{model_name}/Plot_{model_name}.png',
        "Residuals":f'./output/{model_name}/Residuals_{model_name}.png',
        "outlier_analysis":f'./output/{model_name}/outlier_analysis_{model_name}.png',
        "metrics-z":f'./output/{model_name}/{model_name}_metrics-z.png'

    }
    for key,value in images_dict.items():
        if os.path.exists(value):
            wandb.log({key:wandb.Image(Image.open(value))})

    # log
    arti_log = wandb.Artifact('log',type='log')
    arti_log.add_file('./logs/log.txt',name=config['Experiment']['Run_name']+"_log")
    run.log_artifact(arti_log)


def write_result(test_labels,pred_red,indices_test,probability,rep_arr,model_name):
    file_name = f'./output/{model_name}/{model_name}.npz'
    np.savez(file_name,z_true=test_labels,z_pred=pred_red,indices = indices_test,pdf = probability,representation=rep_arr)
    print(f"save {file_name} success!")


def log_result(deltaz,bias,nmad,outlier,model_name):
    log_file = open(f'./output/{model_name}/{model_name}_log_result.txt','w')
    log_file.write(f'Bias:{bias}\n')
    log_file.write(f'NMAD:{nmad}\n')
    log_file.write(f'Outlier fraction:{outlier}\n')
    log_file.close()
    print(f'log ./output/{model_name}/{model_name}_log_result.txt success!')