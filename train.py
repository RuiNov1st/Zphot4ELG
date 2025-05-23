from data.dataset import make_dataset_v2
import numpy as np
import os
import math
import sys
import wandb # weights & bias experiment track
from PIL import Image
import pandas as pd
from models.model import Multimodal_model,init_weights
import torch
from utils import set_gpu,read_config,log_result,write_result,set_wandb,upload_wandb,write_config
from torch import nn
import torch.nn.functional as F
from visual import training_monitor
from metrics import compute_metrics,metrics_z_plot,make_plot,check_probability
from outlier_analysis import outlier_analysis
from torch.nn.parallel import DataParallel
import time
# from torchinfo import summary



def evaluation(test_loader,model_checkpoint,config,zbins_midpoint,Nbins,channels,features_num,device):
    run_name = config['Experiment']['Run_name']

    model = Multimodal_model([i for i in range(7)],[i for i in range(7,10)],features_num,Nbins)

    model.load_state_dict(torch.load(model_checkpoint))

    model = model.to(device[0])
    if len(device)>1:
        model = torch.nn.DataParallel(model,device_ids=device)

    
    if config['Model']['MODEL_TYPE'] == 'Classification':
        loss_fn = nn.CrossEntropyLoss()
    elif config['Model']['MODEL_TYPE'] == 'Regression':
        loss_fn = nn.MSELoss()
        assert Nbins == 1

    test_loss = 0.
    correction_predictions = 0
    total_samples = 0

    # use for metrics compute:
    image_arr = []
    output_arr = []
    label_arr = []
    indice_arr = []
    rep_arr = [] # representation

    model.eval()
    with torch.no_grad():
        for idx,data in enumerate(test_loader):
            image, label,label_encode,catalog_data,indice,_ = data
            target = label if config['Model']['MODEL_TYPE']  == 'Regression' else label_encode

            assert target is not None
            # to cuda:
            image,target,catalog_data = image.to(device[0]),target.to(device[0]),catalog_data.to(device[0])

            output,rep = model(image,catalog_data)


            if config['Model']['MODEL_TYPE']  == 'Classification':
                probability = F.softmax(output,dim=1) # pdf

            # loss = torch.sqrt(loss_fn(output.view(-1),label)) # regression
            if config['Model']['MODEL_TYPE'] == 'Regression':
                loss = torch.sqrt(loss_fn(output.view(-1),target))
            else:
                loss = loss_fn(output,target)

            test_loss+=loss.item()

            # log data:
            image_arr.append(image)
            output_arr.append(probability if config['Model']['MODEL_TYPE']  == 'Classification' else output)
            label_arr.append(label)
            indice_arr.append(indice)
            rep_arr.append(rep)
        
            if config['Model']['MODEL_TYPE'] == 'Classification':
                predictions = torch.argmax(output,dim=1)
                correction_predictions  += (predictions == target).sum().item()
                total_samples += target.size(0)
            
                    
        print(f"loss/test:{test_loss/(idx+1)}")
        if config['Model']['MODEL_TYPE'] == 'Classification':
            accuracy = correction_predictions / total_samples
            print(f"accuracy/test:{accuracy}",end=' ')

    
    image_arr = np.array(torch.cat(image_arr,dim=0).cpu())
    if config['Model']['MODEL_TYPE']  == 'Classification':
        probability_arr = np.array(torch.cat(output_arr, dim=0).cpu())  # Concatenate along the first dimension
        pred_red = np.sum(probability_arr*zbins_midpoint,axis=1)
    else:
        probability_arr = None
        pred_red = np.array(torch.cat(output_arr, dim=0).cpu().view(-1))
    true_red = np.array(torch.cat(label_arr, dim=0).cpu())    # Concatenate labels
    indice_arr = np.array(torch.cat(indice_arr, dim=0).cpu())  # Concatenate indices
    rep_arr = np.array(torch.cat(rep_arr,dim=0).cpu())
   
    print(pred_red.shape,true_red.shape)

    # metrics compute:
    deltaz,bias,nmad,outlier = compute_metrics(pred_red,true_red)
    # function of metrics and redshift：
    metrics_z_plot(pred_red,true_red,run_name)
    # plot
    make_plot(pred_red,true_red,deltaz,bias,nmad,outlier,run_name)
    # pdf:
    if config['Model']['MODEL_TYPE'] == 'Classification':
        check_probability(image_arr,pred_red,true_red,probability_arr,zbins_midpoint,run_name)
    # output
    log_result(deltaz,bias,nmad,outlier,run_name)
    write_result(true_red,pred_red,indice_arr,probability_arr,rep_arr,run_name)
    # analyse outliers' characteristics:
    if config['Data'].get('CATALOG_PATH') is not None:
        outlier_analysis(f'./output/{run_name}/{run_name}.npz',config['Data']['CATALOG_PATH'],run_name)
    else:
        outlier_analysis(f'./output/{run_name}/{run_name}.npz',os.path.join(config['Data']['PATH'],'df.csv'),run_name)

    
    return bias,nmad,outlier

class EarlyStopper:
    def __init__(self, patience=1):
        self.patience = patience
        self.min_delta = float('inf')
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.min_delta = 0.*self.min_validation_loss
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def train(train_dataloader,valid_dataloader,config,Nbins,channels,features_num,device):
    
    # define model:
    model = Multimodal_model([i for i in range(7)],[i for i in range(7,10)],features_num,Nbins)
    
    # use pretrained weights for subnetworks:
    model.load_weights(finetune=True,image_weights=config['Model']['ImageEncoder_Weights'],catalog_weights=config['Model']['CatalogEncoder_Weights'],freeze=False)


    # finetune in pretrained weights:
    if config['Train']['CONTINUE_TRAIN']:
        model.load_state_dict(torch.load(config['Train']['CONTINUE_CHECKPOINT']))
    # else:
    #     model.apply(init_weights) # init

    model = model.to(device[0])

    if len(device)>1:
        model = torch.nn.DataParallel(model,device_ids=device)
        print("DP setting finish")
    
    model_save_path = f"./weights/{config['Experiment']['Run_name']}.pth"

    # define loss and optimizer
    if config['Model']['MODEL_TYPE'] == 'Classification':
        loss_fn = nn.CrossEntropyLoss()
    elif config['Model']['MODEL_TYPE'] == 'Regression':
        loss_fn = nn.MSELoss()
        assert Nbins == 1

    optimizer = torch.optim.Adam(model.parameters(),lr=config['Train']['LEARNING_RATE'],weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    running_loss_list = []
    val_loss_list = []
    best_epoch = 0
    best_vloss = np.inf
    
    validation_freq = 1
    # early stopper:
    early_stopper = EarlyStopper(patience=10)

    # training:
    for e in range(config['Train']['EPOCH']):
        print(f"Epoch {e}:",end=' ')
        time1 = time.time()
        # make sure gradient tracking is on 
        model.train(True)
        running_loss = 0. # loss around one epoch in training
        correction_predictions = 0
        total_samples = 0
        min_lr = 1e-7

        for idx,data in enumerate(train_dataloader):
            image, label,label_encode,catalog_data,indice,_ = data
            target = label if config['Model']['MODEL_TYPE']  == 'Regression' else label_encode

            assert target is not None
            # to cuda:
            image,target,catalog_data = image.to(device[0]),target.to(device[0]),catalog_data.to(device[0])
            optimizer.zero_grad()

            output,_ = model(image,catalog_data)
            if config['Model']['MODEL_TYPE'] == 'Regression':
                loss = torch.sqrt(loss_fn(output.view(-1),target))
            else:
                loss = loss_fn(output,target)

            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if config['Model']['MODEL_TYPE'] == 'Classification':
                predictions = torch.argmax(output,dim=1)
                correction_predictions  += (predictions == target).sum().item()
                total_samples += target.size(0)

        time2 = time.time()

        # update learning rate
        scheduler.step()

        for param_group in optimizer.param_groups:
            if param_group['lr'] < min_lr:
                param_group['lr'] = min_lr
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {e+1}: Learning rate is {current_lr}")


        print(f"loss/train:{running_loss/(idx+1)} time:{time2-time1}",end=' ')
        wandb.log({"training loss":running_loss/(idx+1)})
        running_loss_list.append(running_loss/(idx+1))

        if config['Model']['MODEL_TYPE'] == 'Classification':
            accuracy = correction_predictions / total_samples
            wandb.log({"training accuracy": accuracy})
            print(f"accuracy/train:{accuracy}",end=' ')


        
        # validation
        if e % validation_freq == 0:
            model.eval() # set the model to evaluation mode
            # disable gradient computation and reduce memory consumption
            with torch.no_grad():
                val_loss = 0.
                val_correction_predictions = 0
                val_total_samples = 0
                for idx,data in enumerate(valid_dataloader):
                    image, label,label_encode,catalog_data,indice,_ = data
                    target = label if config['Model']['MODEL_TYPE']  == 'Regression' else label_encode
                    assert target is not None
                    image,target,catalog_data = image.to(device[0]),target.to(device[0]),catalog_data.to(device[0])
                    voutput,_ = model(image,catalog_data)

                    if config['Model']['MODEL_TYPE'] == 'Regression':
                        vloss = torch.sqrt(loss_fn(voutput.view(-1),target)) # regression
                    else:
                        vloss = loss_fn(voutput,target)


                    val_loss+=vloss.item()

                    
                    if config['Model']['MODEL_TYPE'] == 'Classification':
                        predictions = torch.argmax(voutput,dim=1)
                        val_correction_predictions  += (predictions == target).sum().item()
                        val_total_samples += target.size(0)
                
                val_loss = val_loss/(idx+1)
                val_loss_list.append(val_loss)
                time3 = time.time()
                print(f"loss/valid:{val_loss} time:{time3-time2}")
                wandb.log({"val_loss":val_loss})
                if config['Model']['MODEL_TYPE'] == 'Classification':
                    accuracy = val_correction_predictions / val_total_samples
                    wandb.log({"val accuracy": accuracy})
                    print(f"accuracy/val:{accuracy}",end=' ')


                # gain improvement in validation set:
                if val_loss<best_vloss:
                    best_vloss = val_loss
                    best_epoch = e
                    if len(device)>1:
                        torch.save(model.module.state_dict(),model_save_path)
                    else:
                        torch.save(model.state_dict(),model_save_path)
                    print(f"Best Epoch currently is {e}. Best Val loss:{best_vloss}")
                
                # judge whether to stop:
                if early_stopper.early_stop(val_loss):
                    break
       
    
    # visual loss:
    training_monitor(running_loss_list,val_loss_list,config['Experiment']['Run_name'])


def main(only_test=False,resume_run=False):
    # set GPU device
    device = set_gpu(device_list=[0])
    
    # load config file
    config = read_config('./config/config_split.yaml')

    # create dir for this run
    if not os.path.exists(f"./output/{config['Experiment']['Run_name']}"):
        os.mkdir(f"./output/{config['Experiment']['Run_name']}")
    
    # init wandb:
    run,config = set_wandb(config,resume_run)


    train_loader,val_loader,test_loader,catalog,zbins_midpoint,Nbins,channels,features_num = make_dataset_v2(config) # image,label,catalog,indice
    print("load dataset finish!")

    
    if not only_test:
        train(train_loader,val_loader,config,Nbins,channels,features_num,device)
    

    model_checkpoint = f"./weights/{config['Experiment']['Run_name']}.pth"
    bias,nmad,outlier = evaluation(test_loader,model_checkpoint,config,zbins_midpoint,Nbins,channels,features_num,device)
    
    # update log to wandb：
    upload_wandb(config,run,bias,nmad,outlier)
    
    # write down config to dir
    write_config(config)

    wandb.finish()
    

if __name__ == '__main__':
    # only test: only test, not train
    # resume run: run again with the same run id
    main(only_test=False,resume_run=False)


