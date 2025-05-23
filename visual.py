import numpy  as np
import matplotlib.pyplot as plt


def training_monitor(training_loss,val_loss,run_name):
    fig = plt.figure()
    plt.plot(np.arange(len(training_loss)),training_loss,label='train_loss')
    plt.plot(np.arange(len(val_loss)),val_loss,label='val_loss')
    plt.title("Model Accuracy & Loss")
    plt.xlabel("Epoch")
    
    plt.legend()

    plt.savefig(f'./output/{run_name}/{run_name}_loss_acc.png')

