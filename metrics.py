import numpy as np
import random
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns

def compute_metrics(pred_red,labels):
    def outlier_compute(deltaz,nmad):
        # return len(np.where(np.abs(deltaz)>5*nmad)[0])/len(deltaz)
        return len(np.where(np.abs(deltaz)>0.15)[0])/len(deltaz)
    # delta z
    deltaz = (pred_red - labels)/(1+labels)
    # bias
    # bias = np.mean(deltaz)
    bias = np.median(deltaz)
    # NMAD
    nmad = 1.48*(np.median(abs(deltaz-np.median(deltaz))))
    # outlier:
    outlier = outlier_compute(deltaz,nmad)
    

    return deltaz,bias,nmad,outlier


''' Function that makes plots '''
def make_plot(pred_red,labels,deltaz,bias,nmad,outlier,model_name = 'model'):
    """
        Makes a histogram of the prediction bias and a plot of the estimated Photometric redshift 
        and the Prediction bias versus the Spectroscopic redshift given the labels, red, dz, smad,
        name and lim and it saves both of them in .png files. (Mostly used for debugging purposes)


        Arguments:
            labels (ndarray): The labels for the images used.

            red (ndarray): Contains the predicted redshift values for the test images.

            dz (ndarray): Residuals for every test image.

            smad (float):The MAD deviation.

            name (str): The name of the model.

            lim (float): The limit of the axes of the plot

    """
    # Constructing the filenames
    
    file_name_01 = f'./output/{model_name}/Residuals_{model_name}.png'
    file_name_02 = f'./output/{model_name}/Plot_{model_name}.png'
    # Plotting the residuals 
    plt.figure()
    plt.hist(deltaz, bins=100)
    plt.xticks(np.arange(-0.1, 0.1, step=0.02))
    plt.xlim(-0.1,0.1)
    plt.xlabel('Δz')
    plt.ylabel('Relative Frequency')
    plt.title('Δz Distribution')
    plt.savefig(file_name_01, bbox_inches='tight')

    # Plotting the predictions vs the data 
    fig = plt.figure(figsize = (3,3))
    axis = fig.add_axes([0,0.4,1,1])
    axis2 = fig.add_axes([0,0,1,0.3])
    axis.set_ylabel('Photometric Redshift')
    axis2.set_ylabel('bias Δz/(1+z)')
    axis2.set_xlabel('Spectroscopic Redshift')
    lim = np.max(labels)
    axis.plot([0, lim], [0, lim], 'k-', lw=1)
    axis.set_xlim([0, lim])
    axis.set_ylim([0,lim])

    # outlier indicator:
    x_z = np.arange(0,lim,0.1)
    axis.plot(x_z,0.15*(1+x_z)+x_z,'steelblue',linestyle = '--',lw=1)
    axis.plot(x_z,-0.15*(1+x_z)+x_z,'steelblue',linestyle = '--',lw=1)
    axis.text(0.1, 0.95, f'Bias: {bias:.4f}\nNMAD: {nmad:.4f}\nOutlier fraction:{outlier:.4f}',transform=axis.transAxes,verticalalignment='top', horizontalalignment='left')


    axis2.plot([0, lim], [0, 0], 'k-', lw=1)
    axis2.set_xlim([0, lim])
    # axis.plot(labels,red,'ko', markersize=0.3, alpha=0.3)
    axis.scatter(labels,pred_red,marker='o',color='k',s=0.3,alpha=0.3)
    #axis.hist2d(labels,red,bins =150)
    # axis2.plot(labels,  np.asarray(red) - np.asarray(labels),'ko', markersize=0.3, alpha=0.3)
    # axis2.scatter(labels,  np.asarray(pred_red) - np.asarray(labels),color='k',marker='o',s=0.3,alpha=0.3)
    axis2.scatter(labels,  deltaz,color='k',marker='o',s=0.3,alpha=0.3)
    axis2.axhline(0.15,color='steelblue',linestyle='--',lw=1)
    axis2.axhline(-0.15,color='steelblue',linestyle='--',lw=1)
    
    
    plt.savefig(file_name_02,dpi = 300,transparent = False,bbox_inches = 'tight')


def plot_probability(pred_red,label,predictions,zbins,predict=False):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    pred_red_label = np.digitize(pred_red,zbins,right=True)
    if not predict:
        true_red_label = np.digitize(label,zbins,right=True)
    
    plt.plot(range(len(zbins)),predictions)
    plt.axvline(pred_red_label,linestyle='--',color='red')
    if not predict:
        plt.axvline(true_red_label,linestyle='-.',color='green')
    

    
def plot_img(image,pred_red,label,predict=False):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    # image: (channel,width,height)
    # image: g,r,i,z
    g = image[0,:,:]
    r = image[1,:,:]
    i = image[2,:,:]
    rgb = make_lupton_rgb(i,r,g,Q=10,stretch=0.05)
    plt.imshow(rgb,origin='lower')
    # plt.imshow(image[:,:,:3])# select the first 3 bands
    if predict:
        plt.title(f"pred(r):{round(pred_red,3)}")
    else:
        plt.title(f"pred(r):{round(pred_red,3)} true(g):{round(float(label),3)}")


def check_probability(images,pred_red,labels,predictions,zbins,model_name,predict=False):
    """
    randomly check 9 PDFs
    """
    
    num_rows,num_cols=3,3
    num_img = num_rows * num_cols
    idx_list = [random.randint(0,len(pred_red)-1) for i in range(num_img)]
    plt.figure(figsize=(2*2*num_cols,2*num_rows))
    for i in range(num_img):
        idx = idx_list[i]
        plt.subplot(num_rows,2*num_cols,2*i+1)
        if predict:
            plot_img(images[idx],pred_red[idx],None,predict)
            plt.subplot(num_rows,2*num_cols,2*i+2)
            plot_probability(pred_red[idx],None,predictions[idx],zbins,predict)
        else:
            plot_img(images[idx],pred_red[idx],labels[idx],predict)
            plt.subplot(num_rows,2*num_cols,2*i+2)
            plot_probability(pred_red[idx],labels[idx],predictions[idx],zbins,predict)

    
    # plt.show()
    plt.savefig(f'./output/{model_name}/check_pdf_{model_name}.png',dpi=200)




def metrics_z_plot(z_pred,z_true,model_name,interval=0.5):
    """
    Function of metrics and redshift
    """
    # redshift interval：
    z_min = 0.
    z_max = np.max(z_true)
    z_max = (z_max+interval)//interval*interval

    z_intervals = np.arange(z_min,z_max+interval,interval)

    # metrics computation：
    deltaz_list,bias_list,nmad_list,outlier_list = [],[],[],[]
    for i in range(1,len(z_intervals)):
       
        idx = np.where((z_intervals[i-1]<=z_true) & (z_true<z_intervals[i]))[0]
        
        true_data = z_true[idx]
        pred_data = z_pred[idx]

        if len(true_data) == 0:
            deltaz,bias,nmad,outlier = np.nan, np.nan, np.nan, np.nan
        else:
            deltaz,bias,nmad,outlier = compute_metrics(pred_data,true_data)
        deltaz_list.append(deltaz)
        bias_list.append(bias)
        nmad_list.append(nmad)
        outlier_list.append(outlier)
    
    
    x = [(z_intervals[i]+z_intervals[i+1])/2 for i in range(len(z_intervals)-1)]
    metrics_name = ['Bias','NMAD','Outlier_fraction']
    metrics_data = [bias_list,nmad_list,outlier_list]
    fig, axs = plt.subplots(3, 1,figsize=(6,12))
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    # plt.tight_layout()
    for i in range(len(metrics_name)):
        ax1 = axs[i]
        ax1.plot(x, metrics_data[i],label=metrics_name[i],color='steelblue')
        ax1.scatter(x, metrics_data[i],label=metrics_name[i])
        for j, value in enumerate(metrics_data[i]):
            ax1.annotate(f'{value:.2f}', (x[j], metrics_data[i][j]), textcoords="offset points", xytext=(0,5), ha='center')
        
        ax1.axhline(0.,linestyle='--',color='k')
        ax1.set_xlabel('z')
        ax1.set_ylabel(metrics_name[i], color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')

       
        ax2 = ax1.twinx()

        ax2.hist(z_true,bins=int(z_max/interval),range=[0.,z_max],alpha=0.5,color='orange')
        ax2.set_ylabel('Count', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        axs[i].set_title(f"{metrics_name[i]} - z")

    plt.savefig(f'./output/{model_name}/{model_name}_metrics-z.png')




