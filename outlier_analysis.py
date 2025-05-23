import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from metrics import compute_metrics



def galaxy_type_fun(row):
    if not isinstance(row['main_type'],str):
        return row['SUBCLASS']
    elif not isinstance(row['SUBCLASS'],str):
        return row['main_type']
    else:
        return f"{row['main_type']}_{row['SUBCLASS']}"



def feature_visual(normal_df,outlier_df,df,model_name):
   # features to check：
    patterns = ['z_spec', 'MAG_G', 'MAG_R', 'MAG_I', 'MAG_Z', 'MAG_W1',
        'MAG_W2', 'ra_2', 'dec_2', 'ebv','galaxy_type']
    means_diff = []

   
    fig = plt.figure(figsize=(10, 15))
    plt.subplots_adjust(wspace=0.3, hspace=0.5) 
    gs = gridspec.GridSpec(int(np.ceil(len(patterns)/2)), 2)

    # 11 subplots
    for i in range(11):
        p = patterns[i]
        # adjust the last subplot location
        if i == 10:
            ax = fig.add_subplot(gs[-1, :])
            normal_galaxy_type_counts = normal_df[p].value_counts(normalize=True).sort_index()
            outlier_galaxy_type_counts = outlier_df[p].value_counts(normalize=True).sort_index()

            normal_galaxy_type_name = list(normal_galaxy_type_counts.index)
            normal_galaxy_type_count = list(normal_galaxy_type_counts.values)
            outlier_galaxy_type_name = list(outlier_galaxy_type_counts.index)
            outlier_galaxy_type_count = list(outlier_galaxy_type_counts.values)

            ax.plot(normal_galaxy_type_name,normal_galaxy_type_count,label='normal')
            ax.scatter(normal_galaxy_type_name,normal_galaxy_type_count)
            ax.plot(outlier_galaxy_type_name,outlier_galaxy_type_count,label='outlier',color='orange')
            ax.scatter(outlier_galaxy_type_name,outlier_galaxy_type_count,color='orange')

            ax.set_title(p)
            ax.legend()
            
        else:
            ax = fig.add_subplot(gs[i // 2, i % 2]) 
            
            counts_normal, bin_edges_normal = np.histogram(normal_df[p], bins=10,range=(df[df[p].apply(np.isfinite)][p].min(),df[df[p].apply(np.isfinite)][p].max()))
            probabilities_normal = counts_normal / len(normal_df[p])
            counts_outlier, bin_edges_outlier = np.histogram(outlier_df[p], bins=10,range=(df[df[p].apply(np.isfinite)][p].min(),df[df[p].apply(np.isfinite)][p].max()))
            probabilities_outlier = counts_outlier / len(outlier_df[p])
            ax.plot((bin_edges_normal[:-1]+bin_edges_normal[1:])/2, probabilities_normal,label='normal')
            ax.scatter((bin_edges_normal[:-1]+bin_edges_normal[1:])/2, probabilities_normal)
            ax.axvline(x = np.mean(normal_df[p]),linestyle='--')
            ax.plot((bin_edges_outlier[:-1]+bin_edges_outlier[1:])/2, probabilities_outlier,label='outlier',color='orange')
            ax.scatter((bin_edges_outlier[:-1]+bin_edges_outlier[1:])/2, probabilities_outlier,color='orange')
            ax.axvline(x = np.mean(outlier_df[p]),color='orange',linestyle='--')
            ax.legend()
            ax.set_title(p)
            # means_diff.append(np.abs(np.mean(normal_df[p])-np.mean(outlier_df[p])))
    
    # means_dict = dict(zip(patterns,means_diff))
    plt.suptitle("Outliers Analysis",y=0.93)
    # save：
    plt.savefig(f'./output/{model_name}/outlier_analysis_{model_name}.png')

        

def outlier_analysis(data_path,catalog_path,model_name):
    # test data
    data = np.load(data_path)
    z_pred = data['z_pred']
    z_true = data['z_true']
    test_indices = data['indices']
    df = pd.read_csv(catalog_path)

    # df gaxly type: combine main_type和subclass
    df['galaxy_type'] = df.apply(lambda x:galaxy_type_fun(x),axis=1)


    # outlier
    deltaz,bias,nmad,outlier = compute_metrics(z_pred,z_true)
    outlier_indices_temp = np.where(np.abs(deltaz)>0.15)[0]
    outlier_indices = test_indices[outlier_indices_temp]
    # inlier
    normal_indices_temp = np.where(np.abs(deltaz)<=0.15)[0]
    normal_indices = test_indices[normal_indices_temp]
    # df：
    normal_df = df.iloc[normal_indices]
    outlier_df = df.iloc[outlier_indices]

    # feature check：
    feature_visual(normal_df,outlier_df,df,model_name)
