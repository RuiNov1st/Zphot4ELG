import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 看一下星等分布情况
def check_dataset_magnitude_dist(df_all,train_idx,valid_idx,test_idx,model_name):
    bands = ['G','R','I','Z','W1','W2']
    ## 星等分布绘图：
    def show_magnitude_dist(df,model_name,name):
        fig,ax = plt.subplots(6,1,figsize=(7,14))
        plt.tight_layout()
        for b in range(len(bands)):
            tmp_df = df[f'MAG_{bands[b]}']
            idx = np.where(df[f'MAG_{bands[b]}_flag']==1)[0]
            ax[b].hist(df[f'MAG_{bands[b]}'].iloc[idx])
            ax[b].set_title(f'MAG_{bands[b]}')
        plt.savefig(f'./output/{model_name}/MAG_dist_{name}.png')
    # 总体：
    show_magnitude_dist(df_all,model_name,'all')
    # 训练集：
    df_train = df_all.iloc[train_idx]
    show_magnitude_dist(df_train,model_name,'train')
    # 验证集
    df_valid = df_all.iloc[valid_idx]
    show_magnitude_dist(df_valid,model_name,'valid')
    # 测试集
    df_test= df_all.iloc[test_idx]
    show_magnitude_dist(df_test,model_name,'test')