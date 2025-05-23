import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.utils import resample

"""
Description：设置一系列条件函数对输入的数据进行限制，返回数据集和星表数据的子集，不另外生成文件，节省内存空间
"""

def filter_magnitude(df):
    """
    将R波段按照21.5进行分割，看看两部分的效果
    """
    print(f"before magnitude filter: {len(df)}")
    # R<=21.5
    df_m1 = df[df['MAG_R']<=21.5]
    print(f"after magnitude filter: {len(df_m1)}")

    return df_m1.index

def filter_lowz(df,threshold=0.5):
    """
    将小于等于threshold部分的红移取出
    """
    print(f"before magnitude filter: {len(df)}")
    df_lowz = df[df['z_spec']<=threshold]
    print(f"after magnitude filter: {len(df_lowz)}")

    return df_lowz.index

def filter_highz(df,threshold=0.5):
    """
    将大于threshold部分的红移取出
    """
    print(f"before magnitude filter: {len(df)}")
    df_highz = df[df['z_spec']>threshold]
    print(f"after magnitude filter: {len(df_highz)}")

    return df_highz.index



def apply_filter(index,data,df):
    """
    对输入数据和星表进行filter
    index: filter index
    data:(images,labels,ebv,...)
    df:catalog
    """
    new_data = []
    for d in data:
        new_data.append(d[index])
    new_df = df.iloc[index]

    return new_data,new_df




def data_filter(data,df):
    """
    接口函数
    """
    # 需要自己设定调整filter
    # filter_index = filter_magnitude(df)
    filter_index = filter_lowz(df)
    new_data,new_df = apply_filter(filter_index,data,df)
    return new_data,new_df,filter_index


