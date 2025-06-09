import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.utils import resample

"""
Description: A set of conditional functions is applied to filter the input data, returning subsets of both the dataset and catalog without creating additional files, thereby saving memory.
"""

def filter_magnitude(df):
    """
    Split the R band at 21.5
    """
    print(f"before magnitude filter: {len(df)}")
    # R<=21.5
    df_m1 = df[df['MAG_R']<=21.5]
    print(f"after magnitude filter: {len(df_m1)}")

    return df_m1.index

def filter_lowz(df,threshold=0.5):
    """
    Split the redshift at threshold
    """
    print(f"before magnitude filter: {len(df)}")
    df_lowz = df[df['z_spec']<=threshold]
    print(f"after magnitude filter: {len(df_lowz)}")

    return df_lowz.index

def filter_highz(df,threshold=0.5):
    """
    Split the redshift at threshold
    """
    print(f"before magnitude filter: {len(df)}")
    df_highz = df[df['z_spec']>threshold]
    print(f"after magnitude filter: {len(df_highz)}")

    return df_highz.index



def apply_filter(index,data,df):
    """
    Apply filtering to the input data and the catalog.
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
    used in other file
    setting filter function by yourself
    """
    # filter_index = filter_magnitude(df)
    filter_index = filter_lowz(df)
    new_data,new_df = apply_filter(filter_index,data,df)
    return new_data,new_df,filter_index


