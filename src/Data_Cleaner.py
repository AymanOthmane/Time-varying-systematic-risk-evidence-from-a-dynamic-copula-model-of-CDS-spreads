import os
import pandas as pd
import numpy as np
import matplotlib as plt
from pykalman import KalmanFilter

def DataCleaner(cwd, path):
    df = pd.read_excel(cwd + path, index_col=0)
    df.index = pd.to_datetime(df.index,unit='D')
    df = df.dropna(axis=1, thresh=1350).dropna(axis=0, thresh=45)
    df = df.interpolate(method='linear', axis=1)
    df.to_excel(cwd + f'/data/CLEAN_CDX NA IG {df.index.name}.xlsx')
    return df

    
def kalman_fillna(df):
    kalman = KalmanFilter()
    filled_df = df
    for column in df.columns:
        # Step 1: Temporarily fill NaNs to avoid issues with the Kalman filter
        filled_column = df[column].fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Step 2: Apply the Kalman filter
        state_means, _ = kalman.em(filled_column.values, n_iter=5).smooth(filled_column.values)

        # Step 3: Replace the original NaNs with the Kalman-filtered values
        filled_df[column] = df[column].where(df[column].notna(), state_means)
    return filled_df