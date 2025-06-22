"""
Created on Wed Jun 11 16:40:06 2025

@author: AXILLIOS
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import pandas as pd
import time
import sys
def cls():
    print(chr(27) + '[2J') 
def pause():
    input('PRESS ENTER TO CONTINUE.')
#------------------------------------------------------------
def tic():
    t1=float(time.time());
    return t1
#------------------------------------------------------------
def toc(t1,s):
    t2=float(time.time());dt=t2-t1;
    s1='time taken '+s 
    print('%s %e' % (s1,dt) )     
#---------------------------------------------------------
def RETURN():
    sys.exit()

# -------------------------------------------------------------------------

def loadData(path, files, index):

    data_list = []
    y_list = []
    fNames = []

    for label, file in enumerate(files):
        df = pd.read_csv(path + file)

        # Extract and select feature names on first file
        if not fNames:
            all_features = df.columns[1:].tolist()  # Exclude time column

        all_features = df.columns[1:].tolist()  # Exclude time column
        fNames = all_features
        
        # Drop first row (label row) and time column, subset features
        df = df.iloc[1:, 1:][fNames]
        df = df.astype(np.float32)

        data_list.append(df)
        y_list.append(np.full(df.shape[0], label))

    X = np.concatenate(data_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    if index == 0:
        Xn = X[:,0:6]
        tag ='ONLY RAW DATA'
        
    elif index == 1:
        Xn = X
        tag ='ALL FEATURES'

    elif index == 2:
        column_indices = [64, 66, 9, 65, 35, 32, 22, 38, 42, 20]
        Xn = X[:, column_indices]
        tag ='RELIFF FEATURES 10 Best'
        '''
        Index 64: feature 0: IQR_gyro y
        Index 66: feature 1: FFT_acceleration x
        Index 9: feature 2: Production Cubic Magnitude of Angular Velocity
        Index 65: feature 3: IQR_gyro z
        Index 35: feature 4: Signal Magnitude Area Gyroscope
        Index 32: feature 5: gyro_z_window_max
        Index 22: feature 6: acceleration_z_window_mean
        Index 38: feature 7: RMS_acceleration z
        Index 42: feature 8: MAD_acceleration x
        Index 20: feature 9: acceleration_y_window_max
        '''
        
    elif index == 3:
        
        tag ='ALL NORMALIZE FEATURES'
        
    elif index == 4:
        
        tag ='RELIFF NORMALIZE FEATURES 10 Best'
        
    return Xn, y, np.array(fNames), tag

    
# -------------------------------------------------------------------------
cls() 

files = ['movement_0_feat.csv',
         'x_axis_with _random_movements_feat.csv',
         'x_1deg_per_min_feat.csv', 
         'x_2deg_per_min_feat.csv',
         'x_anomaly_detection_3dpersec_feat.csv',
         'y_axis_with _random_movements_feat.csv',
         'y_1deg_per_min_feat.csv', 
         'y_2deg_per_min_feat.csv',
         'y_anomaly_detection_3dpersec_feat.csv',
         'z_axis_with _random_movements_feat.csv',
         'z_1deg_per_min_feat.csv', 
         'z_2deg_per_min_feat.csv',
         'z_anomaly_detection_3dpersec_feat.csv',
         ]

output_file_1 = [f.replace('.csv', '_10feats.csv') for f in files]

input_path = './FEATS/'
output_path_1 = './10_BEST_FEATS/'

find = 2

if find == 0:
    
    # Load CSV
    X, y, fNames, Data_tag = loadData(input_path , files, index=1)
    
    import csv

    data1 = X.tolist()
    data2 = y.tolist()
    file_path1 = 'data_X.csv'
    file_path2 = 'data_y.csv'

    # Save data1 (X) — assume it's a 2D array
    with open(file_path1, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data1)

    # Save data2 (y) — assume it's a 1D array
    with open(file_path2, 'w', newline='') as file:
        writer = csv.writer(file)
        for val in data2:
            writer.writerow([val])  # Wrap in list to write one value per row 

elif find == 1:
    
    df = pd.read_csv(input_path + 'y_1deg_per_min_feat.csv')
    
    # Drop the time column (assumed to be the first column)
    df = df.iloc[:, 1:]
    
    # Feature indices
    feature_indices = [64, 66, 9, 65, 35, 32, 22, 38, 42, 20]
    
    # Get the column names at these indices
    matched_features = [df.columns[i] for i in feature_indices]
    
    # Print result
    print("Matched feature names:")
    for i, (idx, name) in enumerate(zip(feature_indices, matched_features)):
        print(f"Index {idx}: feature {i}: {name}")
        
elif find == 2:
    
    column_indices = [64, 66, 9, 65, 35, 32, 22, 38, 42, 20]
    
    for file_idx in range(len(files)):

        df1 = pd.read_csv(input_path + files[file_idx])
        
        # Drop the time column (assumed to be the first column)
        df1 = df1.iloc[:, 1:]
        
        df1 = df1.iloc[:, column_indices]
        df1.to_csv(output_path_1 + output_file_1[file_idx], index=False)