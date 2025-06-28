'''
Created on Thu Apr  3 20:08:53 2025

@author: Achillios Pitsilkas
'''
import os
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import sys
from scipy.fft import fft
from scipy.stats import iqr

# ---------------- Utility Functions ----------------

def cls():
    print(chr(27) + '[2J') 

def pause():
    input('PRESS ENTER TO CONTINUE.')

def tic():
    return float(time.time())

def toc(t1, s):
    t2 = float(time.time())
    print(f'{s} time taken: {t2 - t1:.6e} seconds')

def RETURN():
    sys.exit()

cls()

def compute_fft_features(signal):
    fft_vals = np.abs(fft(signal))
    fft_vals = fft_vals[:len(fft_vals) // 2]
    return {
        'fft_mean': np.mean(fft_vals),
        'fft_max': np.max(fft_vals),
        'fft_dom_freq_index': np.argmax(fft_vals),
        'fft_energy': np.sum(fft_vals**2).mean
    }

# ---------------- File Configuration ----------------

train_labels = [
    'time (ms)', 'acceleration x', 'acceleration y', 'acceleration z',
    'gyro x', 'gyro y', 'gyro z'
]

input_path = './RAW/'
output_path_1 = './CLEAN/'
output_path_2 = './FEATS/'

os.makedirs(output_path_1, exist_ok=True)
os.makedirs(output_path_2, exist_ok=True)

input_file_1 = [
    'movement_0.csv', 'x_axis_with_random_movements.csv',
    'y_axis_with_random_movements.csv', 'z_axis_with_random_movements.csv',
    'x_1step_per_min.csv', 'y_1step_per_min.csv', 'z_1step_per_min.csv',
    'x_2step_per_min.csv', 'y_2step_per_min.csv', 'z_2step_per_min.csv',
    'x_anomaly_detection_3_step_per_min.csv', 'y_anomaly_detection_3_step_per_min.csv',
    'z_anomaly_detection_3_step_per_min.csv'
]

output_file_1 = [f.replace('.csv', '_clear.csv') for f in input_file_1]

output_file_2 = [f.replace('.csv', '_feat.csv') for f in input_file_1]

input_file_2 =  [
    'movement_0_clear.csv', 'x_axis_with_random_movements_clear.csv',
    'y_axis_with_random_movements_clear.csv', 'z_axis_with_random_movements_clear.csv',
    'x_1step_per_min_clr_pr.csv', 'y_1step_per_min_clr_pr.csv', 'z_1step_per_min_clr_pr.csv',
    'x_2step_per_min_clr_pr.csv', 'y_2step_per_min_clr_pr.csv', 'z_2step_per_min_clr_pr.csv',
    'x_anomaly_detection_3_step_per_min_clear.csv', 'y_anomaly_detection_3_step_per_min_clear.csv',
    'z_anomaly_detection_3_step_per_min_clear.csv'
]

location = 'C:\\Users\\user\\OneDrive\\Έγγραφα\\Final work Experiments\\'
copy_path = [
    'dataset1 1hr do nothing',
    'dataset2 1hr  do nothing (random small movements)\\X axis',
    'dataset2 1hr  do nothing (random small movements)\\Y axis',
    'dataset2 1hr  do nothing (random small movements)\\Z axis',
    'dataset3 1hr x degrees per min\\X axis',
    'dataset3 1hr x degrees per min\\Y axis',
    'dataset3 1hr x degrees per min\\Z axis',
    'dataset4 1hr x∙n degrees per min\\X axis',
    'dataset4 1hr x∙n degrees per min\\Y axis',
    'dataset4 1hr x∙n degrees per min\\Z axis',
    'dataset5 1hr anomaly detection\\X axis',
    'dataset5 1hr anomaly detection\\Y axis',
    'dataset5 1hr anomaly detection\\Z axis'
]

# ---------------- Data Process Option ----------------

data_process = 1  # 0: raw -> clear; 1: clear -> features

if data_process == 0:
    for file_idx in range(len(input_file_1)):
        in_file = input_path + input_file_1[file_idx]
        out_file = output_path_1 + output_file_1[file_idx]
        
        # Remove first 19 lines
        with open(in_file, 'r', newline='', errors='replace') as infile:
            lines = list(csv.reader(infile))
        with open(out_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            for i, row in enumerate(lines, start=1):
                if i > 17:
                    writer.writerow(row)

        # Read CSV with headers manually added
        df1 = pd.read_csv(out_file, names=train_labels)
        start_time = df1['time (ms)'].dropna().min()
        df1 = df1.sort_values('time (ms)').reset_index(drop=True)
        df1 = df1[df1['time (ms)'] <= 3600100 + start_time]

        # Save clean dataset to disk
        df1.to_csv(out_file, index=False)

        final_csv_path = os.path.join(location, copy_path[file_idx], output_file_1[file_idx])
        os.makedirs(os.path.dirname(final_csv_path), exist_ok=True)
        df1.to_csv(final_csv_path, index=False)

        # Plot sensor data
        plt.style.use('ggplot')
        fig, axs = plt.subplots(len(df1.columns) - 1, 1, figsize=(12, 18), sharex=False)

        for i, column in enumerate(df1.columns[1:]):
            axs[i].plot(df1['time (ms)'], df1[column], label=column)
            axs[i].set_ylabel(column)
            axs[i].legend(loc='upper right')
            axs[i].grid(True)
            axs[i].tick_params(labelbottom=True)
            axs[i].ticklabel_format(style='plain', axis='x')
        axs[-1].set_xlabel('Time (ms)')
        fig.suptitle(f'Sensor Data over Time from {output_file_1[file_idx]}', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        plot_path_1 = os.path.join(output_path_1, output_file_1[file_idx] + '_plot.png')
        plot_path_2 = os.path.join(location, copy_path[file_idx], output_file_1[file_idx] + '_plot.png')
        plt.savefig(plot_path_1, dpi=600, bbox_inches='tight')
        plt.savefig(plot_path_2, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f'File saved as {output_file_1[file_idx]}')
        print('=' * 70)

elif data_process == 1:
    
    for file_idx in range(len(input_file_2)):
        
        df1 = pd.read_csv(output_path_1 + input_file_2[file_idx])
        #--------------------------------------------------------------------
        
        # Compute vector magnitudes
        df1['Acceleration Sum Vector Magnitude'] = np.sqrt(
            df1['acceleration x']**2 + df1['acceleration y']**2 + df1['acceleration z']**2).round(3)
        df1['Sum Vector Magnitude of Angular Velocity'] = np.sqrt(
            df1['gyro x']**2 + df1['gyro y']**2 + df1['gyro z']**2).round(3)
        
        #--------------------------------------------------------------------
        
        # Cubic product features
        df1['Acceleration Cubic Product Magnitude'] = (
            abs(df1['acceleration x'] * df1['acceleration y'] * df1['acceleration z']) ** (1/3)).round(3)
        df1['Production Cubic Magnitude of Angular Velocity'] = (
            abs(df1['gyro x'] * df1['gyro y'] * df1['gyro z']) ** (1/3)).round(3)
        
        #--------------------------------------------------------------------
        
        # Gradient features dt
        gradient_targets = {
            'acceleration x': 'da_x/dT', 'acceleration y': 'da_y/dT', 'acceleration z': 'da_z/dT',
            'gyro x': 'dg_x/dT', 'gyro y': 'dg_y/dT', 'gyro z': 'dg_z/dT'
        }
        for col, new_col in gradient_targets.items():
            df1[new_col] = np.gradient(df1[col], df1['time (ms)']).round(3)
            
        #--------------------------------------------------------------------
        
        # Window-based features //min, max, average//
        window = 32
        df1['window_id'] = (df1.index // window) # for every 32 measurements due to ESP32_FFT library limmitations (must be multiple of 2)
        
        sensor_cols = ['acceleration x', 'acceleration y', 'acceleration z',
                       'gyro x', 'gyro y', 'gyro z']
        agg_funcs = ['mean', 'max', 'min']
        for col in sensor_cols:
            for func in agg_funcs:
                new_col = f'{col.replace(' ', '_')}_window_{func}'
                df1[new_col] = df1.groupby('window_id')[col].transform(func).round(3)
                
        #--------------------------------------------------------------------        
        
        # Signal magnitude area SMA  
        df1['window_id'] = (df1.index // window) 
            # Define sensor groups and feature names
        sensor_cols2 = [
            ['acceleration x', 'acceleration y', 'acceleration z'],
            ['gyro x', 'gyro y', 'gyro z']
        ]    
        feats = ['Signal Magnitude Area Accelerometer', 'Signal Magnitude Area Gyroscope']
        for i, cols in enumerate(sensor_cols2):
            # Calculate row-wise magnitude sum: |x| + |y| + |z|
            df1[f'_abs_sum_{i}'] = df1[cols].abs().sum(axis=1)
            # Group by window and take mean of that sum to compute SMA
            sma_series = df1.groupby('window_id')[f'_abs_sum_{i}'].mean()
            # Map SMA result back to the original dataframe
            df1[feats[i]] = df1['window_id'].map(sma_series).round(3)
            # Cleanup temporary columns
        df1.drop(columns=['window_id', '_abs_sum_0', '_abs_sum_1'], inplace=True)
      
        #--------------------------------------------------------------------    
        # Root Mean Square RMS           
        def compute_feat_per_window(df, columns_to_process, feats, choice):
            df = df.copy()
            df['window_id'] = (df1.index // window) 
            for col in columns_to_process:
                feat_col_name = f'{feats[choice]}_{col}'
                df[feat_col_name] = np.nan
            grouped = df.groupby('window_id')       
            for window_id, group in grouped:
                for col in columns_to_process:
                    if choice == 0: # RMS
                        val = np.sqrt(sum(group[col] ** 2)/ window) 
                    elif choice == 1: # MAD
                        val = (sum(np.abs(group[col] - group[col].mean()))) / window
                    elif choice == 2: # VAR
                        val = np.var(group[col])
                    elif choice == 3: # STD
                        val = np.std(group[col])
                    elif choice == 4: # IQR
                        val = iqr(group[col])
                    elif choice == 5: # FFT
                        val = np.real(fft(group[col]))
                    elif choice == 6: # ENERGY
                        val = (np.sum(abs(fft(group[col])**2))) / window
                    df.loc[df['window_id'] == window_id, f'{feats[choice]}_{col}'] = val
            df[[f'{feats[choice]}_{col}' for col in columns_to_process]] = df[[f'{feats[choice]}_{col}' 
                                                                               for col in columns_to_process]].round(3)
            df.drop(columns=['window_id'], inplace=True)
            return df
        
        sensor_cols = ['acceleration x', 'acceleration y', 'acceleration z', 
                       'gyro x', 'gyro y', 'gyro z']
        feats = ['RMS','MAD','VAR','STD','IQR','FFT','ENERGY']
        
        #--------------------------------------------------------------------
        # Root Mean Square (RMS), Mean Absolute Deviation (MAD), Variance (VAR), Standard Deviation (STD), 
        # Interquartile Range (IQR), Fast Fourier Transform (FFT), Energy  
        for choice in range(len(feats)):
            df1 = compute_feat_per_window(df1, sensor_cols, feats, choice)

        # Saving feat to CSV
        df1.to_csv(output_path_2 + output_file_2[file_idx], index=False)
        
        final_csv_path = os.path.join(location, copy_path[file_idx], output_file_2[file_idx])
        os.makedirs(os.path.dirname(final_csv_path), exist_ok=True)
        df1.to_csv(final_csv_path, index=False)
        
        print(f'File saved as {output_file_2[file_idx]}\n')

        
if data_process == 2: # data labeling

    for file_idx in range(len(output_file_2)):
        df1 = pd.read_csv(output_path_2 + output_file_2[file_idx])
        
