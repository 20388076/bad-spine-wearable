'''
Created on Mon Apr 21 20:22:50 2025

@author: AXILLIOS PITSILAKS
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

import time
import sys

def cls():
    print(chr(27) + '[2J') 
def pause():
    input('PRESS ENTER TO CONTINUE.')
    #    ------------------------------------------------------------
def tic():
    t1 = float(time.time());
    return (t1)
#------------------------------------------------------------
def toc(t1,s):
    t2 = float(time.time()); dt = t2 - t1;
    s1 = 'time taken ' + s 
    print('%s %e' % (s1,dt) )     
#---------------------------------------------------------
def RETURN():
    sys.exit()
    
    
cls() 

#=========================== SET USER =========================================
user=1 # 0:ACHILLIOS, 1:OTHER USERS
#==============================================================================

# File paths in DataFrame format: [1 deg/min, 2 deg/min]
file_path = './CLEAN/'
file_names = pd.DataFrame([
    ['x_1step_per_min_clear.csv', 'x_2step_per_min_clear.csv'],
    ['y_1step_per_min_clear.csv', 'y_2step_per_min_clear.csv'],
    ['z_1step_per_min_clear.csv', 'z_2step_per_min_clear.csv']
])

save_folder = pd.DataFrame([
    ['dataset3 1hr x degrees per min', 'dataset4 1hr x∙n degrees per min'],
]) 

subfolder = [ 'X axis',
              'Y axis',
              'Z axis'
]  

location = 'C:\\Users\\user\\OneDrive\\Έγγραφα\\Final work Experiments\\'

output_names = pd.DataFrame([
    ['x_1step_per_min_clr_pr.csv', 'x_2step_per_min_clr_pr.csv'],
    ['y_1step_per_min_clr_pr.csv', 'y_2step_per_min_clr_pr.csv'],
    ['z_1step_per_min_clr_pr.csv', 'z_2step_per_min_clr_pr.csv']
])

# Corresponding cutoff times for [1 step/min, 2 deg/min datasets]


# Loop over each row (sensor: x, y, z)
for axis_index in range(file_names.shape[0]):
    # Loop over each column (1 deg/min, 2 deg/min)
    for rate_index in range(file_names.shape[1]):
        file_name = file_names.iloc[axis_index, rate_index]
        print(f'Processing file: {file_name}')

        try:
            df = pd.read_csv(file_path + file_name)
        except FileNotFoundError:
            print(f'File {file_name} not found. Skipping.')
            continue
        # Read the full file to extract timestamps
        original_df = df.copy()
        
        # Extract the original timestamps
        time_col = original_df['time (ms)'].values
        
        # Filter and duplicate sensor data (without affecting time)
        start_time = df['time (ms)'].dropna().min()
        cutoff_time = [2400000 + start_time, 1200000 + start_time]
        cut_df = df[df['time (ms)'] <= cutoff_time[rate_index]].reset_index(drop=True)
        
        # Prepare to fill up to the original file length
        sensor_columns = df.columns.drop('time (ms)')
        sensor_data = []
        
        i = 0
        while len(sensor_data) < len(time_col):
            for j in range(len(cut_df)):
                if len(sensor_data) >= len(time_col):
                    break
                sensor_data.append(cut_df.loc[j, sensor_columns].values)
            i += 1
        
        # Convert to DataFrame and truncate to match original time length
        sensor_df = pd.DataFrame(sensor_data[:len(time_col)], columns=sensor_columns)
        
        # Combine with original timestamps
        final_df = pd.DataFrame()
        final_df['time (ms)'] = time_col
        for col in sensor_columns:
            final_df[col] = sensor_df[col]

        if user==0:
            
            # FOR ACHILLIOS Ensure output path exists
            output_path = os.path.join(location, save_folder.iloc[0, rate_index], subfolder[axis_index])
            os.makedirs(output_path, exist_ok=True)
            
            # Save the processed DataFrame
            output_file = os.path.join(output_path, output_names.iloc[axis_index, rate_index])
            final_df.to_csv(output_file, index=False)
            
        elif user==1:
            # FOR ALL USERS Save the processed DataFrame
            output_path='./CLEAN/'
            final_df.to_csv(output_path+output_names.iloc[axis_index, rate_index], index=False)
            
        # Plotting all data
        plt.style.use('ggplot')
        fig, axs = plt.subplots(len(final_df.columns) - 1, 1, figsize=(12, 18), sharex=False)
        
        for i, column in enumerate(final_df.columns[1:]):  # skip 'time (ms)'
            axs[i].plot(final_df['time (ms)'], final_df[column], label=column)
            axs[i].set_ylabel(column)
            axs[i].legend(loc='upper right')
            axs[i].grid(True)
            #axs[i].tick_params(axis='x', labelrotation=45) # rotate if overlap
            axs[i].tick_params(labelbottom=True)
            axs[i].ticklabel_format(style='plain', axis='x')
        axs[-1].set_xlabel('Time (ms)')
        fig.suptitle(f'Sensor Data over Time from {output_names.iloc[axis_index, rate_index]}', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
               
        # Save the plot as a PNG file
        filename = output_names.iloc[axis_index, rate_index] + '_plot.png'  
        full_path = os.path.join(output_path, filename)
        
        # Save the figure
        plt.savefig(full_path, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
           
            