'''
Created on Thu Apr  3 20:08:53 2025

@author: Achillios Pitsilkas
'''
# -------------------------------  Working Directory -----------------------------
# Set the working directory to the script's location if running in Visual Studio Code
import os
# Change working directory for this script
os.chdir(r'C:\Users\user\OneDrive\Έγγραφα\PlatformIO\Projects\bad-spine-wearable-1\src\data_process_and_classification') # modify this path to your working directory

# ============================= Utility Functions =============================

# ----------------------------- Import Libraries ------------------------------
import time
import sys
# ----------------------------- Kernel clean ----------------------------------
def cls():
    print(chr(27) + '[2J') 
# ----------------------------- Kernel pause ----------------------------------
def pause():
    input('PRESS ENTER TO CONTINUE.')
# ----------------------------- Process time count ----------------------------
def tic():
    return float(time.time())
# ----------------------------- Process time return ---------------------------
def toc(t1, s):
    t2 = float(time.time())
    print(f'{s} time taken: {t2 - t1:.6e} seconds')
# ----------------------------- Kernel break ----------------------------------
def RETURN():
    sys.exit()
# =============================================================================

# ----------------------------- Kernel clean call -----------------------------
cls()

# ============================= Data Process Option Guide =====================
r'''
This program processes raw sensor data (accelerometer and gyroscope) from CSV files, cleans it, extracts features, and saves the results.
It supports 4 main stages of data processing:    
0. Data Cleaning: Removes unnecessary lines from raw CSV files and saves cleaned data and plots them.
    The results are saved in new CSV files with '_clean' suffix
1. Data Processing only for 1 and 2 steps per min: Reads cleaned data, filters it based on time, and duplicates sensor data to fill up to the original file length.
    It saves the processed data in new CSV files with '_cln_pr' suffix.
2. Feature Extraction: Computes various features from the cleaned data: vector magnitudes,
    cubic products, gradients, window-based statistics and features: min, mean, max, signal magnitude area, root mean square,
    mean absolute deviation, variance, standard deviation, interquartile range, fast Fourier transform, and energy.
    The features which computed over defined windows of data with 'window' variable.
    The results are saved in new CSV files with '_feat_prepr' suffix.
3. FFT Feature Processing: Reads the preprocessed feature files, computes the standard deviation of FFT features across all datasets,
    and replaces the FFT values in each dataset with the value from the line that has the highest standard deviation.
    The results are saved in new CSV files with '_feat' suffix.
4. Build Machine Learning - ML dataset (X_data, y_data): Combines all feature files into a single dataset, assigns class labels based on file names,
    and saves the combined features and labels into 'X_data.csv' and 'y_data.csv', splits combine data to train - test datasets for ML, with the same 
    pattern to be followed for raw and g converted data.
5. Feature Selection using ReliefF algorithm: Applies the ReliefF feature selection algorithm to the training dataset,
    selects the top features based on their importance scores, and saves the reduced feature set and feature weights to CSV files.
    The results are saved in 'X_train_reduced_idx_py.csv' and 'Python_relieff_feature_indices_weights.csv'.
6. Plotting the weight order best features and combine the ESP32 computation time.
    The results are saved in 'Relieff_Feature_Weights.png' and 'ESP32_computation_time.png'.
    Also, returns the indices of the 10 best selected weight based features for Edge AI implementation.

To run the program, set the 'data_process' variable set the variable accordingly to the desired operation:
- 0 for Data Cleaning
- 1 for Data Processing
- 2 for Feature Extraction
- 3 for FFT Feature Processing
- 4 for Build ML dataset (X_data, y_data) 
- 5 for Feature Selection using ReliefF algorithm
- 6 for Plotting the weight order best features and combine the ESP32 computation time.
To run all stages sequentially, set 'auto' to 1. To run only one stage, set 'auto' to 0.
The 'window' variable defines the size of the window for all data processes. 
'''
# ------------------------------ Data Process Option --------------------------
# 0: raw -> clean; 
# 1: clean(1 step/min & 2 steps/min) -> processed(1 step/min & 2 steps/min); 
# 2: clean -> features preprocessed; 
# 3: features preprocessed -> features final; 4: features final -> X_data, y_data; 5: X_data, y_data -> ReliefF selected features; 
# 6: Plotting the weight order best features and combine the ESP32 computation time.
data_process = 6
# ------------------------------ Auto Runner Option ---------------------------
# 0: run only one stage; 1: run all stages
auto = 0
# ----------------------------- Window Size -----------------------------------
# Define window size for data trimming to fit window size and window-based features
window = 16 
# ----------------------------- Plotting Option -------------------------------
pl = 1  # 0: no plots; 1: plots

# ----------------------------- Matlab Option for ReleifF -------------------------------
matlab = 0 # 0: python ReleifF ; 1: Matlab ReleifF 

# ---------------- File Configuration ----------------

def TrainTestSplit(X, y, train_size, test_size):
    from sklearn.model_selection import train_test_split
    X_train_parts = []
    X_test_parts = []
    y_train_parts = []
    y_test_parts = []

    for class_label in np.unique(y):
        class_data = X[y == class_label]
        X_train_class, X_test_class = train_test_split(
            class_data, train_size=train_size, test_size=test_size, shuffle=False
        )
        X_train_parts.append(X_train_class)
        X_test_parts.append(X_test_class)
        y_train_parts.append(np.full(X_train_class.shape[0], class_label))
        y_test_parts.append(np.full(X_test_class.shape[0], class_label))

    X_train = np.concatenate(X_train_parts, axis=0)
    X_test = np.concatenate(X_test_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)

    return X_train, X_test, y_train, y_test


# ---------------- Import Libraries ------------------
import os
import pandas as pd
import numpy as np
# ----------------------------------------------------
# Define the location for saving processed files
# --- files for data_process = 0 ---

input_path_0 = './0_RAW/'
output_path_0 = './1_CLEAN/'

# Create output directory if it doesn't exist
os.makedirs(output_path_0, exist_ok=True) 

input_file_0 = [
    'movement_0.csv', 'x_axis_with_random_movements.csv',
    'y_axis_with_random_movements.csv', 'z_axis_with_random_movements.csv',
    'x_1step_per_min.csv', 'y_1step_per_min.csv', 'z_1step_per_min.csv',
    'x_2step_per_min.csv', 'y_2step_per_min.csv', 'z_2step_per_min.csv',
    'x_anomaly_detection_3_step_per_min.csv', 'y_anomaly_detection_3_step_per_min.csv',
    'z_anomaly_detection_3_step_per_min.csv'
]
output_file_0 = [f.replace('.csv', '_clean.csv') for f in input_file_0]

# --- files for data_process = 1 ---

input_path_1 = output_path_0
output_path_1 = output_path_0

input_file_1  = pd.DataFrame([
    ['x_1step_per_min_clean.csv', 'x_2step_per_min_clean.csv'],
    ['y_1step_per_min_clean.csv', 'y_2step_per_min_clean.csv'],
    ['z_1step_per_min_clean.csv', 'z_2step_per_min_clean.csv']
])

output_names = pd.DataFrame([
    ['x_1step_per_min_cln_pr.csv', 'x_2step_per_min_cln_pr.csv'],
    ['y_1step_per_min_cln_pr.csv', 'y_2step_per_min_cln_pr.csv'],
    ['z_1step_per_min_cln_pr.csv', 'z_2step_per_min_cln_pr.csv']
])

# --- files for data_process = 2 ---

input_path_2 = output_path_0
output_path_2 = './2_FEATS_PREPROCESSSED/'

# Create output directory if it doesn't exist
os.makedirs(output_path_2, exist_ok=True) 

input_file_2 =  [
    'movement_0_clean.csv', 'x_axis_with_random_movements_clean.csv',
    'y_axis_with_random_movements_clean.csv', 'z_axis_with_random_movements_clean.csv',
    'x_1step_per_min_cln_pr.csv', 'y_1step_per_min_cln_pr.csv', 'z_1step_per_min_cln_pr.csv',
    'x_2step_per_min_cln_pr.csv', 'y_2step_per_min_cln_pr.csv', 'z_2step_per_min_cln_pr.csv',
    'x_anomaly_detection_3_step_per_min_clean.csv', 'y_anomaly_detection_3_step_per_min_clean.csv',
    'z_anomaly_detection_3_step_per_min_clean.csv'
]
output_file_2 = [f.replace('.csv', '_feat_prepr.csv') for f in input_file_0]

# --- files for data_process = 3 ---

input_path_3 = output_path_2
output_path_3 = './3_FEATS/'

# Create output directory if it doesn't exist
os.makedirs(output_path_3, exist_ok=True) 

input_file_3 = output_file_2
output_file_3 = [f.replace('.csv', '_feat.csv') for f in input_file_0]

# --- files for data_process = 4 ---

input_path_4 = output_path_3
output_path_4 = './4_FEATS_COMBINED/'
os.makedirs(output_path_4, exist_ok=True)

# --- files for data_process = 5 ---

input_path_5 = output_path_4
output_path_5 = './5_FEATS_SELECTION/'

# Create output directory if it doesn't exist
os.makedirs(output_path_5, exist_ok=True) 

input_file_5 = ['X_train.csv',
                'y_train.csv',
                ]
output_file_5 = 'X_train_reduced_idx_py.csv'
weights_file = 'Python_relieff_feature_indices_weights.csv'

# --- files for plots ---

# Define the path for saving plots
def plot (output_path):
    plot_path = os.path.join(output_path, 'PLOTS')
    # Create output directory if it doesn't exist
    os.makedirs(plot_path, exist_ok=True)
    return plot_path

# ---------------- Data Process 0: for cleaning raw data ----------------
def stage_0():
    print('\n ======= Data Process: 0 =======\n')
    # ---------------- Import Libraries ------------------
    import csv
    import matplotlib.pyplot as plt
    # ----------------------------------------------------
    
    # ---- Check input files before processing ----
    for f in input_file_0:
        path = os.path.join(input_path_0, f)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'No _raw.csv files found in {input_path_0}. Cannot build clean.csv. Please provide the raw data files.')

    # ================= Step 1: Read & Clean =================
    
    train_labels = [
        'time (ms)', 'acceleration x', 'acceleration y', 'acceleration z',
        'gyro x', 'gyro y', 'gyro z'
    ]
    
    datasets = []
    row_counts = []
    
    for file_idx, in_name in enumerate(input_file_0):
        in_file = os.path.join(input_path_0, in_name)
        out_file = os.path.join(output_path_0, output_file_0[file_idx])
        print(f'Processing file: {input_file_0[file_idx]}')

        # Number of extra lines to remove at the start of the file
        extra_lines = 17

        # Remove first extra lines
        with open(in_file, 'r', newline='', errors='replace') as infile:
            lines = list(csv.reader(infile))
        with open(out_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            for i, row in enumerate(lines, start=1):
                if i > extra_lines:
                    writer.writerow(row)

        # Read CSV with headers manually added
        df = pd.read_csv(out_file, names=train_labels)
        
        # Find starting time and sort
        start_time = df['time (ms)'].dropna().min()
        df = df.sort_values('time (ms)').reset_index(drop=True)
        
        # Limit time range to 1 hour and 1 min so the window value does not effect the wanted data (wanted data =< 1 hour data)
        df = df[df['time (ms)'] <= 3660000 + start_time]  
        
        # Store for later processing
        datasets.append(df)
        row_counts.append(len(df))
        
    # ================= Step 2: Find min rows (multiple of window) =================
    
    min_rows = min(row_counts)
    adjusted_rows = (min_rows // window) * window  # largest multiple of window <= min_rows
    
    if adjusted_rows != min_rows:
        print(f'\nMinimum rows {min_rows} adjusted down to {adjusted_rows} to fit window size {window}.\n')
    else:
        print(f'\nMinimum rows {min_rows} fits perfectly into windows of size {window}.\n')
    
    # ================= Step 3: Trim and Save clean dataset to disk =================
    
    for file_idx, df in enumerate(datasets):
        trimmed_df = df.iloc[:adjusted_rows].copy()
        out_file = os.path.join(output_path_0, output_file_0[file_idx])
        trimmed_df.to_csv(out_file, index=False)
        # Print output file name
        print(f'Saved trimmed file: {output_file_0[file_idx]} with {len(trimmed_df)} rows.')
    print('\nAll datasets aligned and trimmed successfully.\n')
    
    # ================= Step 4: Plotting all datasets for inspection =================
    
    if pl == 1:     
        # Plot sensor data
        plt.style.use('ggplot')
        fig, axs = plt.subplots(len(datasets), 2, figsize=(14, 2 * len(datasets)), sharex=False)
        
        if len(datasets) == 1:  # Handle single dataset case
            axs = [axs]
        
        for i, df in enumerate(datasets):
            # Convert time to minutes
            time_minutes = (df['time (ms)'] - df['time (ms)'].min()) / 60000.0
        
            # Accelerometer subplot
            axs[i][0].plot(time_minutes, df['acceleration x'], label='Acc X', color='red')
            axs[i][0].plot(time_minutes, df['acceleration y'], label='Acc Y', color='green')
            axs[i][0].plot(time_minutes, df['acceleration z'], label='Acc Z', color='blue')
            axs[i][0].legend(loc='upper right', fontsize=8)
            axs[i][0].set_title(f'{input_file_0[i]} - Accelerometer', fontsize=9)
        
            # X ticks and label
            axs[i][0].set_xticks([0, 30, 60])
            axs[i][0].set_xticklabels(['0', '30', '60'], fontsize=7)
            if i == len(datasets) - 1:  # Only label bottom plots
                axs[i][0].set_xlabel('Time (min)', fontsize=9)
        
            # Y ticks with rotation
            acc_min = round(min(df[['acceleration x','acceleration y','acceleration z']].min()), 2)
            acc_max = round(max(df[['acceleration x','acceleration y','acceleration z']].max()), 2)
            axs[i][0].set_yticks([acc_min, 0, acc_max])
            axs[i][0].set_yticklabels([str(acc_min), '0', str(acc_max)], fontsize=7, rotation=45)
        
            # Gyroscope subplot
            axs[i][1].plot(time_minutes, df['gyro x'], label='Gyro X', color='red')
            axs[i][1].plot(time_minutes, df['gyro y'], label='Gyro Y', color='green')
            axs[i][1].plot(time_minutes, df['gyro z'], label='Gyro Z', color='blue')
            axs[i][1].legend(loc='upper right', fontsize=8)
            axs[i][1].set_title(f'{input_file_0[i]} - Gyroscope', fontsize=9)
        
            # X ticks and label
            axs[i][1].set_xticks([0, 30, 60])
            axs[i][1].set_xticklabels(['0', '30', '60'], fontsize=7)
            if i == len(datasets) - 1:  # Only label bottom plots
                axs[i][1].set_xlabel('Time (min)', fontsize=9)
        
            # Y ticks with rotation
            gyro_min = round(min(df[['gyro x','gyro y','gyro z']].min()), 2)
            gyro_max = round(max(df[['gyro x','gyro y','gyro z']].max()), 2)
            axs[i][1].set_yticks([gyro_min, 0, gyro_max])
            axs[i][1].set_yticklabels([str(gyro_min), '0', str(gyro_max)], fontsize=7, rotation=45)
    
        fig.suptitle('Accelerometer & Gyroscope Data Morphology', fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        
        plot_name = 'data_morphology_overview.png'
        plot_path_all = os.path.join(plot(output_path_0), plot_name)
        plt.savefig(plot_path_all, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f'File saved as {plot_name}')
        print('=' * 70)
    
# ---------------- Data Process 1: Preprocess  1 step/min, 2 step/min datasets----------------
def stage_1():
    print('\n======= Data Process: 1 =======\n')
    # ---------------- Import Libraries ------------------
    import matplotlib.pyplot as plt
    # ----------------------------------------------------
    
    # ================= Step 1: Repeat 1 step/min, 2 step/min datasets movements =================
    
    # Loop over each row (sensor: x, y, z)
    for axis_index in range(input_file_1 .shape[0]):
        
        # Loop over each column (1 step/min, 2 step/min)
        for rate_index in range(input_file_1 .shape[1]):
            
            # ---- Check input files before processing ----
            file_name = input_file_1 .iloc[axis_index, rate_index]
            print(f'Processing file: {file_name}')

            try:
                df = pd.read_csv(input_path_1 + file_name)
            except FileNotFoundError:
                raise FileNotFoundError(f'No _clean.csv files found in {input_path_1}. Cannot build cln_pr.csv.')
            
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
            output_name = output_names.iloc[axis_index, rate_index]        
            final_df.to_csv(output_path_1+output_name, index=False)
            
            # Print output file name
            print(f'Saving file: {output_name}\n')
    
    # ================= Step 2: Plotting all datasets for inspection =================        
    
    if pl == 1:   
        
        # Gather processed datasets for plotting
        processed_datasets = []
        processed_names = []
        
        for axis_index in range(input_file_1.shape[0]):
            for rate_index in range(input_file_1.shape[1]):
                file_name = output_names.iloc[axis_index, rate_index]
                file_path = os.path.join(output_path_1, file_name)
                if os.path.isfile(file_path):
                    df = pd.read_csv(file_path)
                    processed_datasets.append(df)
                    processed_names.append(file_name)
        
        # Create subplots
        fig, axs = plt.subplots(len(processed_datasets), 2, figsize=(14, 2 * len(processed_datasets)), sharex=False)
        if len(processed_datasets) == 1:
            axs = [axs]
        
        for i, df in enumerate(processed_datasets):
            # Convert time to minutes
            time_minutes = (df['time (ms)'] - df['time (ms)'].min()) / 60000.0
        
            # Accelerometer subplot
            axs[i][0].plot(time_minutes, df['acceleration x'], label='Acc X', color='red')
            axs[i][0].plot(time_minutes, df['acceleration y'], label='Acc Y', color='green')
            axs[i][0].plot(time_minutes, df['acceleration z'], label='Acc Z', color='blue')
            axs[i][0].legend(loc='upper right', fontsize=8)
            axs[i][0].set_title(f'{processed_names[i]} - Accelerometer', fontsize=9)
        
            axs[i][0].set_xticks([0, 30, 60])
            axs[i][0].set_xticklabels(['0', '30', '60'], fontsize=7)
            if i == len(processed_datasets) - 1:
                axs[i][0].set_xlabel('Time (min)', fontsize=9)
        
            acc_min = round(min(df[['acceleration x','acceleration y','acceleration z']].min()), 2)
            acc_max = round(max(df[['acceleration x','acceleration y','acceleration z']].max()), 2)
            axs[i][0].set_yticks([acc_min, 0, acc_max])
            axs[i][0].set_yticklabels([str(acc_min), '0', str(acc_max)], fontsize=7, rotation=45)
        
            # Gyroscope subplot
            axs[i][1].plot(time_minutes, df['gyro x'], label='Gyro X', color='red')
            axs[i][1].plot(time_minutes, df['gyro y'], label='Gyro Y', color='green')
            axs[i][1].plot(time_minutes, df['gyro z'], label='Gyro Z', color='blue')
            axs[i][1].legend(loc='upper right', fontsize=8)
            axs[i][1].set_title(f'{processed_names[i]} - Gyroscope', fontsize=9)
        
            axs[i][1].set_xticks([0, 30, 60])
            axs[i][1].set_xticklabels(['0', '30', '60'], fontsize=7)
            if i == len(processed_datasets) - 1:
                axs[i][1].set_xlabel('Time (min)', fontsize=9)
        
            gyro_min = round(min(df[['gyro x','gyro y','gyro z']].min()), 2)
            gyro_max = round(max(df[['gyro x','gyro y','gyro z']].max()), 2)
            axs[i][1].set_yticks([gyro_min, 0, gyro_max])
            axs[i][1].set_yticklabels([str(gyro_min), '0', str(gyro_max)], fontsize=7, rotation=45)
        
        fig.suptitle('Stage 1: Accelerometer & Gyroscope Data Morphology', fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        
        plot_name = '1and2_steps_per_min_morphology_overview.png'
        plot_file = os.path.join(plot(output_path_0), plot_name)
        plt.savefig(plot_file, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()       

# ---------------- Data Process 2: Feature Extraction ----------------
def stage_2():
    print('\n======= Data Process: 2 =======\n')
    # ---------------- Import Libraries ------------------
    from scipy.fft import fft
    from scipy.stats import iqr
    # ----------------------------------------------------
    
    # ---- Check input files before processing ----
    for f in input_file_2:
        path = os.path.join(input_path_2, f)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'No _clean.csv files found in {input_path_2}. Cannot build feat_prepr.csv.')
    
    for file_idx in range(len(input_file_2)):  
        df = pd.read_csv(input_path_2 + input_file_2[file_idx])
        print(f'Processing file: {input_file_2[file_idx]}')
        
        # Normalize acceleration to acceleration in g (9.80665 m/s^2)
        #--------------------------------------------------------------------
        df['acc_g_x'] = (df['acceleration x'] / 9.80665).round(3)
        df['acc_g_y'] = (df['acceleration y'] / 9.80665).round(3)
        df['acc_g_z'] = (df['acceleration z'] / 9.80665).round(3)
        #--------------------------------------------------------------------
        
        # Compute vector magnitudes
        #--------------------------------------------------------------------
        df['Acceleration Sum Vector Magnitude'] = np.sqrt(
            df['acc_g_x']**2 + df['acc_g_y']**2 + df['acc_g_z']**2).round(3)
        df['Sum Vector Magnitude of Angular Velocity'] = np.sqrt(
            df['gyro x']**2 + df['gyro y']**2 + df['gyro z']**2).round(3)
        #--------------------------------------------------------------------
        
        # Cubic product features
        #--------------------------------------------------------------------
        df['Acceleration Cubic Product Magnitude'] = (
            abs(df['acc_g_x'] * df['acc_g_y'] * df['acc_g_z']) ** (1/3)).round(3)
        df['Production Cubic Magnitude of Angular Velocity'] = (
            abs(df['gyro x'] * df['gyro y'] * df['gyro z']) ** (1/3)).round(3)
        #--------------------------------------------------------------------
        
        # Gradient features dt
        #--------------------------------------------------------------------
        gradient_targets = {
            'acc_g_x': 'da_x/dT', 'acc_g_y': 'da_y/dT', 'acc_g_z': 'da_z/dT',
            'gyro x': 'dg_x/dT', 'gyro y': 'dg_y/dT', 'gyro z': 'dg_z/dT'
        }
        for col, new_col in gradient_targets.items():
            df[new_col] = np.gradient(df[col], df['time (ms)']).round(3)
        #--------------------------------------------------------------------

        # Orientation angles (theta_x, theta_y, theta_z) in radians
        #--------------------------------------------------------------------
        # Gravity magnitude (~1 g if MPU6050 calibrated well)
        df['g_mag'] = (np.sqrt(df['acc_g_x']**2 + df['acc_g_y']**2 + df['acc_g_z']**2)).round(3)

        df['theta_x'] = (np.arccos(df['acc_g_x'] / df['g_mag'])).round(3)
        df['theta_y'] = (np.arccos(df['acc_g_y'] / df['g_mag'])).round(3)
        df['theta_z'] = (np.arccos(df['acc_g_z'] / df['g_mag'])).round(3)        

        # Cleanup helper column
        df.drop(columns=['g_mag'], inplace=True) 
        #--------------------------------------------------------------------
        
        # ================= Window-based features ================
        # Features computed over defined windows of data
        
        # Gravity vector features
        #--------------------------------------------------------------------       
        df['window_id'] = (df.index // window) # Create a new column for window IDs base on window size index

        # min, max, average
        #--------------------------------------------------------------------     
        sensor_cols = ['acc_g_x', 'acc_g_y', 'acc_g_z',
                       'gyro x', 'gyro y', 'gyro z']
        agg_funcs = ['mean', 'max', 'min']
        for col in sensor_cols:
            for func in agg_funcs:
                new_col = f'{col.replace(' ', '_')}_window_{func}'
                df[new_col] = df.groupby('window_id')[col].transform(func).round(3) 
        #--------------------------------------------------------------------        
        
        # Signal magnitude area SMA  
        #-------------------------------------------------------------------- 
        # Define sensor groups and feature names
        sensor_cols2 = [
            ['acc_g_x', 'acc_g_y', 'acc_g_z'],
            ['gyro x', 'gyro y', 'gyro z']
        ]    
        feats = ['Signal Magnitude Area Accelerometer', 'Signal Magnitude Area Gyroscope']
        for i, cols in enumerate(sensor_cols2):
            
            # Calculate row-wise magnitude sum: |x| + |y| + |z|
            df[f'_abs_sum_{i}'] = df[cols].abs().sum(axis=1)
            
            # Group by window and take mean of that sum to compute SMA
            sma_series = df.groupby('window_id')[f'_abs_sum_{i}'].mean()
            
            # Map SMA result back to the original dataframe
            df[feats[i]] = df['window_id'].map(sma_series).round(3)
            
            # Cleanup temporary columns
        df.drop(columns=['_abs_sum_0', '_abs_sum_1'], inplace=True)
        #--------------------------------------------------------------------
           
        def compute_feat_per_window(df, columns_to_process, feats, choice):
            df = df.copy()
            #df['window_id'] = (df.index // window) 
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
            return df
        
        sensor_cols = ['acc_g_x', 'acc_g_y', 'acc_g_z', 
                       'gyro x', 'gyro y', 'gyro z']
        feats = ['RMS','MAD','VAR','STD','IQR','FFT','ENERGY']
        
    
        # Root Mean Square (RMS), Mean Absolute Deviation (MAD), Variance (VAR), Standard Deviation (STD), 
        # Interquartile Range (IQR), Fast Fourier Transform (FFT), Energy  
        #--------------------------------------------------------------------
        for choice in range(len(feats)):
            df = compute_feat_per_window(df, sensor_cols, feats, choice)
        #--------------------------------------------------------------------
        df.drop(columns=['window_id'], inplace=True)
        # Saving feat to CSV
        df.to_csv(output_path_2 + output_file_2[file_idx], index=False)
        
        # Print output file name
        print(f'File saved as: {output_file_2[file_idx]}\n')

# ---------------- Data Process 3: ----------------
def stage_3():
    print('\n======= Data Process: 3 =======\n')
     # ================= Step 1: Find STDs in FFTs features =================
    fft_std_path = os.path.join(input_path_3, 'fft_std.csv')
    # ------- _feat_prepr.csv' Checker --------
    feat_files = sorted([f for f in os.listdir(input_path_3) if f.endswith('_feat_prepr.csv')])
    if not feat_files:
        raise FileNotFoundError(f'No _feat_prepr.csv files found in {input_path_3}. Cannot build fft_std.csv.')
    
    # reading first file header to discover FFT columns
    first_df = pd.read_csv(os.path.join(input_path_3, feat_files[0]), nrows=0)
    fft_columns = [c for c in first_df.columns if c.startswith('FFT_')]
    if not fft_columns:
        raise ValueError('No FFT_ columns found in feature files.')
        
    # reading only FFT columns from all files
    dfs = []
    for f in feat_files:
        p = os.path.join(input_path_3, f)
        df_tmp = pd.read_csv(p, usecols=fft_columns)
        dfs.append(df_tmp)
        print(f'Loaded FFT cols from {f}, shape {df_tmp.shape}')
        
    # stack and compute std across files (axis=0 => across files)
    stack = np.stack([d.values for d in dfs], axis=0)   # shape (n_files, n_rows, n_cols)
    std_values = np.std(stack, axis=0, ddof=0)          # shape (n_rows, n_cols)
    fft_std_df = pd.DataFrame(std_values, columns=fft_columns).round(3)
    fft_std_df.to_csv(fft_std_path, index=False)
    print(f'Created fft_std.csv with shape {fft_std_df.shape}')
    # ================= Step 2: Find the window line with the biggest std value =================
    n_rows = len(fft_std_df)
    
    # Number of full windows (it will skip an incomplete tail window) for those who will use only Data Process 3
    n_full_windows = n_rows // window
    if n_full_windows != n_rows / window:
        raise ValueError('window is larger than the number of rows; cannot process.')
    print(f'Using window = {window}; number of full windows = {n_full_windows}; skipping tail if incomplete.')
    
    # Prepare replacement: for each FFT column compute the position-score from fft_std
    best_idx_map = {}
    for col in fft_columns:
        scores = np.zeros(window, dtype=int)
        for w in range(n_full_windows):
            chunk = fft_std_df[col].iloc[w*window:(w+1)*window].values
            # exclude the first row of each chunk
            arg = int(np.nanargmax(chunk[1:])) + 1  # +1 to shift index since we sliced from [1:]
            scores[arg] += 1
        # choose the best index ignoring row 0
        best_idx_map[col] = int(np.argmax(scores[1:])) + 1
        # best_idx_map display is converted from python's counting0-31 to 1-32 
        print('_' * 70 + '\n' + 
              f'Processed {col}:\nbest_idx={best_idx_map[col]+1},\nscore_counts=\n{scores.tolist()}') 
    # ================= Step 3: FFT replacement with the FFT line via the biggest standard deviation of all datasets =================    
    
    # Processing files loop for FFT replacement                                                                                                                                                                                                            
    for file_idx in range(len(input_file_3)): 
        df = pd.read_csv(input_path_3 + input_file_3[file_idx])
        print('=' * 70 + '\n'+ f'Processing file: {input_file_2[file_idx]}\n'+'=' * 70)
        
        # Validate df has the FFT columns
        missing = [c for c in fft_columns if c not in df.columns]
        if missing:
            raise ValueError(f'The following FFT columns are missing from df: {missing}')
        if n_rows != len(df):
            print(f'Warning: fft_std.csv has {n_rows} rows, but df has {len(df)} rows.')
        
        for col in fft_columns:
            best_idx = best_idx_map[col]
            new_col = df[col].to_numpy(copy=True) # Build new column values by copying df value at (window_start + best_idx) into whole window
            for w in range(n_full_windows):
                val = df[col].iat[w*window + best_idx]
                new_col[w*window:(w+1)*window] = val
                
            # If there is an incomplete tail (n_rows % window != 0), we leave those rows unchanged
            df[col] = new_col # Replace df column with new values

        # ---- Save aggregated dataset ----
        out_file = os.path.join(output_path_3, output_file_3[file_idx])
        df.to_csv(out_file, index=False)
        # Print output file name
        print(f'Saved aggregated file: {output_file_3[file_idx]} with shape {df.shape}')

# ---------------- Data Process 4: ----------------
def stage_4():
    print('\n======= Data Process: 4 =======\n')

    # Collect all processed feature files from Stage 3
    feat_files = output_file_3  # already defined in your script
    
    all_X = []
    all_y = []
    all_raw_data = []   
    all_norm_data = [] 
    raw_labels = []
    norm_labels = []
    
    for file_idx, f in enumerate(feat_files):
        print(f'Processing file: {f}')
        df = pd.read_csv(os.path.join(input_path_4, f))
        
        # Explicit column selection for raw and normallize datasets
        df1 = df[['acceleration x', 'acceleration y', 'acceleration z',
                  'gyro x', 'gyro y', 'gyro z']]
        
        df2 = df[['acc_g_x', 'acc_g_y', 'acc_g_z',
                  'gyro x', 'gyro y', 'gyro z']]
        
        all_raw_data.append(df1)
        all_norm_data.append(df2)
        
        # Keeping labels aligned with row counts
        raw_labels.append(np.full(len(df1), file_idx))
        norm_labels.append(np.full(len(df2), file_idx))
        
        # Drop raw sensor columns (first 9)
        df = df.drop(columns=['time (ms)','acceleration x', 'acceleration y', 'acceleration z',
                              'gyro x', 'gyro y', 'gyro z','acc_g_x','acc_g_y', 'acc_g_z'], errors='ignore')
        
        # Window-based aggregation
        df['window_id'] = df.index // window
    
        # Columns for derivative features
        derivative_cols = ['da_x/dT', 'da_y/dT', 'da_z/dT']
        agg_dict = {}
    
        for col in df.columns:
            if col == 'window_id':
                continue
            if col in derivative_cols:
                agg_dict[col] = 'max'   # use max
            else:
                agg_dict[col] = 'median'  # use median
    
        # Aggregate per window
        df_windowed = df.groupby('window_id').agg(agg_dict).reset_index(drop=True).round(3)
    
        all_X.append(df_windowed)
        all_y.append(np.full(len(df_windowed), file_idx))  # dataset number as label
    
    # Concatenate into big DataFrames
    X_data = pd.concat(all_X, axis=0, ignore_index=True)
    y_data = pd.Series(np.concatenate(all_y), name='label')
    
    all_raw_data = pd.concat(all_raw_data, axis=0, ignore_index=True)
    all_norm_data = pd.concat(all_norm_data, axis=0, ignore_index=True)
    
    raw_y = np.concatenate(raw_labels)
    norm_y = np.concatenate(norm_labels)
    raw_y = pd.DataFrame(raw_y)
    norm_y = pd.DataFrame(norm_y)
    
    # Train - Test split before feature selection for models accuracy evaluation
    train_size=0.75
    test_size=0.25
    X_train, X_test, y_train, y_test = TrainTestSplit(X_data, y_data, train_size, test_size)
    
    raw_train, raw_test, y_raw_train, y_raw_test = TrainTestSplit(all_raw_data, raw_y, train_size, test_size)
    norm_train, norm_test, y_norm_train, y_norm_test = TrainTestSplit(all_norm_data, norm_y, train_size, test_size)

    # ---- Save datasets (NO headers, NO index) ----
    save_items = {
        # aggregated features
        'X_data.csv': X_data,
        'y_data.csv': y_data.to_frame(),   # ensure it has name 'label'
        'X_train.csv': pd.DataFrame(X_train, columns=X_data.columns),
        'y_train.csv': pd.DataFrame(y_train, columns=['label']),
        'X_test.csv': pd.DataFrame(X_test, columns=X_data.columns),
        'y_test.csv': pd.DataFrame(y_test, columns=['label']),
    
        # raw version
        'all_raw_data.csv': all_raw_data,
        'y_all_raw_data.csv': pd.DataFrame(raw_y, columns=['label']),
        'all_raw_train.csv': pd.DataFrame(raw_train, columns=all_raw_data.columns),
        'y_all_raw_train.csv': pd.DataFrame(y_raw_train, columns=['label']),
        'all_raw_test.csv': pd.DataFrame(raw_test, columns=all_raw_data.columns),
        'y_all_raw_test.csv': pd.DataFrame(y_raw_test, columns=['label']),
    
        # norm version
        'all_norm_data.csv': all_norm_data,
        'y_all_norm_data.csv': pd.DataFrame(norm_y, columns=['label']),
        'all_norm_train.csv': pd.DataFrame(norm_train, columns=all_norm_data.columns),
        'y_all_norm_train.csv': pd.DataFrame(y_norm_train, columns=['label']),
        'all_norm_test.csv': pd.DataFrame(norm_test, columns=all_norm_data.columns),
        'y_all_norm_test.csv': pd.DataFrame(y_norm_test, columns=['label']),
    }

    for fname, df in save_items.items():
        path = os.path.join(output_path_4, fname)
        df.to_csv(path, index=False, header=True)
        print(f'Saved {fname} with shape {df.shape}')

# ---------------- Data Process 5: ReliefF Feature Selection ----------------
def stage_5():
    print('\n======= Data Process: 5 =======\n') 
    # ---------------- Import Libraries ------------------     
    from skrebate import ReliefF
    # ----------------------------------------------------

    # Paths for X and y (from Stage 4 outputs)
    X_path = os.path.join(input_path_5, input_file_5[0])
    y_path = os.path.join(input_path_5, input_file_5[1])
    
    # Load without headers
    X_data = pd.read_csv(X_path,header=None, skiprows=1)
    y_data = pd.read_csv(y_path, header=None, skiprows=1).squeeze('columns')  # 1D Series
    
    print(f'Loaded X_data: {X_data.shape}')
    print(f'Loaded y_data: {y_data.shape}')
    
    # Convert to numpy arrays
    X = X_data.to_numpy(dtype=np.float32)
    y = y_data.to_numpy(dtype=np.int32) 
    
    # ReliefF (fix random_state to mimic MATLAB consistency)
    relieff = ReliefF(
        n_features_to_select=10,  # adjust as needed
        n_neighbors=100,
        discrete_threshold=10
    )
    relieff.fit_transform(X, y)
    
    # Save feature importance weights + indices
    weights = relieff.feature_importances_
    idx_sorted = np.argsort(weights)[::-1]  # descending order
    
    weights_df = pd.DataFrame({
        'Feature_Index': idx_sorted,
        'ReliefF_Weight': weights[idx_sorted]
    }).round(6)
    weights_df.to_csv(os.path.join(output_path_5, weights_file), index=False,float_format='%.6f')
    
    print(f'Saved ReliefF weights and indices: {weights_file}')

# ---------------- Data Process 6: ReliefF Feature Selection Plotting and 10 best features, displaying for ESP32 use  ----------------
def stage_6():
    print('\n======= Data Process: 6 =======\n')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from math import sqrt

    if matlab == 0:
        # ---- Paths ----
        weights_file = os.path.join(output_path_5, 'Python_relieff_feature_indices_weights.csv')     
        out_file1 = os.path.join(output_path_5, 'Python_X_train_top10.csv')
        out_file2 = os.path.join(output_path_5, 'Python_Top10_w_comp_efficient.csv')
        i = 0  # Python counting is from 0
    else:
        # ---- Paths ----
        weights_file = os.path.join(output_path_5, 'Matlab_relieff_feature_indices_weights.csv')
        out_file1 = os.path.join(output_path_5, 'Matlab_X_train_top10.csv')
        out_file2 = os.path.join(output_path_5, 'Matlab_Top10_w_comp_efficient.csv')
        i = 1  # Matlab counts from 1

    # ---- Step 1: Load weights ----
    weights_df = pd.read_csv(weights_file)
    print(f'Loaded ReliefF weights with shape {weights_df.shape}')

    # ---- Step 2: Load X_train with headers ----
    X_train_file = os.path.join(output_path_4, 'X_train.csv')
    X_train = pd.read_csv(X_train_file)   # must have headers from stage_4
    feature_names = X_train.columns

    # ---- Step 3: Extract top 10 feature indices ----
    top10_indices = weights_df['Feature_Index'].head(10).to_numpy()
    print('Top 10 feature indices:', top10_indices - i)

    # ---- Step 4: Select top 10 from X_train ----
    X_top10 = X_train.iloc[:, top10_indices-i]
    X_top10.to_csv(out_file1, index=False, header=True)
    print(f'Saved reduced X_train with top 10 features: {out_file1}, shape {X_top10.shape}')

    # ---- Consistent color mapping ---- 
    cmap = plt.cm.get_cmap('tab20', len(feature_names)) 
    color_map = {idx: cmap(idx) for idx in range(len(feature_names))}

    # ---- Step 5: Plot index vs weight ----
    plt.figure(figsize=(14,6))
    bars = plt.bar(
        weights_df['Feature_Index'] - i,
        weights_df['ReliefF_Weight'],
        color=[color_map[idx - i] for idx in weights_df['Feature_Index']]
    )

    for bar, weight in zip(bars, weights_df['ReliefF_Weight']):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height(),
                 f'{weight:.3f}',
                 va='top', fontsize=6, rotation=90)

    names = ['Python', 'Matlab']
    plt.xlabel('Feature Index')
    plt.ylabel('ReliefF Weight')
    plt.title(f'{names[i]} ReliefF Feature Importance')

    legend_handles = [
        mpatches.Patch(color=color_map[idx - i], label=f'{idx}: {feature_names[idx - i]}')
        for idx in weights_df['Feature_Index']
    ]
    plt.legend(
        handles=legend_handles,
        fontsize=6.5, loc='upper left', bbox_to_anchor=(1.02, 1),
        ncol=2, frameon=False, title=" Best Feature Index - Name"
    )

    plot_name1 = ['Python_relieff_weights_plot.png','Matlab_relieff_weights_plot.png']
    plt.tight_layout()
    plot_path1 = os.path.join(plot(output_path_5), plot_name1[i])
    plt.savefig(plot_path1, dpi=600)
    plt.show()
    plt.close()
    print(f'Saved plot: {plot_path1}')

    # ---- Step 6: ESP32 Feature Computation Time vs ReliefF Weights ----
 
    times_file = os.path.join(input_path_0, 'feats_computation_times.csv')
    if os.path.isfile(times_file):
        times = pd.read_csv(times_file, header=None).iloc[0].to_numpy()
        times = np.log1p(times)   # safer than log
        if len(times) != len(feature_names):
            raise ValueError(f"Mismatch: {len(times)} times vs {len(feature_names)} features")
    
        # --- Load weights ---
        weights_df = pd.read_csv(weights_file)
        weights_df['Feature_Index'] = weights_df['Feature_Index'] - i  # 0-based
        feature_names = X_train.columns.to_list()
    
        # Merge into DataFrame
        df = pd.DataFrame({
            'Feature_Index': np.arange(len(times)),
            'Feature_Name': feature_names,
            'Time_ms': times
        }).merge(
            weights_df[['Feature_Index', 'ReliefF_Weight']],
            on='Feature_Index',
            how='left'
        )
    
        # Define custom score: higher = better
        df['Max_Weight'] = df['ReliefF_Weight'].max()
        df['Max_Time']   = df['Time_ms'].max()
        df['Score'] = (df['ReliefF_Weight'] * df['Max_Time']) / (df['Time_ms'] * df['Max_Weight'])
    
        # Sort by score
        df_sorted = df.sort_values('Score', ascending=False).reset_index(drop=True)
        colors = [color_map[idx] for idx in df_sorted['Feature_Index']]
    
        # Top-10 by score (shift +1 so indices display 1–75)
        top10 = (df_sorted['Feature_Index'].head(10) + 1).tolist()
    
        # --- Plot ---
        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(
            np.arange(len(df_sorted)),
            df_sorted['Score'],
            color=colors,
            edgecolor='black'
        )
    
        # Label each bar with feature index (1–75)
        for ix, (bar, feat_idx) in enumerate(zip(bars, df_sorted['Feature_Index'])):
            feat_label = feat_idx + 1  # shift for human-readable indexing
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    str(feat_label), ha='center', fontsize=6,
                    color='cyan' if feat_label in top10 else 'black')
    
        # Highlight top-10
        for ix, feat_idx in enumerate(df_sorted['Feature_Index']):
            if (feat_idx + 1) in top10:
                bars[ix].set_edgecolor('cyan')
                bars[ix].set_linewidth(2)
    
        ax.set_xlabel('Features (sorted by Custom Score)')
        ax.set_ylabel('Custom Score (normalized weight/time)')
        ax.set_title(f'{names[i]} Feature Trade-off: Importance vs Computation Time')
    
        # --- Legend (index: name, from 1 to 75) ---
        legend_handles = [
            mpatches.Patch(color=color_map[idx], label=f'{idx+1}: {feature_names[idx]}')
            for idx in df_sorted['Feature_Index']
        ]
        plt.legend(
            handles=legend_handles,
            fontsize=6, loc='upper left', bbox_to_anchor=(1.02, 1),
            ncol=2, frameon=False, title="Feature Index → Name"
        )
    
        plot_name2 = ['Python_Feats_CustomScore.png', 'Matlab_Feats_CustomScore.png']
        plot_path2 = os.path.join(plot(output_path_5), plot_name2[i])
        plt.tight_layout()
        plt.savefig(plot_path2, dpi=600)
        plt.show()
        plt.close()
        print(f'Saved plot: {plot_path2}')
        
        # ---- Step 7: Select top 10 from X_train ----
        X_top10 = X_train.iloc[:, top10]
        X_top10.to_csv(out_file2, index=False, header=True)
        print(f'Saved reduced X_train with top 10 features: {out_file2}, shape {X_top10.shape}')
 
    else:
        print('Warning: feats_computation_times.csv not found, skipping ESP32 plot.')

# ============================= Auto Runner ===================================

if auto == 0:
    if data_process == 0: stage_0()
    elif data_process == 1: stage_1()
    elif data_process == 2: stage_2()
    elif data_process == 3: stage_3()
    elif data_process == 4: stage_4()
    elif data_process == 5: stage_5()
    elif data_process == 6: stage_6()
    
elif auto == 1:
    stage_0()
    stage_1()
    stage_2()
    stage_3()
    stage_4()
    if matlab == 0:
        stage_5()
    stage_6()