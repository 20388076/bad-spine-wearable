'''
Created on Thu Apr  3 20:08:53 2025

@author: Achillios Pitsilkas
'''
# -------------------------------  Working Directory -----------------------------
# Set the working directory to the script's location if running in Visual Studio Code
import os
# Change working directory for this script
os.chdir(r'C:\Users\user\OneDrive - MSFT\PlatformIO\PlatformIO\Projects\bad-spine-wearable-1\src\data_process_and_classification') # modify this path to your working directory

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
    elapsed = t2 - t1
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    milliseconds = int((elapsed * 1000) % 1000)
    print(f" time taken: {minutes:02d} min:{seconds:02d} s:{milliseconds:03d} millis") 
# ----------------------------- Kernel break ----------------------------------
def RETURN():
    sys.exit()
# =============================================================================

# ----------------------------- Kernel clean call -----------------------------
cls()

# ============================= MAIN PROGRAM ==================================
series_of_experiments = 2 #
# ------------------------------ Auto Runner Option ---------------------------
# 0: run only one stage; 1: run all stages
auto = 1
# ------------------------------ Data Process Option --------------------------
# 0: raw -> clean;  
# 1: clean -> features preprocessed; 
# 2: features preprocessed -> features final; 
# 3: features final -> X_data, y_data; 
# 4: a) X_data_train, y_data_train -> ReliefF selected features or only 
#    b) Plotting the weight order best features and combine the ESP32 computation time.

stage = 5

# ----------------------------- Matlab Option for ReleifF ---------------------
matlab = 1 # 0: python ReleifF ; 1: Matlab ReleifF   
# ----------------------------- Window Size -----------------------------------
# Define window in sec for data trimming to fit window size and window-based features per classifier
windows = [1,1] # sec  IF window_search = 0 <-- Change this table to set time window per classifier
# ----------------------------- Window Search Value -----------------------------------
window_search = 0 # Set 1 to search for the best time window from a list of time window named candidate_windows
if series_of_experiments == 1:
    candidate_windows = [1,2,4,6,8,10,20,40,60,80,100,120,140]  # in sec (8s - 2min, 10s)
else:
    candidate_windows = [0.8,1,2,4,6,8,10]
        

# ----------------------------- Sample Rate Dataset ---------------------------
# Available Datasets
# 1) 9.71 Hz
# 2) 10 Hz
# 3) 50 Hz
sampleRate = 9.71 # Sample rate in Hz     <-- Change this value to set sample rate
# ----------------------------- Classifier Factory ----------------------------
def get_classifier(cl):
    if cl == 0: # Decision Tree 
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(), 'DT'
    elif cl == 1:  # Random Forest 
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(), 'RF'
    
# if you add more classifiers change this value based on the overall number of classifiers
n_classifiers = 2  # DecisionTree, RandomForest
train_size = 1  # or 1.0 if you want full-dataset mode
test = 1 -  train_size 
test_size = test
# ----------------------------- File Configuration ----------------------------

def get_paths(stage, sampleRate, classifier_name, mode='exp'):
    base = f"{sampleRate}_Hz_sampling/{classifier_name}"
    if series_of_experiments == 1:
        exp_files = [f'x_1_{sampleRate}.csv', f'y_1_{sampleRate}.csv', f'z_1_{sampleRate}.csv',
                     f'x_2_{sampleRate}.csv', f'y_2_{sampleRate}.csv', f'z_2_{sampleRate}.csv',
                     f'x_3_{sampleRate}.csv', f'y_3_{sampleRate}.csv', f'z_3_{sampleRate}.csv',
                     f'x_4_{sampleRate}.csv', f'y_4_{sampleRate}.csv', f'z_4_{sampleRate}.csv',
                     f'x_5_{sampleRate}.csv', f'y_5_{sampleRate}.csv', f'z_5_{sampleRate}.csv']
        validate_files = [f'try_x_1_{sampleRate}.csv', f'try_y_1_{sampleRate}.csv', f'try_z_1_{sampleRate}.csv',
                          f'try_x_2_{sampleRate}.csv', f'try_y_2_{sampleRate}.csv', f'try_z_2_{sampleRate}.csv',
                          f'try_x_3_{sampleRate}.csv', f'try_y_3_{sampleRate}.csv', f'try_z_3_{sampleRate}.csv',
                          f'try_x_4_{sampleRate}.csv', f'try_y_4_{sampleRate}.csv', f'try_z_4_{sampleRate}.csv',
                          f'try_x_5_{sampleRate}.csv', f'try_y_5_{sampleRate}.csv', f'try_z_5_{sampleRate}.csv']
        exp_path = f'./0_RAW/series_of_experiments_1/{sampleRate}_Hz_sampling/'
    else:
        exp_files = [f'good_1_{sampleRate}.csv', f'good_2_{sampleRate}.csv', f'good_3_{sampleRate}.csv',
                     f'mid_1_{sampleRate}.csv', f'mid_2_{sampleRate}.csv', f'mid_3_{sampleRate}.csv',
                     f'bad_1_{sampleRate}.csv', f'bad_2_{sampleRate}.csv', f'bad_3_{sampleRate}.csv']
        validate_files = [f'test_good_1_{sampleRate}.csv', f'test_good_2_{sampleRate}.csv', f'test_good_3_{sampleRate}.csv',
                          f'test_mid_1_{sampleRate}.csv', f'test_mid_2_{sampleRate}.csv', f'test_mid_3_{sampleRate}.csv',
                          f'test_bad_1_{sampleRate}.csv', f'test_bad_2_{sampleRate}.csv', f'test_bad_3_{sampleRate}.csv']
        exp_path = f'./0_RAW/series_of_experiments_2/{sampleRate}_Hz_sampling/'
        
    files_to_use = exp_files if mode == 'exp' else validate_files

    if stage == 0:
        in_files = files_to_use
        in_path = exp_path
        out_path = f'./1_CLEAN/{base}/{"EXP" if mode=="exp" else "VALIDATE"}/'
        out_files = [f"{fname.replace('.csv', f'{classifier_name}_clean.csv')}" for fname in files_to_use]
    elif stage == 1:
        in_files = [f"{fname.replace('.csv', f'{classifier_name}_clean.csv')}" for fname in files_to_use]
        in_path = f'./1_CLEAN/{base}/{"EXP" if mode=="exp" else "VALIDATE"}/'
        out_path = f'./2_FEATS_PREPROCESSSED/{base}/{"EXP" if mode=="exp" else "VALIDATE"}/'
        out_files = [f"{fname.replace('.csv', f'{classifier_name}_feat_prepr.csv')}" for fname in files_to_use]
    elif stage == 2:
        in_files = [f"{fname.replace('.csv', f'{classifier_name}_feat_prepr.csv')}" for fname in files_to_use]
        in_path = f'./2_FEATS_PREPROCESSSED/{base}/{"EXP" if mode=="exp" else "VALIDATE"}/'
        out_path = f'./3_FEATS/{base}/{"EXP" if mode=="exp" else "VALIDATE"}/'
        out_files = [f"{fname.replace('.csv', f'{classifier_name}_feat.csv')}" for fname in files_to_use]
    elif stage == 3:
        in_files = [f"{fname.replace('.csv', f'{classifier_name}_feat.csv')}" for fname in files_to_use]
        in_path = f'./3_FEATS/{base}/{"EXP" if mode=="exp" else "VALIDATE"}/'
        out_path = f'./4_FEATS_COMBINED/{base}/'
        out_files = []     
    elif stage == 4:
        in_files = [f"X_data_{sampleRate}{classifier_name}.csv",
                 f"y_data_{sampleRate}{classifier_name}.csv"]
        in_path = f'./4_FEATS_COMBINED/{base}/'
        out_path = f'./5_FEATS_SELECTION/{base}/'
        time_path = exp_path 
    else:
        raise ValueError("Invalid stage")

    os.makedirs(out_path, exist_ok=True)
    
    return in_files, in_path, (out_files if stage != 4 else time_path), out_path
# **************** Stage Functions *********************
# ---------------- Data Process 0: for cleaning raw data ----------------
def stage_0(mode):
    print(f'\n ======= Data Process: 0 Mode:{mode} for {window}s =======\n')
    # ---------------- Import Libraries ------------------
    import os, pandas as pd, matplotlib.pyplot as plt, numpy as np, csv
    # ----------------------------------------------------
    
    # Define the path for saving plots
    def plot (output_path):
        plot_path = os.path.join(output_path, 'PLOTS')
        # Create output directory if it doesn't exist
        os.makedirs(plot_path, exist_ok=True)
        return plot_path
    
    input_file_0, input_path_0, output_file_0, output_path_0 = get_paths(0, sampleRate, classifier_name, mode)

    # ================= Step 1: Read & Clean =================
    
    train_labels = [
        't (ms)', 'a_x', 'a_y', 'a_z',
        'g_x', 'g_y', 'g_z'
    ]
    
    datasets = []
    row_counts = []
    
    for file_idx, in_name in enumerate(input_file_0):
        in_file = os.path.join(input_path_0, in_name)
        out_file = os.path.join(output_path_0, output_file_0[file_idx])
        print(f'Processing file: {input_file_0[file_idx]}')
        if series_of_experiments == 1:
            # Number of extra lines to remove at the start of each file
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
            start_time = df['t (ms)'].dropna().min()
            df = df.sort_values('t (ms)').reset_index(drop=True)
            
            # Limit time range to 1 hour and 1 min so the window value does not effect the wanted data (wanted data =< 1 hour data)
            df = df[df['t (ms)'] <= 3660000 + start_time]  
            # Store for later processing
            
        if series_of_experiments == 2:
            df = pd.read_csv(in_file)
            start_time = df['t (ms)'].dropna().min()
            time_limit_ms = 1 * 60 * 1000  # 1 minute
            print(f"[INFO] Limiting {in_name} to 1 minute ({time_limit_ms} ms)")

            df = df[df['t (ms)'] <= start_time + time_limit_ms]
            
        # Store for later processing
        datasets.append(df)
        row_counts.append(len(df))
        
    if series_of_experiments == 1:
        # --- Series 1: Align all datasets to the same window multiple ---
        min_rows = min(row_counts)
        adjusted_rows = (min_rows // window_size) * window_size
        if adjusted_rows != min_rows:
            print(f'\n[INFO] Minimum rows {min_rows} adjusted down to {adjusted_rows} to fit window size {window_size}.\n')
        else:
            print(f'\n[INFO] Minimum rows {min_rows} fits perfectly into windows of size {window_size}.\n')
    
        for file_idx, df in enumerate(datasets):
            trimmed_df = df.iloc[:adjusted_rows].copy()
            out_file = os.path.join(output_path_0, output_file_0[file_idx])
            trimmed_df.to_csv(out_file, index=False)
            print(f'Saved trimmed file: {output_file_0[file_idx]} with {len(trimmed_df)} rows.')
    
    else:
        # --- Series 2: Trim each file individually based on its own duration ---
        for file_idx, df in enumerate(datasets):
            total_rows = len(df)
            adjusted_rows = (total_rows // window_size) * window_size
            trimmed_df = df.iloc[:adjusted_rows].copy()
    
            base_name = os.path.basename(input_file_0[file_idx]).lower()
            duration_ms = trimmed_df["t (ms)"].iloc[-1] - trimmed_df["t (ms)"].iloc[0]
            duration_min = duration_ms / 60000.0
    
            out_file = os.path.join(output_path_0, output_file_0[file_idx])
            trimmed_df.to_csv(out_file, index=False)
    
            print(f"[INFO] {output_file_0[file_idx]} → kept {len(trimmed_df)} rows "
                  f"≈ {duration_min:.2f} min (window adjusted)")

    # ================= Plotting all datasets for inspection =================
    if pl == 1:
        print("[INFO] Stage 0 plotting enabled")
        datasets = {fname: pd.read_csv(os.path.join(output_path_0, fname)) for fname in output_file_0}
    
        # Axis labels for Series 1
        axis_labels = {'x': 'X Axis', 'y': 'Y Axis', 'z': 'Z Axis'}
    
        if series_of_experiments == 1:
            # ---- SERIES 1: {axis}_{number}_{rate}.csv ----
            exp_conditions = [f'_{i}_' for i in range(1, 6)]
            exp_names = [f'Exp {i}' for i in range(1, 6)]
    
            for axis, axis_name in axis_labels.items():
                fig, axs = plt.subplots(len(exp_conditions), 2, figsize=(12, 12), sharex=True)
                axs = np.atleast_2d(axs)
    
                for row, cond in enumerate(exp_conditions):
                    match_file = None
                    for fname in datasets.keys():
                        if axis in fname and cond in fname:
                            match_file = fname
                            break
    
                    if match_file is None:
                        print(f"[Warning] No matching file for axis '{axis}' condition '{cond}'")
                        continue
    
                    df = datasets[match_file]
                    time = (df['t (ms)'] - df['t (ms)'].min()) / 60000.0  # minutes
    
                    # Flexible column lookup
                    def find_col(df, names):
                        for n in names:
                            if n in df.columns:
                                return n
                        return None
    
                    a_x = find_col(df, ['a_x', 'AccX', 'A_X', 'ax'])
                    a_y = find_col(df, ['a_y', 'AccY', 'A_Y', 'ay'])
                    a_z = find_col(df, ['a_z', 'AccZ', 'A_Z', 'az'])
                    g_x = find_col(df, ['g_x', 'GyroX', 'G_X', 'gx'])
                    g_y = find_col(df, ['g_y', 'GyroY', 'G_Y', 'gy'])
                    g_z = find_col(df, ['g_z', 'GyroZ', 'G_Z', 'gz'])
    
                    if a_x and a_y and a_z:
                        axs[row, 0].plot(time, df[a_x], 'r', label='Acc X')
                        axs[row, 0].plot(time, df[a_y], 'g', label='Acc Y')
                        axs[row, 0].plot(time, df[a_z], 'b', label='Acc Z')
                        axs[row, 0].set_ylabel('Acceleration (m/s$^2$)', fontsize=10)
                    if g_x and g_y and g_z:
                        axs[row, 1].plot(time, df[g_x], 'r', label='Gyro X')
                        axs[row, 1].plot(time, df[g_y], 'g', label='Gyro Y')
                        axs[row, 1].plot(time, df[g_z], 'b', label='Gyro Z')
                        axs[row, 1].set_ylabel('Angular Velocity (rad/s)', fontsize=10)
    
                    axs[row, 0].set_title(f"{axis_name} - {exp_names[row]} - Acc")
                    axs[row, 1].set_title(f"{axis_name} - {exp_names[row]} - Gyro")
    
                fig.suptitle(f'{axis_name} - Step Conditions', fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plot_path = os.path.join(plot(output_path_0), f'{axis}_axis_plot.png')
                plt.savefig(plot_path, dpi=600)
                plt.show()
                plt.close(fig)
    
        else:
            # ---- SERIES 2: dynamically group by experiment (_1, _2, _3) and condition ----
            conditions = ['good', 'mid', 'bad']
            experiment_nums = ['_1', '_2', '_3']

            def find_col(df, names):
                for n in names:
                    if n in df.columns:
                        return n
                return None

            for exp_tag in experiment_nums:
                fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
                axs = np.atleast_2d(axs)
                fig.suptitle(f'Series 2 - Experiment {exp_tag[-1]} Overview', fontsize=14)

                for row, cond in enumerate(conditions):
                    # Find files matching both the condition and the experiment number
                    cond_files = [
                        fname for fname in datasets.keys()
                        if cond in fname.lower() and exp_tag in fname.lower()
                    ]

                    if not cond_files:
                        print(f"[Warning] No files found for condition '{cond}' and experiment '{exp_tag}'")
                        continue

                    all_acc, all_gyro = [], []

                    # Read and store all runs for this condition/experiment
                    for fname in cond_files:
                        df = datasets[fname]
                        time = (df['t (ms)'] - df['t (ms)'].min()) / 60000.0  # convert ms → minutes
                        
                        # Detect if file was 3-minute limited (contains "_1_")
                        
                        a_x = find_col(df, ['a_x', 'AccX', 'A_X', 'ax'])
                        a_y = find_col(df, ['a_y', 'AccY', 'A_Y', 'ay'])
                        a_z = find_col(df, ['a_z', 'AccZ', 'A_Z', 'az'])
                        g_x = find_col(df, ['g_x', 'GyroX', 'G_X', 'gx'])
                        g_y = find_col(df, ['g_y', 'GyroY', 'G_Y', 'gy'])
                        g_z = find_col(df, ['g_z', 'GyroZ', 'G_Z', 'gz'])

                        if a_x and a_y and a_z:
                            all_acc.append((time, df[a_x], df[a_y], df[a_z]))
                        if g_x and g_y and g_z:
                            all_gyro.append((time, df[g_x], df[g_y], df[g_z]))

                    # Plot accelerometer data
                    for t, ax, ay, az in all_acc:
                        axs[row, 0].plot(t, ax, 'r', label='Accel X', alpha=0.7)
                        axs[row, 0].plot(t, ay, 'g', label='Accel Y', alpha=0.7)
                        axs[row, 0].plot(t, az, 'b', label='Accel Z', alpha=0.7)
                    axs[row, 0].set_title(f"{cond.upper()} - Accelerometer")
                    axs[row, 0].legend(fontsize=8)
                    axs[row, 0].set_ylabel('Acceleration (m/s$^2$)', fontsize=10)
                    # Plot gyroscope data
                    for t, gx, gy, gz in all_gyro:
                        axs[row, 1].plot(t, gx, 'r', label='Gyro X', alpha=0.7)
                        axs[row, 1].plot(t, gy, 'g', label='Gyro Y', alpha=0.7)
                        axs[row, 1].plot(t, gz, 'b', label='Gyro Z', alpha=0.7)
                    axs[row, 1].set_title(f"{cond.upper()} - Gyroscope")
                    axs[row, 1].legend(fontsize=8)
                    axs[row, 1].set_ylabel('Angular Velocity (rad/s)', fontsize=10)

                # Label bottom row
                axs[-1, 0].set_xlabel('Time (min)')
                axs[-1, 1].set_xlabel('Time (min)')
                plt.tight_layout(rect=[0, 0, 1, 0.96])

                plot_name = f'series2_experiment_{exp_tag[-1]}_conditions.png'
                plot_path = os.path.join(plot(output_path_0), plot_name)
                plt.savefig(plot_path, dpi=600)
                plt.show()
                plt.close(fig)
            
    return 

# ---------------- Data Process 1: Feature Extraction ----------------
def stage_1(mode):
    print(f'\n======= Data Process: 1 Mode:{mode} for {window}s  =======\n')
    t1=tic()
    # ---------------- Import Libraries ------------------
    import os, pandas as pd, numpy as np, multiprocessing
    from scipy.fft import fft
    from scipy.stats import iqr
    from joblib import Parallel, delayed
    # ----------------------------------------------------
    input_file_1, input_path_1, output_file_1, output_path_1 = get_paths(1, sampleRate, classifier_name, mode)
    
    def process_file(file_idx, fname):  
        df = pd.read_csv(input_path_1 + input_file_1[file_idx])
        # print(f'Processing file: {input_file_1[file_idx]}')
        
        # Normalize acceleration to acceleration in g (9.80665 m/s^2)
        #--------------------------------------------------------------------
        #df[['ag_x','ag_y','ag_z']] = (df[['a_x','a_y','a_z']] / 9.80665).round(3)
        df[['ag_x','ag_y','ag_z']] = df[['a_x','a_y','a_z']] 
        #--------------------------------------------------------------------
        
        # Compute vector magnitudes
        #--------------------------------------------------------------------
        df['SVM_a'] = np.sqrt((df[['ag_x','ag_y','ag_z']]**2).sum(axis=1)).round(3)
        df['SVM_g'] = np.sqrt((df[['g_x','g_y','g_z']]**2).sum(axis=1)).round(3)
        #--------------------------------------------------------------------
        
        # Cubic product features
        #--------------------------------------------------------------------
        df['CM_a'] = np.cbrt((df[['ag_x','ag_y','ag_z']].prod(axis=1)).abs()).round(3)
        df['CM_g'] = np.cbrt((df[['g_x','g_y','g_z']].prod(axis=1)).abs()).round(3)
        #--------------------------------------------------------------------
        
        # Gradient features dt
        #--------------------------------------------------------------------
        for col, new_col in {
            'ag_x':'jerk_x','ag_y':'jerk_y','ag_z':'jerk_z',
            'g_x':'accl_x','g_y':'accl_y','g_z':'accl_z'
        }.items():
            df[new_col] = np.gradient(df[col], df['t (ms)']).round(3)
        #--------------------------------------------------------------------

        # Orientation angles (theta_x, theta_y, theta_z) in radians
        #--------------------------------------------------------------------
        # Gravity magnitude (~1 g if MPU6050 calibrated well)
        g_mag = np.sqrt((df[['ag_x','ag_y','ag_z']]**2).sum(axis=1))
        df['th_x'] = np.arccos(df['ag_x']/g_mag).round(3)
        df['th_y'] = np.arccos(df['ag_y']/g_mag).round(3)
        df['th_z'] = np.arccos(df['ag_z']/g_mag).round(3)
        #--------------------------------------------------------------------
        
        # ================= Window-based features ================
        # Features computed over defined windows of data
        
        # Gravity vector features
        #--------------------------------------------------------------------       
        df['window_id'] = (df.index //window_size) # Create a new column for window IDs base on window size index

        # min, max, average
        #--------------------------------------------------------------------     
        sensor_cols = ['ag_x', 'ag_y', 'ag_z',
                       'g_x', 'g_y', 'g_z']
        agg_funcs = ['mean', 'max', 'min']
        grouped = df.groupby('window_id')[sensor_cols]
        
        agg_df = grouped.agg(agg_funcs)
        agg_df.columns = [f'{c}_{stat}' for c, stat in agg_df.columns]
        df = df.merge(agg_df, on='window_id', how='left').round(3)
        #--------------------------------------------------------------------        
        
        # Signal magnitude area SMA  
        #-------------------------------------------------------------------- 
        # Define sensor groups and feature names
        sensor_cols2 = [
            ['ag_x', 'ag_y', 'ag_z'],
            ['g_x', 'g_y', 'g_z']
        ]    
        feats = ['SMA_a', 'SMA_g']
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
        
        # Compute RMS, MAD, VAR, STD, IQR, FFT, Energy
        def compute_feat_per_window(df, columns_to_process, feats, choice):
            df = df.copy()
            for col in columns_to_process:
                feat_col_name = f'{feats[choice]}_{col}'
                df[feat_col_name] = np.nan
            grouped = df.groupby('window_id')       
            for window_id, group in grouped:
                for col in columns_to_process:
                    if choice == 0: # RMS
                        val = np.sqrt(sum(group[col] ** 2)/ window_size) 
                    elif choice == 1: # MAD
                        val = (sum(np.abs(group[col] - group[col].mean()))) / window_size
                    elif choice == 2: # VAR
                        val = np.var(group[col])
                    elif choice == 3: # STD
                        val = np.std(group[col])
                    elif choice == 4: # IQR
                        val = iqr(group[col])
                    elif choice == 5: # FFT
                        val = np.real(fft(group[col]))
                    elif choice == 6: # E
                        val = (np.sum(abs(fft(group[col])**2))) / window_size
                    df.loc[df['window_id'] == window_id, f'{feats[choice]}_{col}'] = val
            df[[f'{feats[choice]}_{col}' for col in columns_to_process]] = df[[f'{feats[choice]}_{col}' 
                                                                               for col in columns_to_process]].round(3)
            return df
        sensor_cols = ['ag_x', 'ag_y', 'ag_z', 
                       'g_x', 'g_y', 'g_z']
        feats = ['RMS','MAD','VAR','STD','IQR','FFT','E']
    
        # Root Mean Square (RMS), Mean Absolute Deviation (MAD), Variance (VAR), Standard Deviation (STD), 
        # Interquartile Range (IQR), Fast Fourier Transform (FFT), Energy (E) 
        #--------------------------------------------------------------------
        for choice in range(len(feats)):
            df = compute_feat_per_window(df, sensor_cols, feats, choice)
        #--------------------------------------------------------------------
        df.drop(columns=['window_id'], inplace=True)
        
        # Saving feat to CSV
        outname = output_file_1[file_idx]
        df.to_csv(os.path.join(output_path_1, outname), index=False)
        
        # Print output file name
        # print(f'\nFile saved as: {outname}\n')
        return 
        
    # ---- Check input files before processing ----
    for f in input_file_1:
        path = os.path.join(input_path_1, f)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'No _clean.csv files found in {input_path_1}. Cannot build feat_prepr.csv.')    
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_file)(i, f) for i, f in enumerate(input_file_1)
    )
    t2 = tic()
    toc(t1, t2-t1)
    return 
        
# ---------------- Data Process 2: ----------------
def stage_2(mode):
    print(f'\n======= Data Process: 2 Mode:{mode} for {window}s  =======\n')
    # ---------------- Import Libraries ------------------
    import os, pandas as pd, numpy as np, json, shutil 
    # ----------------------------------------------------
    
    input_file_2, input_path_2, output_file_2, output_path_2 = get_paths(2, sampleRate, classifier_name, mode)
    fft_std_path = os.path.join(output_path_2, '{series_of_experiments}fft_std{classifier_name}.csv')
    best_index_path = os.path.join('./', f'{series_of_experiments}best_fft_index{classifier_name}.json')
    
    feat_files = sorted([f for f in os.listdir(input_path_2) if f.endswith('_feat_prepr.csv')])
    if not feat_files:
        print(f"[Stage 2 - {mode.upper()}] No _feat_prepr.csv files found in {input_path_2}. Skipping stage.")
        return
    
    # Detect FFT columns
    first_df = pd.read_csv(os.path.join(input_path_2, feat_files[0]), nrows=0)
    fft_columns = [c for c in first_df.columns if c.startswith('FFT_')]
    # If no FFT columns: copy/rename the file to output folder as the corresponding _feat.csv
    if not fft_columns:
       for file_idx, fname in enumerate(input_file_2):
           df_path = os.path.join(input_path_2, fname)
           src = df_path
           dst = os.path.join(output_path_2, output_file_2[file_idx])  # expected _feat.csv name
           try:
               shutil.copy2(src, dst)
               print(f"[Stage 2 - {mode.upper()}] No FFT columns in {fname}. Copied to {dst}.")
           except Exception as e:
               print(f"[Stage 2 - {mode.upper()}] Failed to copy {src} -> {dst}: {e}")
           continue
    # ===========================================================
    # VALIDATION MODE → load precomputed best indices (safe skip)
    # ===========================================================
    if mode == 'validate':
        if not os.path.exists(best_index_path):
            print("[Stage 2 - VALIDATE] No best_fft_index.json found. Skipping validation stage 2.")
            return
        with open(best_index_path, 'r') as f:
            best_idx_map = json.load(f)
        print(f"[Stage 2 - VALIDATE] Loaded best FFT indices from {best_index_path}")
    
    # ===========================================================
    # TRAINING MODE → compute std and best index using only train_size
    # ===========================================================
    else:
        dfs = []
        for f in feat_files:
            # Read only FFT columns
            df_tmp = pd.read_csv(os.path.join(input_path_2, f), usecols=fft_columns)
            n_rows = len(df_tmp)
            n_full_windows = n_rows // window_size
        
            if n_full_windows == 0:
                print(f"[Stage 2 - {mode.upper()}] {f} has insufficient rows for one full window. Skipping.")
                continue
        
            # Calculate per-file training rows (must be whole windows)
            train_windows = int(n_full_windows * train_size)
            train_rows = train_windows * window_size
        
            if train_rows == 0:
                print(f"[Stage 2 - {mode.upper()}] {f} has too few rows for training. Skipping.")
                continue
        
            # Keep only the training portion (no window cut)
            df_train = df_tmp.iloc[:train_rows].copy()
            dfs.append(df_train)
        
            print(f"[Stage 2 - EXP] File {f}: using {train_windows}/{n_full_windows} windows "
                  f"({train_size*100:.1f}%) for FFT std computation ({train_rows}/{n_rows} rows).")
        
        # If no valid FFT data collected, exit early
        if not dfs:
            print(f"[Stage 2 - {mode.upper()}] No valid training FFT data found. Skipping stage.")
            return
        
        # Stack all training portions from each file

        train_stack = np.stack([d.values for d in dfs], axis=0)
        # Compute std across all training portions
        std_values = np.std(train_stack, axis=0) 
        fft_std_df = pd.DataFrame(std_values, columns=fft_columns).round(3)
        fft_std_df.to_csv(fft_std_path, index=False)
        
        print(f"[Stage 2 - EXP] Saved training-only fft_std.csv ({fft_std_df.shape})")
    
        # Compute best index per FFT column based on std within training windows
        best_idx_map = {}
        for col in fft_columns:
            scores = np.zeros(window_size, dtype=int)
            for w in range(n_full_windows):
                chunk = fft_std_df[col].iloc[w*window_size:(w+1)*window_size].values
                if len(chunk) < 2:
                    continue
                arg = int(np.nanargmax(chunk[1:])) + 1
                scores[arg] += 1
            best_idx_map[col] = int(np.argmax(scores[1:])) + 1
            
        with open(best_index_path, 'w') as f:
            json.dump(best_idx_map, f, indent=2)
        print(f"[Stage 2 - EXP] Saved best FFT indices to {best_index_path}")    
            
        header_path = os.path.join("../", f"{series_of_experiments}best_fft_index{classifier_name}.h")
        
        with open(header_path, "w", encoding="utf-8") as f:
            f.write("#pragma once\n\n")
            f.write("// Auto-generated header for best FFT indices\n")
            f.write("// Generated by Stage 2 in Python\n\n")
        
            f.write("const int FFT_BEST_INDEX_AG_X = {};\n".format(best_idx_map.get("FFT_ag_x", 0)))
            f.write("const int FFT_BEST_INDEX_AG_Y = {};\n".format(best_idx_map.get("FFT_ag_y", 0)))
            f.write("const int FFT_BEST_INDEX_AG_Z = {};\n".format(best_idx_map.get("FFT_ag_z", 0)))
            f.write("const int FFT_BEST_INDEX_G_X  = {};\n".format(best_idx_map.get("FFT_g_x", 0)))
            f.write("const int FFT_BEST_INDEX_G_Y  = {};\n".format(best_idx_map.get("FFT_g_y", 0)))
            f.write("const int FFT_BEST_INDEX_G_Z  = {};\n".format(best_idx_map.get("FFT_g_z", 0)))
            f.write("\n// Number of FFT best indices available\n")
            f.write("const int NUM_FFT_BEST_INDEX = 6;\n")
        
        print(f" Exported FFT best index header to {header_path}")
    
    # ===========================================================
    # APPLY BEST INDEX REPLACEMENT TO ALL DATA (full dataset)
    # ===========================================================
    if 'best_idx_map' not in locals() or not best_idx_map:
        print(f"[Stage 2 - {mode.upper()}] No valid best indices to apply. Skipping replacement.")
        return 

    # inside your loop:
    for file_idx, fname in enumerate(input_file_2):
        df_path = os.path.join(input_path_2, fname)
        if not os.path.exists(df_path):
            print(f"[Stage 2 - {mode.upper()}] Missing input file {df_path}. Skipping.")
            continue
    
        # Read only header (fast) to detect FFT columns without loading full file
        try:
            first_df = pd.read_csv(df_path, nrows=0)
        except Exception as e:
            print(f"[Stage 2 - {mode.upper()}] Failed to read header from {df_path}: {e}. Skipping.")
            continue
    
        # --- normal FFT-replacement processing for files that DO have FFT columns ---
        df = pd.read_csv(df_path)    
        n_rows = len(df)
        n_full_windows = n_rows // window_size
        if n_full_windows == 0:
            print(f"[Stage 2 - {mode.upper()}] Not enough rows for one full window in {fname}. Skipping.")
            continue
    
        for col in fft_columns:
            best_idx = int(best_idx_map[col])
            new_col = df[col].to_numpy(copy=True)
            for w in range(n_full_windows):
                val = df[col].iat[w*window_size + best_idx]
                new_col[w*window_size:(w+1)*window_size] = val
            df[col] = new_col
    
        out_file = os.path.join(output_path_2, output_file_2[file_idx])
        df.to_csv(out_file, index=False)
        print(f"[Stage 2 - {mode.upper()}] Saved processed file: {output_file_2[file_idx]} ({df.shape})")
    
    print(f"[Stage 2 - {mode.upper()}] Completed successfully.\n")
    return 

# ---------------- Data Process 3: ----------------
def stage_3(mode):
    print(f'\n======= Data Process: 3  Mode:{mode} for {window}s  =======\n')
    
    # ---------------- Import Libraries ------------------
    import os, pandas as pd, numpy as np
    # ----------------------------------------------------
    # ------------------- Sub Function -------------------
    # ----------------------------------------------------
    input_file_3, input_path_3, output_file_3, output_path_3 = get_paths(3, sampleRate, classifier_name, mode)
     
    # Collect all processed feature files from Stage 3
    feat_files = input_file_3  
    
    all_X = []
    all_y = []
    all_raw_data = []   
    all_norm_data = [] 
    raw_labels = []
    norm_labels = []
    
    for file_idx, f in enumerate(feat_files):
        #print(f'Processing file: {f}')
        df = pd.read_csv(os.path.join(input_path_3, f))
        
        # Explicit column selection for raw and normallize datasets
        df1 = df[['a_x', 'a_y', 'a_z',
                  'g_x', 'g_y', 'g_z']].astype(np.float32)
        
        df2 = df[['ag_x', 'ag_y', 'ag_z',
                  'g_x', 'g_y', 'g_z']].astype(np.float32)
        
        all_raw_data.append(df1)
        all_norm_data.append(df2)
        
        # Keeping labels aligned with row counts
        raw_labels.append(np.full(len(df1), file_idx))
        norm_labels.append(np.full(len(df2), file_idx))
        
        # Drop raw sensor columns (first 9)
        df = df.drop(columns=['t (ms)','a_x', 'a_y', 'a_z',
                              'g_x', 'g_y', 'g_z','ag_x','ag_y', 'ag_z'], errors='ignore')
        
        # Window-based aggregation
        df['window_id'] = df.index // window_size
        
        # Columns for derivative features
        derivative_cols = ['jerk_x', 'jerk_y', 'jerk_z','accl_x','accl_y','accl_z']
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
        
        if series_of_experiments == 2:
            # Series 2: good/mid/bad → 0/1/2
            fname_lower = f.lower()
            if 'good' in fname_lower:
                label_val = np.full(len(df_windowed), 0)
            elif 'mid' in fname_lower:
                label_val = np.full(len(df_windowed), 1)
            elif 'bad' in fname_lower:
                label_val = np.full(len(df_windowed), 2)
            else:
                label_val = np.full(len(df_windowed), file_idx)  # fallback
        elif series_of_experiments == 1:
            # Series 1: label based on experiment number in filename (e.g. x_1_9.71.csv → label 0)
            import re
            '''
            # Try to extract the experiment number from the filename
            match = re.search(r'_(\d+)_', f)
            if match:
                exp_number = int(match.group(1))
                label_num = exp_number - 1  # make zero-based (1→0, 2→1, etc.)
            else:
                # fallback if pattern not found
                print(f"[Warning] Could not extract experiment number from filename: {f}")
                label_num = file_idx
                '''
            # Apply one label per window (since all rows in the file belong to the same experiment)
            label_val = np.full(len(df_windowed), file_idx)
        
        all_X.append(df_windowed)
        all_y.append(label_val)  # dataset number as label
    
    # Concatenate into big DataFrames
    X_data = pd.concat(all_X, axis=0, ignore_index=True)
    y_data = pd.Series(np.concatenate(all_y), name='label')
    
    all_raw_data = pd.concat(all_raw_data, axis=0, ignore_index=True)
    all_norm_data = pd.concat(all_norm_data, axis=0, ignore_index=True)
    
    raw_y = pd.Series(np.concatenate(raw_labels), name='label')
    norm_y = pd.Series(np.concatenate(norm_labels), name='label')
    
    if mode == 'exp':
        
        # ---- Save datasets (headers, NO index) ----
        save_items = {
            # aggregated features
            f'X_data_{sampleRate}{classifier_name}.csv': X_data,
            f'y_data_{sampleRate}{classifier_name}.csv': y_data.to_frame(),   # ensure it has name 'label'
            
            # raw version
            f'all_raw_data_{sampleRate}{classifier_name}.csv': all_raw_data,
            f'y_all_raw_data_{sampleRate}{classifier_name}.csv': pd.DataFrame(raw_y, columns=['label']),
            
            # norm version
            f'all_norm_data_{sampleRate}{classifier_name}.csv': all_norm_data,
            f'y_all_norm_data_{sampleRate}{classifier_name}.csv': pd.DataFrame(norm_y, columns=['label']),
        }
            
    else:
        # ---- Save datasets (headers, NO index) ----
        save_items = {
            # aggregated features
            f'VAL_X_{sampleRate}{classifier_name}.csv': X_data,
            f'VAL_y_{sampleRate}{classifier_name}.csv': y_data.to_frame(),   # ensure it has name 'label'
            
            # raw version
            f'VAL_all_raw_data_{sampleRate}{classifier_name}.csv': all_raw_data,
            f'VAL_y_all_raw_data_{sampleRate}{classifier_name}.csv': pd.DataFrame(raw_y, columns=['label']),
            
            # norm version
            f'VAL_all_norm_data_{sampleRate}{classifier_name}.csv': all_norm_data,
            f'VAL_y_all_norm_data_{sampleRate}{classifier_name}.csv': pd.DataFrame(norm_y, columns=['label']),
        }
        
    for fname, df in save_items.items():
        path = os.path.join(output_path_3, fname)
        
        df.to_csv(path, index=False, header=True)
        print(f'Saved {fname} with shape {df.shape}')    
    return   

# ---------------- Data Process 4: ReliefF Feature Selection, Plotting and 10 best features, displaying for ESP32 use  ----------------
def stage_4():
    """
    Stage 4: ReliefF Feature Selection
    - Runs in Python by default.
    - If matlab=1, tries MATLAB Engine API, then Octave, else falls back to manual MATLAB Online.
    """
    print(f'\n======= Data Process: 4 for {window}s  =======\n')
    # ---------------- Import Libraries ------------------
    import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
    # ----------------------------------------------------
    # Define the path for saving plots
    def plot (output_path):
        plot_path = os.path.join(output_path, 'PLOTS')
        # Create output directory if it doesn't exist
        os.makedirs(plot_path, exist_ok=True)
        return plot_path
    
    def run_python_relieff():
        """Python ReliefF (skrebate)"""
        from skrebate import ReliefF
        print(" Running ReliefF in Python (skrebate)...")

        weights_file = os.path.join(output_path_4,
            f'Python_relieff_feature_indices_weights_{sampleRate}{classifier_name}{window}.csv')

        # Load train data
        X_path = os.path.join(input_path_4, input_file_4[0])
        y_path = os.path.join(input_path_4, input_file_4[1])
        X_data = pd.read_csv(X_path, header=None, skiprows=1)
        y_data = pd.read_csv(y_path, header=None, skiprows=1).squeeze("columns")

        X = X_data.to_numpy(dtype=np.float32)
        y = y_data.to_numpy(dtype=np.int32)

        relieff = ReliefF(
            n_features_to_select=10,
            n_neighbors=100,
            discrete_threshold=10,
            n_jobs=-1
        )
        relieff.fit_transform(X, y)

        weights = relieff.feature_importances_
        idx_sorted = np.argsort(weights)[::-1]
        weights_df = pd.DataFrame({
            'Feature_Index': idx_sorted,
            'ReliefF_Weight': weights[idx_sorted]
        }).round(6)
        weights_df.to_csv(weights_file, index=False, float_format='%.6f')

        return weights_file, 0  # base_index = 0 for Python

    def run_matlab_engine():
        """MATLAB Engine API"""
        try:
            import matlab.engine
            print(" Running ReliefF in MATLAB (MATLAB Engine API)...")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            eng = matlab.engine.start_matlab()
            eng.addpath(script_dir, nargout=0)  # add current folder
            eng.relieff_feature_selection(sampleRate,classifier_name,window,input_path_4,output_path_4,Xfname,yfname,nargout=0)
            eng.quit()

            weights_file = os.path.join(output_path_4,
                f'Matlab_relieff_feature_indices_weights_{sampleRate}{classifier_name}{window}.csv')
            return weights_file, 1  # base_index = 1 (MATLAB is 1-based)
        except ImportError:
            print(" MATLAB Engine for Python is not installed. Install with: pip install matlabengine")
            return None, None
        except Exception:
            print(" MATLAB Engine not available:")
            return None, None
        except:
            return None, None
        
    def check_existing_weights(output_path_4, sampleRate, classifier_name, window):
        weights_file = os.path.join(
            output_path_4,
            f'Matlab_relieff_feature_indices_weights_{sampleRate}{classifier_name}{window}.csv'
        )
        if os.path.isfile(weights_file):
            print(f"\nFound existing weights file: {weights_file}\n")
            return weights_file
        else:
            print("\nIt will take couple time\n")
            return None
        
    def wait_for_key():
        import keyboard 
        print("   MATLAB Engine is not working/available.")
        print("   Please see the type of error or run relieff_feature_selection.m manually in MATLAB Online:")
        print("   1. Go to https://matlab.mathworks.com")
        print("   2. Upload relieff_feature_selection_manual.m and training CSVs")
        print("   3. Run: relieff_feature_selection -> ('<sampleRate>')")
        print("   4. Download the generated CSV into:")
        print(f"      {output_path_4}")
        print("Press ENTER to continue or ESC to quit...")
        while True:
            if keyboard.is_pressed("enter"):
                print("Continuing...")
                return True
            elif keyboard.is_pressed("esc"):
                print("Exiting...")
                sys.exit(0)
    # ---- Step 0: Set file paths and index mode ----
    input_file_4, input_path_4, time_path, output_path_4 = get_paths(4, sampleRate, classifier_name)
    Xfname = f'X_data_{sampleRate}{classifier_name}.csv'   
    X_data = pd.read_csv(os.path.join(input_path_4, Xfname))
    yfname = f'y_data_{sampleRate}{classifier_name}.csv'
    y_data = pd.read_csv(os.path.join(input_path_4, yfname))
    VAL_X = pd.read_csv(os.path.join(input_path_4,f'VAL_X_{sampleRate}{classifier_name}.csv'))
    
    if matlab == 1:
        # Step 0: Check if file already exists

        existing_file = check_existing_weights(output_path_4, sampleRate, classifier_name, window)
        if existing_file:
            weights_file, base_index = existing_file, 1  # MATLAB uses 1-based indexing
        else:
            t1 = tic()
            weights_file, base_index = run_matlab_engine()
            t2 = tic()
            toc(t1, t2 - t1)

            if weights_file is None:  # manual fallback
                wait_for_key()
                weights_file = os.path.join(output_path_4, f'Matlab_relieff_feature_indices_weights_{sampleRate}{classifier_name}{window}.csv')
                base_index = 1
    else:
         weights_file, base_index = run_python_relieff()
        
    names = ['Python', 'Matlab']
    # ---- Step 1: Load weights ----
    weights_df = pd.read_csv(weights_file)
    print(f'Loaded ReliefF weights with shape {weights_df.shape}')
    
    # Normalize indices to 0-based
    weights_df['Feature_Index'] = weights_df['Feature_Index'] - base_index
    
    # ---- Step 2: Load features ----
   
    
    feature_names = X_data.columns.to_list()
    
    # ---- Step 3: Order features by ReliefF weight ----
    weights_sorted = weights_df.sort_values('ReliefF_Weight', ascending=False).reset_index(drop=True)
    sorted_indices = weights_sorted['Feature_Index'].to_numpy()
    
    # Reorder train/test
    X_data_sorted = X_data.iloc[:, sorted_indices]
    VAL_X_sorted = VAL_X.iloc[:, sorted_indices]
    
    # Save reordered datasets
    X_data_sorted.to_csv(os.path.join(output_path_4, f'{names[base_index]}_X_data_weight_ordered_{sampleRate}{classifier_name}.csv'), index=False, header=True)
    VAL_X_sorted.to_csv(os.path.join(output_path_4, f'{names[base_index]}_VAL_X_weight_ordered_{sampleRate}{classifier_name}.csv'), index=False, header=True)
    
    # ---- Step 4: Top-10 features ----
    top10_indices = sorted_indices[:10]
    
    # ---- Step 5: Plot ReliefF weights ----
    weights_plot = weights_df.copy().sort_values('ReliefF_Weight', ascending=False).reset_index(drop=True)
    weights_plot['Feature_Name'] = [feature_names[i] for i in weights_plot['Feature_Index']]
    
    cmap = plt.get_cmap('tab20', len(feature_names))
    fig, ax = plt.subplots(figsize=(10, 10))
    bars = ax.barh(
        np.arange(len(weights_plot)),
        weights_plot['ReliefF_Weight'],
        color=[cmap(idx) for idx in weights_plot['Feature_Index']],
        edgecolor='black'
    )
    
    # Highlight top-10
    for ix, feat_idx in enumerate(weights_plot['Feature_Index']):
        if feat_idx in top10_indices:
            bars[ix].set_edgecolor('cyan')
            bars[ix].set_linewidth(2)
            
    ax.set_yticks(np.arange(len(weights_plot)))
    ax.set_yticklabels(weights_plot['Feature_Name'] + " - " + (weights_plot['Feature_Index'] + 1).astype(str), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('ReliefF Weight')
    ax.set_ylabel('Features')
    ax.set_title(f'{names[base_index]} ReliefF Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(plot(output_path_4), f'{names[base_index]}_relieff_weights_plot_{sampleRate}{classifier_name}.png'), dpi=600)
    plt.show()
    
    # ---- Step 6: ESP32 Feature Computation Time vs ReliefF Weights ----
    times_file = os.path.join(time_path, f'feats_computation_times_{sampleRate}.csv')
    if os.path.isfile(times_file):
        times = pd.read_csv(times_file, header=None).iloc[0].to_numpy()
        times = np.log1p(times)   # safer than log
        if len(times) != len(feature_names):
            raise ValueError(f'Mismatch: {len(times)} times vs {len(feature_names)} features')    
        # --- Load weights ---
        #weights_df = pd.read_csv(weights_file)
        weights_df['Feature_Index'] = weights_df['Feature_Index'] - base_index  # 0-based
        #feature_names = X_train.columns.to_list()
        
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
        
        # Top-10 by score (shift - i so indices display 0–74)
        top10 = (df_sorted['Feature_Index'].head(10) - base_index).tolist()
        
        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 10))
        bars = ax.barh(
            np.arange(len(df_sorted)),
            df_sorted['Score'],
            color=[cmap(idx) for idx in weights_plot['Feature_Index']],
            edgecolor='black'
        )
            
        # Highlight top-10
        for ix, feat_idx in enumerate(df_sorted['Feature_Index']):
            if (feat_idx - base_index) in top10:
                bars[ix].set_edgecolor('cyan')
                bars[ix].set_linewidth(2)
                
        # Put feature names directly on y-axis
        ax.set_yticks(np.arange(len(df_sorted)))
        ax.grid(visible = True, which='both', axis='x',linestyle='--', linewidth=0.5)
        ax.set_yticklabels(df_sorted['Feature_Name']+ " - " + (df_sorted['Feature_Index'] + 1).astype(str), fontsize=9)
        
        # Flip so higher indices (bottom of df_sorted) appear at the bottom
        ax.invert_yaxis()
        
        ax.set_xlabel('Custom Score (normalized weight * max(time) / time * max(weight))')
        ax.set_ylabel('Features (sorted by Custom Score)')
        ax.set_title(f'{names[base_index]} Feature Trade-off: Importance vs ESP32 Computation Time')
        
        plot_name2 = [f'Python_Feats_CustomScore_{sampleRate}{classifier_name}.png', f'Matlab_Feats_CustomScore_{sampleRate}{classifier_name}.png']
        plot_path2 = os.path.join(plot(output_path_4), plot_name2[base_index])
        plt.tight_layout()
        plt.savefig(plot_path2, dpi=600)
        plt.show()
        plt.close()
        print(f'Saved plot: {plot_path2}')
        
        # ---- Step 7: Save ALL features reordered by custom score ----
        custom_sorted_indices = df_sorted['Feature_Index'].to_numpy()
        
        X_data_custom_sorted = X_data.iloc[:, custom_sorted_indices]
        VAL_X_custom_sorted  = VAL_X.iloc[:, custom_sorted_indices]
        
        out_file_all_train_custom = os.path.join(output_path_4, f'{names[base_index]}_X_data_custom_reordered_{sampleRate}{classifier_name}.csv')
        out_file_all_test_custom  = os.path.join(output_path_4, f'{names[base_index]}_VAL_X_custom_reordered_{sampleRate}{classifier_name}.csv')
        
        X_data_custom_sorted.to_csv(out_file_all_train_custom, index=False, header=True)
        VAL_X_custom_sorted.to_csv(out_file_all_test_custom, index=False, header=True)
        
        print(f'Saved reordered X_train (custom scores): {out_file_all_train_custom}, shape {X_data_custom_sorted.shape}')
        print(f'Saved reordered VAL_X  (custom scores): {out_file_all_test_custom}, shape {VAL_X_custom_sorted.shape}')
        
    else:
        print('Warning: feats_computation_times.csv not found, skipping ESP32 plot.')
    return 

# ---------------- Data Process 5: ReliefF Feature Selection Plotting and 10 best features, displaying for ESP32 use  ----------------
def stage_5(cl, file_index):
    print(f'\n======= Data Process: 5 for {window}s  =======\n')

    # ---------------- Import Libraries ------------------ 
    import pandas as pd, numpy as np, matplotlib.pyplot as plt, multiprocessing
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.model_selection import  RandomizedSearchCV
    from scipy.stats import randint
    from micromlgen import port
    from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay 
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    # ----------------------------------------------------

    # ---------------- Helper: loadData ------------------
    def loadData(paths, index, input_file):
        """
        Load dataset X, y, feature names, and Data_tag based on index row of input_file.
        returns: X (DataFrame or np.array), y (1D np.array of ints), fNames (list), Data_tag (str)
        """
        row = input_file.iloc[index]
        X_file, y_file, Data_tag = row['X_file'], row['y_file'], row['Data_tag']

        # First 3 datasets are from Stage 4, rest from Stage 6
        base_path = paths[0] if index < 3 else paths[1]

        X_path = os.path.join(base_path, X_file)
        y_path = os.path.join(paths[0], y_file)  # y files appear stored in paths[0] per original code

        if not os.path.exists(X_path):
            raise FileNotFoundError(f"X file not found: {X_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"y file not found: {y_path}")

        # Load X as DataFrame (preserve column names)
        X_df = pd.read_csv(X_path, dtype=np.float32)
        fNames = X_df.columns.tolist()

        # Load y
        y_df = pd.read_csv(y_path, header=None)
        # If first row is textual header, drop it
        if isinstance(y_df.iloc[0, 0], str):
            y_df = y_df.iloc[1:].reset_index(drop=True)
        y = y_df.squeeze("columns").astype(np.int32).to_numpy()
        return X_df, y, fNames, Data_tag
    # ----------------------------------------------------
    
    def capture_output_and_plot(classifier_name, accuracy, Data_tag, 
                                classifier, X_test, y_test):
        import io
        from contextlib import redirect_stdout  
        # Original class names from filenames
        
        if series_of_experiments == 1:
            original_class_names   = [
                f'x_1_0mv_{sampleRate}',f'y_1_0mv_{sampleRate}',f'z_1_0mv_{sampleRate}',
                f'x_2_r_mv_{sampleRate}',f'y_2_r_mv_{sampleRate}',f'z_2_r_mv_{sampleRate}',
                f'x_3_1st_p_min_{sampleRate}', f'y_3_1st_p_min_{sampleRate}', f'z_3_1st_p_min_{sampleRate}',
                f'x_4_2st_p_min_{sampleRate}', f'y_4_2st_p_min_{sampleRate}', f'z_4_2st_p_min_{sampleRate}',
                f'x_5_3st_p_min_w_ad_{sampleRate}', f'y_5_3st_p_min_w_ad_{sampleRate}',f'z_5_3st_p_min_w_ad_{sampleRate}']
            '''
            original_class_names   = [f'exp_1_{sampleRate}',
                                        f'exp_2_{sampleRate}',
                                        f'exp_3_{sampleRate}', 
                                        f'exp_4_{sampleRate}', 
                                        f'exp_5_{sampleRate}', 
            ]'''
        else:
            original_class_names = [f'good_{sampleRate}',f'mid_{sampleRate}',f'bad_{sampleRate}']

        accuracy = str(round(accuracy * 100, 2))
        buf = io.StringIO()
        with redirect_stdout(buf):
            # Plot confusion matrix image as an example 
            # Original class names from filenames
            
            # Generate numeric labels for display
            class_names = [f'Class {i}' for i in range(len(original_class_names))]
            
            # Create a mapping legend
            legend_text = '\n'.join([f'{class_names[i]}: {original_class_names[i]}' 
                                     for i in range(len(class_names))])
            
            labels = np.unique(y_test)  # integers that appear in y_test
            fig, ax = plt.subplots(figsize=(10, 8))  # wider & taller
            disp = ConfusionMatrixDisplay.from_estimator(
                classifier,
                X_test,
                y_test,
                labels=labels,                                      
                display_labels=[f'Class {i}' for i in labels],       
                normalize='true',
                cmap=plt.cm.Blues,
                xticks_rotation=90,
                values_format='.1%',  
                ax=ax            
                )
            
            # Add the legend as a textbox
            plt.gcf().text(1.02, 
                           0.5, 
                           legend_text, 
                           fontsize=12, 
                           va='center', 
                           bbox=dict(facecolor='white', edgecolor='black')
                           )
           
            # Title
            plt.title(f'Series-{series_of_experiments} {classifier_name} Window:{window}sec Confusion Matrix\n' + Data_tag 
                       + accuracy + '%',  fontsize=16)
            
            # Improve readability
            plt.tick_params(axis='x', labelsize=10)
            plt.tick_params(axis='y', labelsize=10)
            
            # Optional: Bold larger numbers or set font size
            for text in disp.ax_.texts:
                text.set_fontsize(6)  # Increase for better visibility (try 10–12 if need 
            # Save and show
            plt.savefig(f'{classifier_name}_image.png', dpi=600, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
            plt.close()
    
        return buf.getvalue(), '{series_of_experiments}{classifier_name}_image.png'
    
    # fix classifier_name1 depending on auto/window_search flags (these are globals in your environment)
    if ('auto' in globals() and auto == 1) and ('window_search' in globals() and window_search == 1):
        classifier_name1 = 'ALL'
    else:
        classifier_name1 = classifier_name if 'classifier_name' in globals() else ''

    paths = [
        f'./4_FEATS_COMBINED/{sampleRate}_Hz_sampling/{classifier_name1}/',
        f'./5_FEATS_SELECTION/{sampleRate}_Hz_sampling/{classifier_name1}/'
    ]

    input_file_train = pd.DataFrame([
        [f"X_data_{sampleRate}{classifier_name1}.csv", f"y_data_{sampleRate}{classifier_name1}.csv", "ALL_DATA "],
        [f"all_raw_data_{sampleRate}{classifier_name1}.csv", f"y_all_raw_data_{sampleRate}{classifier_name1}.csv", "RAW_DATA "],
        [f"all_norm_data _{sampleRate}{classifier_name1}.csv", f"y_all_norm_data_{sampleRate}{classifier_name1}.csv", "G_RAW_DATA "],
        [f"Matlab_X_data_weight_ordered_{sampleRate}{classifier_name1}.csv", f"y_data_{sampleRate}{classifier_name1}.csv", "WEIGHT BASED FEATURES "],
        [f"Matlab_X_data_custom_reordered_{sampleRate}{classifier_name1}.csv", f"y_data_{sampleRate}{classifier_name1}.csv", "SCORE BASED FEATURES "]
    ], columns=['X_file', 'y_file', 'Data_tag'])
 
    input_file_test = pd.DataFrame([
        [f"VAL_X_{sampleRate}{classifier_name1}.csv", f"VAL_y_{sampleRate}{classifier_name1}.csv", "ALL_DATA "],
        [f"VAL_all_raw_data_{sampleRate}{classifier_name1}.csv", f"VAL_y_all_raw_data_{sampleRate}{classifier_name1}.csv", "RAW_DATA "],
        [f"VAL_all_norm_data_{sampleRate}{classifier_name1}.csv", f"VAL_y_all_norm_data_{sampleRate}{classifier_name1}.csv", "G_RAW_DATA "],
        [f"Matlab_VAL_X_weight_ordered_{sampleRate}{classifier_name1}.csv", f"VAL_y_{sampleRate}{classifier_name1}.csv", "WEIGHT BASED FEATURES "],
        [f"Matlab_VAL_X_custom_reordered_{sampleRate}{classifier_name1}.csv", f"VAL_y_{sampleRate}{classifier_name1}.csv", "SCORE BASED FEATURES "]
    ], columns=['X_file', 'y_file', 'Data_tag'])
    
    X_train, y_train, fNames, Data_tag = loadData(paths, file_index, input_file_train)
    
    X_test, y_test, fNs, Data_tag = loadData(paths, file_index, input_file_test)
    
    if classifier_name == 'DT':
        classifier.set_params(criterion = 'gini', random_state = 42, splitter = 'best') 
        
    elif classifier_name == 'RF':
        classifier.set_params(criterion = 'gini', random_state = 42)
        
    if file_index in (3, 4):
        # If X_train is a DataFrame (recommended), use .iloc; otherwise handle numpy arrays:
        if hasattr(X_train, 'iloc'):
            X_train = X_train.iloc[:, :10].copy()
            X_test  = X_test.iloc[:, :10].copy()
            
        else:
            X_train = X_train[:, :10].copy()
            X_test  = X_test[:, :10].copy()
    
        # Keep feature names as a list of the first 10 names
        fNames = fNames[:10]
    
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    
    if train_size < 1:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size = train_size , test_size = test_size, shuffle = False)
        X_val, y_val = shuffle(X_val, y_val, random_state=42)
        
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    scale = 0
    if scale == 1:
        # Fit on training data and transform both train/test
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Convert scaled arrays back to DataFrames with same column names
        X_train = pd.DataFrame(X_train_scaled, columns=fNames)
        X_test = pd.DataFrame(X_test_scaled, columns=fNames)
        
        if train_size < 1:
            X_val_scaled = scaler.transform(X_val)
            X_val = pd.DataFrame(X_val_scaled, columns=fNames)

    # Parameter distributions for RandomizedSearchCV
    param_dists = {
    'DT': {
        'max_depth': randint(1,100),
        'min_samples_split': randint(2, 10),
        'max_features': randint(1,50)
        
    },
    'RF': {
        'n_estimators': randint(3, 100),
        'max_depth': randint(1,25),
        'min_samples_split': randint(2, 10),
        'max_features': randint(1,50)
    }
    }
    if file_index in (1, 2):  
        cv = TimeSeriesSplit(n_splits=2)
    else:
        cv = 5
     
    if classifier_name == 'DT':
        classifier.set_params(criterion = 'gini', random_state = 42, splitter = 'best') 
     
    elif classifier_name == 'RF':
        classifier.set_params(criterion = 'gini', random_state = 42)
     
    search = RandomizedSearchCV(classifier, 
                             param_dists[classifier_name], 
                             n_iter=1000, 
                             cv=cv, 
                             n_jobs=max(1, multiprocessing.cpu_count() - 1), 
                             scoring='accuracy', 
                             random_state=42,
                             verbose=1)
     
    # Fit and get results
    search.fit(X_train, y_train)
    #results = pd.DataFrame(search.cv_results_)
    best_params = search.best_params_
    best_cv_score = search.best_score_
     
    # Test set accuracy
    best_model = search.best_estimator_
     
    #print(best_model.feature_names_in_)
    y_pred = best_model.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
     
    y_pred = best_model.predict(X_train)
    train_score = accuracy_score(y_train, y_pred)
    summary = []
     
    # Store for summary
    summary.append({
     'Classifier': classifier_name,
     'CV Accuracy': round(best_cv_score, 4),
     'Train Accuracy': round(train_score, 4),
     'Test Accuracy': round(test_score, 4),
     'Best Params': best_params,
     })
    # Summary
    print('\n Summary:')
    for s in summary:
        print(f'\n {s['Classifier']}')
        #print(f'   CV Accuracy  : {s['CV Accuracy']}')
        print(f'   Train : {s['Train Accuracy']}')
        print(f'   Test Accuracy: {s['Test Accuracy']}')
        print(f'   Best Params  : {s['Best Params']}')
        


    if pl == 1:
        classification_text, image_path = capture_output_and_plot(classifier_name, test_score, Data_tag, best_model, X_test, y_test) 
                                                                                                                                         
    if model_create == 1:                                                                      
        import joblib
        if scale == 1:
            in_file = os.path.join('../', f"{series_of_experiments}scaler_params{classifier_name}.h")
            with open(in_file, "w") as f:
                f.write("#pragma once\n\n")
                f.write(f"const int SCALER_SIZE = {len(scaler.mean_)};\n")
                f.write("const float SCALER_MEAN[] = {" + ", ".join(map(str, scaler.mean_)) + "};\n")
                f.write("const float SCALER_SCALE[] = {" + ", ".join(map(str, scaler.scale_)) + "};\n")
                
            print("Scaler parameters exported to include/scaler_params.h")
        # Save the trained model
        model_path = f"{series_of_experiments}BEST{classifier_name}W{window}F{file_index}.pkl"
        model_bundle = best_model
        joblib.dump(model_bundle, model_path)
        print(f"Saved best model to {model_path}")  
                                                                 
        # Export model to c code
        model_code = port(best_model)
        
        # save_dir = paths[1]
        # Windows path (use raw string or double backslashes)
        save_dir = '../' 
        
        # Ensure path exists (optional safety check)
        os.makedirs(save_dir, exist_ok=True)
        
        # Full path to save header file
        save_path = os.path.join(save_dir, f'{series_of_experiments}{classifier_name}{sampleRate}W{window}.h')
        
        # Write the file
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(model_code) 

    return test_score, Data_tag
   
# ============================= Auto Runner ===================================
import pandas as pd, matplotlib.pyplot as plt
modes = ['exp', 'validate']
if auto == 0: 
    # ----------------------------- Plotting Option ---------------------------
    pl = 1 # 0: no plots; 1: plots
    file_index = [0,0]
    model_create = 1  # enable classifier model create
    for cl in range(n_classifiers):
        window = windows[cl]
        window_size = int(round(window * sampleRate))
        print(f'Window size set to {window_size} rows for sample rate {sampleRate} Hz and window time {window} sec.')
        classifier, classifier_name = get_classifier(cl)
        if stage == 0: 
            for i in range(0,2):
                stage_0(mode = modes[i])
        elif stage == 1: 
            for i in range(0,2):
                stage_1(mode = modes[i])
        elif stage == 2:
            for i in range(0,2):
                stage_2(mode = modes[i])
        elif stage == 3: 
            for i in range(0,2):
                stage_3(mode = modes[i])
        elif stage == 4: stage_4()
        elif stage == 5:     
            test_score, Data_tag = stage_5(cl, file_index[cl])
            print(f"Final {Data_tag}: {classifier_name} accuracy: {test_score*100:.2f}%")


elif auto == 1 and window_search == 0:
    model_create = 1  # enable classifier model create
    file_index = 0
    pl = 1 # 0: no plots; 1: plots
    for cl in range(n_classifiers):
        window = windows[cl]
        window_size = int(round(window * sampleRate))
        print(f'Window size set to {window_size} rows for sample rate {sampleRate} Hz and window time {window} sec.')
        classifier, classifier_name = get_classifier(cl)
        for i in range(0,2):
            stage_0(mode = modes[i])
        for i in range(0,2):
            stage_1(mode = modes[i])
        for i in range(0,2):
            stage_2(mode = modes[i])
        for i in range(0,2):
            stage_3(mode = modes[i])
        stage_4()
        #for file_index in range(0,5):
        test_score, Data_tag = stage_5(cl, file_index)
        print(f"Final {Data_tag}: {classifier_name} accuracy: {test_score*100:.2f}%")
    
elif auto == 1 and window_search == 1:
    
    model_create = 0  # set =1 if you want micromlgen export too
    pl = 0  # disable plotting during search
    # results storage
    results_df = pd.DataFrame(columns=['Classifier','Window','File_Index','Accuracy'])
    # track best window (only file_index=0 in search loop)
    best_windows = {cl: {'score': -1.0, 'window': None} for cl in range(n_classifiers)}
    
    # ---------------- Main Search (file_index = 0 only) ----------------
    for w in range(len(candidate_windows)):
        window = candidate_windows[w]
        window_size = int(round(window * sampleRate))
        print(f"\n=== Running pipeline for window={window}s ({window_size} rows) ===")
        
        # run all stages once per window
        classifier_name = 'ALL'
        for i in range(0,2):
            stage_0(mode = modes[i])
        for i in range(0,2):
            stage_1(mode = modes[i])
        for i in range(0,2):
            stage_2(mode = modes[i])
        for i in range(0,2):
            stage_3(mode = modes[i])
        stage_4()
        for cl in range(n_classifiers):
            classifier, classifier_name = get_classifier(cl)
            
            # always use file_index=0 for search
            file_index = 0
            test_score, Data_tag = stage_5(cl, file_index)            
            acc = test_score * 100
            print(f"   {Data_tag}: {classifier_name} accuracy: {acc:.2f}%")
            
            # store results
            results_df.loc[len(results_df)] = [classifier_name, window, file_index, acc]
            
            # update best window
            if acc > best_windows[cl]['score']:
                best_windows[cl]['score'] = acc
                best_windows[cl]['window'] = window

    print("\n================= Window search finished =================")
    for cl in range(n_classifiers):
        clf_name = get_classifier(cl)[1]
        best = best_windows[cl]
        print(f"Best window for {clf_name}: {best['window']}s "
              f"(accuracy={best['score']:.2f}% with file_index=0)")
    
    # ---------------- Plot search results ----------------
    plt.figure(figsize=(10,6))
    colors = plt.cm.tab10.colors

    for cl in range(n_classifiers):
        clf_name = get_classifier(cl)[1]
        subset = results_df[results_df['Classifier']==clf_name]
        grouped = subset.groupby('Window')['Accuracy'].mean().reset_index()
        color = colors[cl % len(colors)]
        
        plt.plot(grouped['Window'], grouped['Accuracy'], marker='o', color=color, label=clf_name)
        
        # mark best window (from search with file_index=0)
        best = best_windows[cl]
        plt.scatter(best['window'], best['score'], marker='*', s=200, 
                    color=color, edgecolor='black', zorder=5)

    plt.xlabel("Window (s)")
    plt.ylabel("Accuracy (%)")         
    plt.title("Classifier Accuracy vs Window Size (file_index=0)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results_window_search.png", dpi=600)
    plt.show()

    # ---------------- Re-run with best window & best file_index ----------------
    pl = 1  # enable plots
    model_create = 0  # set =1 if you want micromlgen export too

    best_final_indices = {}  # store best file_index per classifier

    for cl in range(n_classifiers):
        clf_name = get_classifier(cl)[1]
        best_window = best_windows[cl]['window']
        
        best_final = {'score': -1.0, 'file_index': None}
        window = best_window
        window_size = int(round(window * sampleRate))
        
        print(f"\nTesting {clf_name} with window={window}s and file_index={file_index}")
        classifier, classifier_name = get_classifier(cl)
        
        for i in range(0,2):
            stage_0(mode = modes[i])
        for i in range(0,2):
            stage_1(mode = modes[i])
        for i in range(0,2):
            stage_2(mode = modes[i])
        for i in range(0,2):
            stage_3(mode = modes[i])
        stage_4()
        
        for file_index in range(3,5):
            test_score, Data_tag = stage_5(cl, file_index)
            acc = round(test_score * 100, 2)
            print(f"   {Data_tag}: {clf_name} accuracy={acc:.2f}%")

            # log each re-run
            results_df.loc[len(results_df)] = [clf_name, window, str(file_index), acc]

            if acc > best_final['score']:
                best_final['score'] = acc
                best_final['file_index'] = file_index

        # final best combination
        print(f"\n>>> Final BEST for {clf_name}: "
              f"window={best_window}s, file_index={best_final['file_index']}, "
              f"accuracy={best_final['score']:.2f}%")

        # save best file_index for use in final run
        best_final_indices[cl] = best_final['file_index']

        # store final best row with actual file_index (not a string tag)
        results_df.loc[len(results_df)] = [clf_name, best_window, str(best_final['file_index']), best_final['score']]
        
    # ---------------- Final run with best (window, file_index) ----------------
    print("\n================= Final Run with Best Parameters =================")
    pl = 1   # force plotting for final best run
    model_create = 1  # set to 1 if you also want micromlgen C export

    for cl in range(n_classifiers):
        clf_name = get_classifier(cl)[1]
        best_window = best_windows[cl]['window']
        file_index = best_final_indices[cl]   # directly from re-run

        print(f"\n>>> Last run for {clf_name} with window={best_window}s and file_index={file_index}")
        classifier, classifier_name = get_classifier(cl)
        
        for i in range(0,2):
            stage_0(mode = modes[i])
        for i in range(0,2):
            stage_1(mode = modes[i])
        for i in range(0,2):
            stage_2(mode = modes[i])
        for i in range(0,2):
            stage_3(mode = modes[i])
        stage_4()
        test_score, Data_tag = stage_5(cl, file_index)
        acc = round(test_score * 100, 2)
        print(f"Final run {Data_tag}: {clf_name} accuracy={acc:.2f}%")

        # add explicit last run row
        results_df.loc[len(results_df)] = [clf_name, best_window, Data_tag, acc]

    # Save updated results
    results_df.to_csv("all_results_classifier_windows.csv", index=False)
    print("\nSaved updated results (with last run) to all_results_classifier_windows.csv")
