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
series_of_experiments = 1 #
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
windows = [2,2] # sec  IF window_search = 0 <-- Change this table to set time window per classifier
# ----------------------------- Window Search Value -----------------------------------
window_search = 0 # Set 1 to search for the best time window from a list of time window named candidate_windows
candidate_windows = [4,6,8,10,20,40,60,80,100,120,140,160,180,200] # in sec (1s - 3min, 20s)
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

# ----------------------------- File Configuration ----------------------------

def get_paths(stage, sampleRate, classifier_name):
    base = f"{sampleRate}_Hz_sampling/{classifier_name}"
    if series_of_experiments == 1:
        exp_files = [
            f'x_1_0mv_{sampleRate}.csv',f'y_1_0mv_{sampleRate}.csv',f'z_1_0mv_{sampleRate}.csv',
            f'x_2_r_mv_{sampleRate}.csv',f'y_2_r_mv_{sampleRate}.csv', f'z_2_r_mv_{sampleRate}.csv',
            f'x_3_1st_p_min_{sampleRate}.csv', f'y_3_1st_p_min_{sampleRate}.csv', f'z_3_1st_p_min_{sampleRate}.csv',
            f'x_4_2st_p_min_{sampleRate}.csv', f'y_4_2st_p_min_{sampleRate}.csv', f'z_4_2st_p_min_{sampleRate}.csv',
            f'x_5_3st_p_min_w_ad_{sampleRate}.csv', f'y_5_3st_p_min_w_ad_{sampleRate}.csv',f'z_5_3st_p_min_w_ad_{sampleRate}.csv'
        ]
        test_files =  [f'test_x_1_{sampleRate}.csv', f'test_y_1_{sampleRate}.csv', f'test_z_1_{sampleRate}.csv',
                       f'test_x_2_{sampleRate}.csv', f'test_y_2_{sampleRate}.csv', f'test_z_2_{sampleRate}.csv',
                       f'test_x_3_{sampleRate}.csv', f'test_y_3_{sampleRate}.csv', f'test_z_3_{sampleRate}.csv',
                       f'test_x_4_{sampleRate}.csv', f'test_y_4_{sampleRate}.csv', f'test_z_4_{sampleRate}.csv',
                       f'test_x_5_{sampleRate}.csv', f'test_y_5_{sampleRate}.csv', f'test_z_5_{sampleRate}.csv'  
                      ]
        
        exp_path = f'./0_RAW/series_of_experiments_1/{sampleRate}_Hz_sampling/'
    elif series_of_experiments == 2:
        exp_files = []
        exp_path = f'./0_RAW/series_of_experiments_2/{sampleRate}_Hz_sampling/'
    if stage == 0:
        in_files = exp_files
        in_path =  exp_path
        out_path = f'./1_CLEAN/{base}/'
        out_files = [f"{fname.replace('.csv', f'{classifier_name}_clean.csv')}"
                 for fname in exp_files]
    elif stage == 1:
        in_files = [f"{fname.replace('.csv', f'{classifier_name}_clean.csv')}"
                 for fname in exp_files]
        in_path = f'./1_CLEAN/{base}/'
        out_path = f'./2_FEATS_PREPROCESSSED/{base}/'
        out_files = [f"{fname.replace('.csv', f'{classifier_name}_feat_prepr.csv')}"
                 for fname in exp_files]
    elif stage == 2: 
        in_files = [f"{fname.replace('.csv', f'{classifier_name}_feat_prepr.csv')}"
                 for fname in exp_files]
        in_path =  f'./2_FEATS_PREPROCESSSED/{base}/'
        out_path = f'./3_FEATS/{base}/'
        out_files = [f"{fname.replace('.csv', f'{classifier_name}_feat.csv')}"
                 for fname in exp_files]
    elif stage == 3:
        in_files = [f"{fname.replace('.csv', f'{classifier_name}_feat.csv')}"
                 for fname in exp_files]
        in_path = f'./3_FEATS/{base}/'
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
def stage_0():
    print(f'\n ======= Data Process: 0 for {window}s =======\n')
    # ---------------- Import Libraries ------------------
    import os, pandas as pd, matplotlib.pyplot as plt, csv
    # ----------------------------------------------------
    
    # Define the path for saving plots
    def plot (output_path):
        plot_path = os.path.join(output_path, 'PLOTS')
        # Create output directory if it doesn't exist
        os.makedirs(plot_path, exist_ok=True)
        return plot_path
    
    input_file_0, input_path_0, output_file_0, output_path_0 = get_paths(0, sampleRate, classifier_name)
    r'''
    # ---- Check input files before processing ----
    for f in input_file_0:
        path = os.path.join(input_path_0, f)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'No _raw.csv files found in {input_path_0}. Cannot build clean.csv. Please provide the raw data files.')
            '''
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
        datasets.append(df)
        row_counts.append(len(df))
        
    # ================= Step 2: Find min rows (multiple of window) =================
    
    min_rows = min(row_counts)
    adjusted_rows = (min_rows // window_size) * window_size  # largest multiple of window <= min_rows
    
    if adjusted_rows != min_rows:
        print(f'\nMinimum rows {min_rows} adjusted down to {adjusted_rows} to fit window size {window_size}.\n')
    else:
        print(f'\nMinimum rows {min_rows} fits perfectly into windows of size {window_size}.\n')
    
    # ================= Step 3: Trim and Save clean dataset to disk =================
    
    for file_idx, df in enumerate(datasets):
        trimmed_df = df.iloc[:adjusted_rows].copy()
        out_file = os.path.join(output_path_0, output_file_0[file_idx])
        trimmed_df.to_csv(out_file, index=False)
        # Print output file name
        print(f'Saved trimmed file: {output_file_0[file_idx]} with {len(trimmed_df)} rows.')
    print('\nAll datasets aligned and trimmed successfully.\n')
    
    # ================= Step 4: Plotting all datasets for inspection =================
    
    if pl == 0:     
        # Load all datasets
        datasets = {fname: pd.read_csv(output_path_0 + fname) for fname in output_file_0}
        
        # -------------------------------
        # Axis-specific plots
        # -------------------------------
        axis_labels = {'x': 'X Axis', 'y': 'Y Axis', 'z': 'Z Axis'}
        step_conditions = {
            0: f'1_0mv_{sampleRate}{classifier_name}_clean.csv',
            1: f'2_r_mv_{sampleRate}{classifier_name}_clean.csv',   
            2: f'3_1st_p_min_{sampleRate}{classifier_name}_clean.csv',
            3: f'4_2st_p_min_{sampleRate}{classifier_name}_clean.csv',
            4: f'5_3st_p_min_w_ad_{sampleRate}{classifier_name}_clean.csv'
        }
        
        for axis, axis_name in axis_labels.items():
            fig, axs = plt.subplots(len(step_conditions), 2, figsize=(12, 14), sharex=True)
            
            for row, (step, suffix) in enumerate(step_conditions.items()):
                fname = f'{axis}_{suffix}'
                df = datasets[fname]
                time = (df['t (ms)'] - df['t (ms)'].min()) / 60000.0
        
                # Left subplot = Accelerometer (a_x, a_y, a_z)
                axs[row, 0].plot(time, df['a_x'], color='red', label='Acc X')
                axs[row, 0].plot(time, df['a_y'], color='green', label='Acc Y')
                axs[row, 0].plot(time, df['a_z'], color='blue', label='Acc Z')
                if step == 0 or 1:
                    axs[row, 0].set_title(f'{axis_name} - {step} step/min with random movements - Accelerometer', fontsize=11)
                elif step == 4:
                    axs[row, 0].set_title(f'{axis_name} - {step} step/min with anomaly - Accelerometer', fontsize=11)
                else:
                    axs[row, 0].set_title(f'{axis_name} - {step} step/min - Accelerometer', fontsize=11)
                axs[row, 0].set_ylabel('Acceleration (m/s$^2$)', fontsize=10)
                axs[row, 0].legend(fontsize=8)
        
                # Right subplot = Gyroscope (g_x, g_y, g_z)
                axs[row, 1].plot(time, df['g_x'], color='red', label='Gyro X')
                axs[row, 1].plot(time, df['g_y'], color='green', label='Gyro Y')
                axs[row, 1].plot(time, df['g_z'], color='blue', label='Gyro Z')
                if step == 0:
                    axs[row, 1].set_title(f'{axis_name} - {step} step/min with random movements - Gyroscope', fontsize=11)
                elif step == 3:
                    axs[row, 1].set_title(f'{axis_name} - {step} step/min with anomaly - Gyroscope', fontsize=11)
                else:
                    axs[row, 1].set_title(f'{axis_name} - {step} step/min - Gyroscope', fontsize=11)
                axs[row, 1].set_ylabel('Angular Velocity (rad/s)', fontsize=10)
                axs[row, 1].legend(fontsize=8)
        
                # X-label only at bottom row
                if row == len(step_conditions) - 1:
                    axs[row, 0].set_xlabel('Time (min)', fontsize=10)
                    axs[row, 1].set_xlabel('Time (min)', fontsize=10)
        
            fig.suptitle(f'{axis_name} - Step Conditions', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            plot_name = f'{axis}_axis_plot.png'
            plot_path_all = os.path.join(plot(output_path_0), plot_name)
            plt.savefig(plot_path_all, dpi=600)
            plt.show()
            
    return 

# ---------------- Data Process 1: Feature Extraction ----------------
def stage_1():
    print(f'\n======= Data Process: 1 for {window}s  =======\n')
    t1=tic()
    # ---------------- Import Libraries ------------------
    import os, pandas as pd, numpy as np, multiprocessing
    from scipy.fft import fft
    from scipy.stats import iqr
    from joblib import Parallel, delayed
    # ----------------------------------------------------
    input_file_1, input_path_1, output_file_1, output_path_1 = get_paths(1, sampleRate, classifier_name)
    
    def process_file(file_idx, fname):  
        df = pd.read_csv(input_path_1 + input_file_1[file_idx])
        # print(f'Processing file: {input_file_1[file_idx]}')
        
        # Normalize acceleration to acceleration in g (9.80665 m/s^2)
        #--------------------------------------------------------------------
        df[['ag_x','ag_y','ag_z']] = (df[['a_x','a_y','a_z']] / 9.80665).round(3)
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
def stage_2():
    print(f'\n======= Data Process: 2 for {window}s  =======\n')
    # ---------------- Import Libraries ------------------
    import os, pandas as pd, numpy as np
    # ----------------------------------------------------
    
    input_file_2, input_path_2, output_file_2, output_path_2 = get_paths(2, sampleRate, classifier_name)
    
     # ================= Step 1: Find STDs in FFTs features =================
    fft_std_path = os.path.join(input_path_2, 'fft_std.csv')
    # ------- _feat_prepr.csv' Checker --------
    feat_files = sorted([f for f in os.listdir(input_path_2) if f.endswith('_feat_prepr.csv')])
    if not feat_files:
        raise FileNotFoundError(f'No _feat_prepr.csv files found in {input_path_2}. Cannot build fft_std.csv.')
    
    # reading first file header to discover FFT columns
    first_df = pd.read_csv(os.path.join(input_path_2, feat_files[0]), nrows=0)
    fft_columns = [c for c in first_df.columns if c.startswith('FFT_')]
    if not fft_columns:
        raise ValueError('No FFT_ columns found in feature files.')
        
    # reading only FFT columns from all files
    dfs = []
    for f in feat_files:
        p = os.path.join(input_path_2, f)
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
    n_full_windows = n_rows // window_size
    if n_full_windows != n_rows / window_size:
        raise ValueError('window size is larger than the number of rows; cannot process.')
    print(f'Using window = {window_size}; number of full windows = {n_full_windows}; skipping tail if incomplete.')
    
    # Prepare replacement: for each FFT column compute the position-score from fft_std
    best_idx_map = {}
    for col in fft_columns:
        scores = np.zeros(window_size, dtype=int)
        for w in range(n_full_windows):
            chunk = fft_std_df[col].iloc[w*window_size:(w+1)*window_size].values
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
    for file_idx in range(len(input_file_2)): 
        df = pd.read_csv(input_path_2 + input_file_2[file_idx])
        #print('=' * 70 + '\n'+ f'Processing file: {input_file_2[file_idx]}\n'+'=' * 70)
        
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
                val = df[col].iat[w*window_size+ best_idx]
                new_col[w*window_size:(w+1)*window_size] = val
                
            # If there is an incomplete tail (n_rows % window != 0), we leave those rows unchanged
            df[col] = new_col # Replace df column with new values
            
        # ---- Save aggregated dataset ----
        out_file = os.path.join(output_path_2, output_file_2[file_idx])
        df = df.drop(columns=['t (ms)'],errors='ignore')
        df.to_csv(out_file, index=False)
        # Print output file name
        print(f'Saved aggregated file: {output_file_2[file_idx]} with shape {df.shape}')
    return 

# ---------------- Data Process 3: ----------------
def stage_3():
    print(f'\n======= Data Process: 3 for {window}s  =======\n')
    
    # ---------------- Import Libraries ------------------
    import os, pandas as pd, numpy as np
    # ----------------------------------------------------
    # ------------------- Sub Function -------------------
    def TrainTestSplit(X, y, train_size=0.75, test_size=0.25):
        '''
        Deterministic split per class: keeps order, no shuffle.
        Train = first part of rows
        Test  = last part of rows
        '''
        # Ensure y is a Series with same index
        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=X.index, name='label')
        elif isinstance(y, pd.DataFrame):
            # flatten single-column DataFrame to Series
            y = y.iloc[:, 0]
            
        X_train_parts, X_test_parts = [], []
        y_train_parts, y_test_parts = [], []
        
        for class_label in y.unique():
            class_mask = (y == class_label)
            class_data = X.loc[class_mask]
            class_labels = y.loc[class_mask]
            
            n_total = len(class_data)
            n_train = int(n_total * train_size)
            
            # Deterministic split (no shuffle)
            X_train_class = class_data.iloc[:n_train]
            X_test_class  = class_data.iloc[n_train:]
            y_train_class = class_labels.iloc[:n_train]
            y_test_class  = class_labels.iloc[n_train:]
            
            # Collect
            X_train_parts.append(X_train_class)
            X_test_parts.append(X_test_class)
            y_train_parts.append(y_train_class)
            y_test_parts.append(y_test_class)
            
        # Concatenate back
        X_train = pd.concat(X_train_parts, axis=0, ignore_index=True)
        X_test  = pd.concat(X_test_parts, axis=0, ignore_index=True)
        y_train = pd.concat(y_train_parts, axis=0, ignore_index=True)
        y_test  = pd.concat(y_test_parts, axis=0, ignore_index=True)
        
        # Safeguard: total lengths must match input
        assert len(X_train) + len(X_test) == len(X), \
            f'Split mismatch: {len(X)} rows in, {len(X_train)+len(X_test)} out'
        assert len(y_train) + len(y_test) == len(y), \
            f'Label mismatch: {len(y)} rows in, {len(y_train)+len(y_test)} out'
            
        return X_train, X_test, y_train, y_test
    # ----------------------------------------------------
    input_file_3, input_path_3, output_file_3, output_path_3 = get_paths(3, sampleRate, classifier_name)
     
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
        
        all_X.append(df_windowed)
        all_y.append(np.full(len(df_windowed), file_idx))  # dataset number as label
        
    # Concatenate into big DataFrames
    X_data = pd.concat(all_X, axis=0, ignore_index=True)
    y_data = pd.Series(np.concatenate(all_y), name='label')
    
    all_raw_data = pd.concat(all_raw_data, axis=0, ignore_index=True)
    all_norm_data = pd.concat(all_norm_data, axis=0, ignore_index=True)
    
    raw_y = np.concatenate(raw_labels)
    norm_y = np.concatenate(norm_labels)
    
    raw_y = pd.Series(np.concatenate(raw_labels), name='label')
    norm_y = pd.Series(np.concatenate(norm_labels), name='label')
    
    # Train - Test split before feature selection for models accuracy evaluation
    train_size = 0.50
    test_size = 0.50
    #from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size = 0.75, test_size = 0.25)
    X_train, X_test, y_train, y_test = TrainTestSplit(X_data, y_data, train_size, test_size)
    
    raw_train, raw_test, y_raw_train, y_raw_test = TrainTestSplit(all_raw_data, raw_y, train_size, test_size)
    norm_train, norm_test, y_norm_train, y_norm_test = TrainTestSplit(all_norm_data, norm_y, train_size, test_size)
    
    # ---- Save datasets (headers, NO index) ----
    save_items = {
        # aggregated features
        f'X_data_{sampleRate}{classifier_name}.csv': X_data,
        f'y_data_{sampleRate}{classifier_name}.csv': y_data.to_frame(),   # ensure it has name 'label'
        f'X_train_{sampleRate}{classifier_name}.csv': pd.DataFrame(X_train, columns=X_data.columns, dtype=np.float32),
        f'y_train_{sampleRate}{classifier_name}.csv': pd.DataFrame(y_train, columns=['label']),
        f'X_test_{sampleRate}{classifier_name}.csv': pd.DataFrame(X_test, columns=X_data.columns, dtype=np.float32),
        f'y_test_{sampleRate}{classifier_name}.csv': pd.DataFrame(y_test, columns=['label']),
        
        # raw version
        f'all_raw_data_{sampleRate}{classifier_name}.csv': all_raw_data,
        f'y_all_raw_data_{sampleRate}{classifier_name}.csv': pd.DataFrame(raw_y, columns=['label']),
        f'all_raw_train_{sampleRate}{classifier_name}.csv': pd.DataFrame(raw_train, columns=all_raw_data.columns, dtype=np.float32),
        f'y_all_raw_train_{sampleRate}{classifier_name}.csv': pd.DataFrame(y_raw_train, columns=['label']),
        f'all_raw_test_{sampleRate}{classifier_name}.csv': pd.DataFrame(raw_test, columns=all_raw_data.columns, dtype=np.float32),
        f'y_all_raw_test_{sampleRate}{classifier_name}.csv': pd.DataFrame(y_raw_test, columns=['label']),
        
        # norm version
        f'all_norm_data_{sampleRate}{classifier_name}.csv': all_norm_data,
        f'y_all_norm_data_{sampleRate}{classifier_name}.csv': pd.DataFrame(norm_y, columns=['label']),
        f'all_norm_train_{sampleRate}{classifier_name}.csv': pd.DataFrame(norm_train, columns=all_norm_data.columns, dtype=np.float32),
        f'y_all_norm_train_{sampleRate}{classifier_name}.csv': pd.DataFrame(y_norm_train, columns=['label']),
        f'all_norm_test_{sampleRate}{classifier_name}.csv': pd.DataFrame(norm_test, columns=all_norm_data.columns, dtype=np.float32),
        f'y_all_norm_test_{sampleRate}{classifier_name}.csv': pd.DataFrame(y_norm_test, columns=['label']),
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
            eng.relieff_feature_selection(sampleRate,classifier_name,window, nargout=0)
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
    X_train = pd.read_csv(os.path.join(input_path_4, f'X_train_{sampleRate}{classifier_name}.csv'))
    X_test  = pd.read_csv(os.path.join(input_path_4, f'X_test_{sampleRate}{classifier_name}.csv'))
    feature_names = X_train.columns.to_list()
    
    # ---- Step 3: Order features by ReliefF weight ----
    weights_sorted = weights_df.sort_values('ReliefF_Weight', ascending=False).reset_index(drop=True)
    sorted_indices = weights_sorted['Feature_Index'].to_numpy()
    
    # Reorder train/test
    X_train_sorted = X_train.iloc[:, sorted_indices]
    X_test_sorted  = X_test.iloc[:, sorted_indices]
    
    # Save reordered datasets
    X_train_sorted.to_csv(os.path.join(output_path_4, f'{names[base_index]}_X_train_weight_ordered_{sampleRate}{classifier_name}.csv'), index=False, header=True)
    X_test_sorted.to_csv(os.path.join(output_path_4, f'{names[base_index]}_X_test_weight_ordered_{sampleRate}{classifier_name}.csv'), index=False, header=True)
    
    # ---- Step 4: Top-10 features ----
    top10_indices = sorted_indices[:10]
    
    # ---- Step 5: Plot ReliefF weights ----
    weights_plot = weights_df.copy().sort_values('ReliefF_Weight', ascending=False).reset_index(drop=True)
    weights_plot['Feature_Name'] = [feature_names[i] for i in weights_plot['Feature_Index']]
    
    cmap = plt.cm.get_cmap('tab20', len(feature_names))
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
        
        X_train_custom_sorted = X_train.iloc[:, custom_sorted_indices]
        X_test_custom_sorted  = X_test.iloc[:, custom_sorted_indices]
        
        out_file_all_train_custom = os.path.join(output_path_4, f'{names[base_index]}_X_train_custom_reordered_{sampleRate}{classifier_name}.csv')
        out_file_all_test_custom  = os.path.join(output_path_4, f'{names[base_index]}_X_test_custom_reordered_{sampleRate}{classifier_name}.csv')
        
        X_train_custom_sorted.to_csv(out_file_all_train_custom, index=False, header=True)
        X_test_custom_sorted.to_csv(out_file_all_test_custom, index=False, header=True)
        
        print(f'Saved reordered X_train (custom scores): {out_file_all_train_custom}, shape {X_train_custom_sorted.shape}')
        print(f'Saved reordered X_test  (custom scores): {out_file_all_test_custom}, shape {X_test_custom_sorted.shape}')
        
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
    #import m2cgen as m2c
    # ----------------------------------------------------


    def loadData(paths, index, input_file):
        r'''
        Load dataset X, y, feature names, and tag based on index from input_file.
        
        Parameters
        ----------
        paths : list
            [stage4_path, stage6_path]
        index : int
            Row index of input_file
        
        Returns
        -------
        X : pd.DataFrame
            Feature matrix (keeps column names)
        y : pd.Series
            Labels
        fNames : list
            Feature names
        Data_tag : str
            Tag describing the dataset
        '''
    
        row = input_file.iloc[index]
        X_file, y_file, Data_tag = row['X_file'], row['y_file'], row['Data_tag']
    
        # First 3 datasets are from Stage 4, rest from Stage 6
        base_path = paths[0] if index < 3 else paths[1]
    
        X_path = os.path.join(base_path, X_file)
        y_path = os.path.join(paths[0], y_file)
    
        # ---- Load X as DataFrame ----
        X_data = pd.read_csv(X_path, dtype=np.float32)
        fNames = X_data.columns.tolist()
        X = X_data
    
        # ---- Load y as Series ----
        y_data = pd.read_csv(y_path, header=None)
    
        # If first row is text (e.g., 'label'), drop it
        if isinstance(y_data.iloc[0, 0], str):
            y_data = y_data.iloc[1:]
    
        y = y_data.squeeze("columns").astype(np.int32)
    
        return X, y, fNames, Data_tag
    
    def capture_output_and_plot(classifier_name, accuracy, Data_tag, 
                                classifier, X_test, y_test):
        import io
        from contextlib import redirect_stdout  
        # Original class names from filenames
        original_class_names   = [
            f'x_1_0mv_{sampleRate}',f'y_1_0mv_{sampleRate}',f'z_1_0mv_{sampleRate}',
            f'x_2_r_mv_{sampleRate}',f'y_2_r_mv_{sampleRate}',f'z_2_r_mv_{sampleRate}',
            f'x_3_1st_p_min_{sampleRate}', f'y_3_1st_p_min_{sampleRate}', f'z_3_1st_p_min_{sampleRate}',
            f'x_4_2st_p_min_{sampleRate}', f'y_4_2st_p_min_{sampleRate}', f'z_4_2st_p_min_{sampleRate}',
            f'x_5_3st_p_min_w_ad_{sampleRate}', f'y_5_3st_p_min_w_ad_{sampleRate}',f'z_5_3st_p_min_w_ad_{sampleRate}'
        ]
        
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
            
            # Remove scientific notation    
            #plt.gca().set_xticklabels(class_names, rotation=90)
            #plt.gca().set_yticklabels(class_names)
            
            # Add the legend as a textbox
            plt.gcf().text(1.02, 
                           0.5, 
                           legend_text, 
                           fontsize=12, 
                           va='center', 
                           bbox=dict(facecolor='white', edgecolor='black')
                           )
           
            # Title
            plt.title(f'{classifier_name} Confusion Matrix \n' + Data_tag 
                       + accuracy + '%', fontsize=16)
            
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
    
        return buf.getvalue(), '{classifier_name}_image.png'
    # ========================================================================
    
                                 
    if auto == 1 and window_search == 1:
        classifier_name1 = 'ALL'
    else:
        classifier_name1 = classifier_name
    
    
    paths = [f'./4_FEATS_COMBINED/{sampleRate}_Hz_sampling/{classifier_name1}/',f'./5_FEATS_SELECTION/{sampleRate}_Hz_sampling/{classifier_name1}/']
    
    input_file_train = pd.DataFrame([
        [f"X_train_{sampleRate}{classifier_name1}.csv", f"y_train_{sampleRate}{classifier_name1}.csv", "ALL_DATA "],
        [f"all_raw_train_{sampleRate}{classifier_name1}.csv", f"y_all_raw_train_{sampleRate}{classifier_name1}.csv", "RAW_DATA "],
        [f"all_norm_train _{sampleRate}{classifier_name1}.csv", f"y_all_norm_train_{sampleRate}{classifier_name1}.csv", "G_RAW_DATA "],
        [f"Matlab_X_train_weight_ordered_{sampleRate}{classifier_name1}.csv", f"y_train_{sampleRate}{classifier_name1}.csv", "WEIGHT BASED FEATURES "],
        [f"Matlab_X_train_custom_reordered_{sampleRate}{classifier_name1}.csv",f"y_train_{sampleRate}{classifier_name1}.csv", "SCORE BASED FEATURES "]
            ], columns=['X_file', 'y_file', 'Data_tag'])
        
    input_file_test = pd.DataFrame([
        [f"X_data.csv", f"y_data.csv", "ALL_DATA "],
        [f"all_raw_test_{sampleRate}{classifier_name1}.csv", f"y_all_raw_test_{sampleRate}{classifier_name1}.csv", "RAW_DATA "],
        [f"all_norm_test_{sampleRate}{classifier_name1}.csv", f"y_all_norm_test_{sampleRate}{classifier_name1}.csv", "G_RAW_DATA "],
        [f"Matlab_X_test_weight_ordered_{sampleRate}{classifier_name1}.csv", f"y_test_{sampleRate}{classifier_name1}.csv", "WEIGHT BASED FEATURES "],
        [f"Matlab_X_test_custom_reordered_{sampleRate}{classifier_name1}.csv", f"y_test_{sampleRate}{classifier_name1}.csv", "SCORE BASED FEATURES "]
            ], columns=['X_file', 'y_file', 'Data_tag'])
    
    
        
    X_train, y_train, fNames, Data_tag = loadData(paths, file_index, input_file_train)
    X_test, y_test, fNs, Data_tag = loadData(paths, file_index, input_file_test)
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
    
    
    # Parameter distributions for RandomizedSearchCV
    param_dists = {
        'DT': {
            'max_depth': randint(1,25),
            'min_samples_split': randint(2, 10),
            #'max_features': randint(1,2)
            
        },
        'RF': {
            'n_estimators': randint(100, 500),
            'max_depth': randint(1,25),
            'min_samples_split': randint(2, 10),
            #'max_features': randint(1,2)
        }
    }

    ts_cv = TimeSeriesSplit(n_splits=2) 
    if classifier_name == 'DT':
        classifier.set_params(criterion = 'gini', random_state = 42, splitter = 'random') 
        
    elif classifier_name == 'RF':
        classifier.set_params(criterion = 'gini', random_state = 42)
    from sklearn.metrics import make_scorer
    
    search = RandomizedSearchCV(classifier, 
                                param_dists[classifier_name], 
                                n_iter=1000, 
                                cv=ts_cv, 
                                n_jobs=max(1, multiprocessing.cpu_count() - 1), 
                                scoring='accuracy', 
                                random_state=42,
                                verbose=1)
    
    # Fit and get results
    search.fit(X_train, y_train)
    results = pd.DataFrame(search.cv_results_)
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
        print(f'   CV Accuracy  : {s['CV Accuracy']}')
        print(f'   Train : {s['Train Accuracy']}')
        print(f'   Test Accuracy: {s['Test Accuracy']}')
        print(f'   Best Params  : {s['Best Params']}')

    if pl == 1:
        classification_text, image_path = capture_output_and_plot(classifier_name,
                                                                      test_score, 
                                                                          Data_tag,  
                                                                          best_model, 
                                                                          X_test,
                                                                          y_test) 
    if model_create == 1:                                                                      
        import joblib

        # Save the trained model
        model_path = f"BEST{classifier_name}W{window}F{file_index}.pkl"
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
        save_path = os.path.join(save_dir, f'{classifier_name}{sampleRate}W{window}.h')
        
        # Write the file
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(model_code)   
                                
    return test_score, Data_tag
    
# ============================= Auto Runner ===================================
import pandas as pd, matplotlib.pyplot as plt

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
        if stage == 0: stage_0()
        elif stage == 1: stage_1()
        elif stage == 2: stage_2()
        elif stage == 3: stage_3()
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
        stage_0()
        stage_1()
        stage_2()
        stage_3()
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
        stage_0()
        stage_1()
        stage_2()
        stage_3()
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
        
        stage_0()
        stage_1()
        stage_2()
        stage_3()
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
        
        stage_0()
        stage_1()
        stage_2()
        stage_3()
        stage_4()
        test_score, Data_tag = stage_5(cl, file_index)
        acc = round(test_score * 100, 2)
        print(f"Final run {Data_tag}: {clf_name} accuracy={acc:.2f}%")

        # add explicit last run row
        results_df.loc[len(results_df)] = [clf_name, best_window, Data_tag, acc]

    # Save updated results
    results_df.to_csv("all_results_classifier_windows.csv", index=False)
    print("\nSaved updated results (with last run) to all_results_classifier_windows.csv")


