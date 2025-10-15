# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 09:28:25 2025

@author: AXILLIOS
"""
import joblib
import os
import pandas as pd
import csv
from joblib import Parallel, delayed
import multiprocessing

classifier_names = ['DT','RF']
file_indexs = [0,0]
for cl in range(0,2):
    file_index = file_indexs[cl]
    windows = [2,2]
    sampleRate = 9.71
    classifier_name = classifier_names[cl]
    window = windows[cl]
    window_size = int(round(window * sampleRate))

    print(f'\n======= Data Process: 0 for {window}s  =======\n')
    exp_files = ['try_x_1_9.71.csv', 'try_y_1_9.71.csv','try_z_1_9.71.csv',
                 'try_x_2_9.71.csv','try_y_2_9.71.csv','try_z_2_9.71.csv',
                 'try_x_3_9.71.csv','try_y_3_9.71.csv','try_z_3_9.71.csv',
                 'try_x_4_9.71.csv','try_y_4_9.71.csv','try_z_4_9.71.csv',
                 'try_x_5_9.71.csv','try_y_5_9.71.csv','try_z_5_9.71.csv'
                 ]
    '''exp_files = [ f'x_1_0mv_{sampleRate}.csv',f'y_1_0mv_{sampleRate}.csv',f'z_1_0mv_{sampleRate}.csv',
                f'x_2_r_mv_{sampleRate}.csv',f'y_2_r_mv_{sampleRate}.csv', f'z_2_r_mv_{sampleRate}.csv',
                f'x_3_1st_p_min_{sampleRate}.csv', f'y_3_1st_p_min_{sampleRate}.csv', f'z_3_1st_p_min_{sampleRate}.csv',
                f'x_4_2st_p_min_{sampleRate}.csv', f'y_4_2st_p_min_{sampleRate}.csv', f'z_4_2st_p_min_{sampleRate}.csv',
                f'x_5_3st_p_min_w_ad_{sampleRate}.csv', f'y_5_3st_p_min_w_ad_{sampleRate}.csv',f'z_5_3st_p_min_w_ad_{sampleRate}.csv']'''
    
    input_file_0 = exp_files
    input_path_0 = './TRY/RAW/'
    output_file_0 =  [f"{fname.replace('.csv', f'_clean.csv')}"
             for fname in exp_files]
    output_path_0 = './TRY/'
    
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
        print(f'Saved trimmed file: {output_file_0} with {len(trimmed_df)} rows.')
    print('\nAll datasets aligned and trimmed successfully.\n')
    
    # ================= Step 4: Plotting all datasets for inspection =================
    
    print(f'\n======= Data Process: 1 for {window}s  =======\n')
    import numpy as np
    from scipy.fft import fft
    from scipy.stats import iqr
    
    input_file_1 = output_file_0
    input_path_1 = output_path_0 
    output_path_1 = output_path_0 
    output_file_1 = [f"{fname.replace('.csv', f'_feat_prepr.csv')}" for fname in exp_files]
    
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
        return

    # ---- Check input files before processing ----  
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_file)(i, f) for i, f in enumerate(input_file_1)
    )
        
    print(f'\n======= Data Process: 2 for {window}s  =======\n')
    
    
    input_file_2 = output_file_1
    input_path_2 = output_path_0 
    output_path_2 = output_path_0 
    output_file_2 = [f"{fname.replace('.csv', f'_feat.csv')}"for fname in exp_files]
    
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
        
    print(f'\n======= Data Process: 3 for {window}s  =======\n')
    
    input_file_3 = output_file_2
    input_path_3 = output_path_0 
    output_path_3 = output_path_0
    feat_files = input_file_3  
    
    all_X = []
    all_y = []
    all_raw_data = []   
    all_norm_data = [] 
    raw_labels = []
    norm_labels = []
    #labels = [6]
    
    for file_idx, f in enumerate(feat_files):
        #print(f'Processing file: {f}')
        df = pd.read_csv(os.path.join(input_path_3, f))
        
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
        all_y.append(np.full(len(df_windowed), file_idx))  # dataset number as label  file_idx
        
    # Concatenate into big DataFrames
    X_data = pd.concat(all_X, axis=0, ignore_index=True)
    y_data = pd.Series(np.concatenate(all_y), name='label')
    out_path1 = os.path.join(output_path_0,'X_data.csv')
    out_path2 = os.path.join(output_path_0,'y_data.csv')
    X_data.to_csv(out_path1, index=False, header=True)
    y_data.to_csv(out_path2, index=False, header=True)
    
    print(f'\n======= Data Process: 5 for {window}s  =======\n')
    from sklearn.metrics import (accuracy_score,ConfusionMatrixDisplay)
    model= joblib.load(f"BEST{classifier_name}W{window}F{file_index}.pkl")
    # --- Step 1: Extract model’s training feature names ---
    expected_features = model.feature_names_in_
    
    # --- Step 2: Keep only those columns ---
    X_new = X_data.loc[:, expected_features]
    
    # --- Step 3: Predict ---
    y_pred = model.predict(X_new)  # works like normal sklearn model
    test_score = accuracy_score(y_data, y_pred)
    

    def capture_output_and_plot(classifier_name, accuracy, Data_tag, 
                                classifier, X_test, y_test):
        # Original class names from filenames
        original_class_names   = [
            f'x_1_0mv_{sampleRate}',f'y_1_0mv_{sampleRate}',f'z_1_0mv_{sampleRate}',
            f'x_2_r_mv_{sampleRate}',f'y_2_r_mv_{sampleRate}', f'z_2_r_mv_{sampleRate}',
            f'x_3_1st_p_min_{sampleRate}', f'y_3_1st_p_min_{sampleRate}', f'z_3_1st_p_min_{sampleRate}',
            f'x_4_2st_p_min_{sampleRate}', f'y_4_2st_p_min_{sampleRate}', f'z_4_2st_p_min_{sampleRate}',
            f'x_5_3st_p_min_w_ad_{sampleRate}', f'y_5_3st_p_min_w_ad_{sampleRate}',f'z_5_3st_p_min_w_ad_{sampleRate}'
        ]
        import io
        
        import matplotlib.pyplot as plt
        from contextlib import redirect_stdout  
        accuracy = str(round(accuracy * 100, 2))
        buf = io.StringIO()
        with redirect_stdout(buf):
            # Plot confusion matrix image as an example 
            
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
            # plt.gca().set_xticklabels(class_names, rotation=90)
            # plt.gca().set_yticklabels(class_names)
            # Highlight misclassified cells in red
            # --- Add red shades to misclassified cells ---
            cm = disp.confusion_matrix
            max_misclass = cm[np.where(~np.eye(cm.shape[0], dtype=bool))].max()
            
            for (i, j), val in np.ndenumerate(cm):
                if i != j and val > 0:
                    alpha = min(1.0, 0.2 + 0.8 * (val / max_misclass))
                    rect = plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        facecolor=(1, 0, 0, alpha),
                        edgecolor='none'
                    )
                    ax.add_patch(rect)
                    
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
            plt.savefig(f'TRYResults_{classifier_name}_image.png', dpi=600, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
            plt.close()
    
        return buf.getvalue(), '{classifier_name}_image.png'



    
    classification_text, image_path = capture_output_and_plot(classifier_name,
                                                                  test_score, 
                                                                      'TRY_ALL_DATA',  
                                                                      model, 
                                                                      X_new,
                                                                      y_data)