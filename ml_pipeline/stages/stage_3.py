"""
Created on Wed Apr 15 12:12:58 2026

@author: AXILLIOS
"""
# ---------------- Data Process 3: ----------------
def stage_3(window,window_size,sampleRate,classifier_name,series_of_experiments,mode):
    print(f'\n======= Data Process: 3  Mode:{mode} for {window}s  =======\n')
    
    # ---------------- Import Libraries ------------------
    import os, pandas as pd, numpy as np
    from utils.get_paths import get_paths
    # ----------------------------------------------------
    # ------------------- Sub Function -------------------
    # ----------------------------------------------------
    
    input_file_3, input_path_3, output_file_3, output_path_3, plot_path = get_paths(3, sampleRate, classifier_name, series_of_experiments, mode)
     
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

            # Try to extract the experiment number from the filename
            match = re.search(r'_(\d+)_', f)
            if match:
                exp_number = int(match.group(1))
                label_num = exp_number - 1  # make zero-based (1→0, 2→1, etc.)
            else:
                # fallback if pattern not found
                print(f"[Warning] Could not extract experiment number from filename: {f}")
                label_num = file_idx
 
            # Apply one label per window (since all rows in the file belong to the same experiment)
            label_val = np.full(len(df_windowed), label_num)
        
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
            f'y_data_{sampleRate}{classifier_name}.csv': y_data.to_frame(),   # ensure it has name 'label' on first row
            
            # raw version
            f'all_raw_data_{sampleRate}{classifier_name}.csv': all_raw_data,
            f'y_all_raw_data_{sampleRate}{classifier_name}.csv': pd.DataFrame(raw_y, columns=['label']),
            
            # norm version
            f'all_norm_data_{sampleRate}{classifier_name}.csv': all_norm_data,
            f'y_all_norm_data_{sampleRate}{classifier_name}.csv': pd.DataFrame(norm_y, columns=['label']),
        }
            
    elif mode == 'test':
         # ---- Save datasets (headers, NO index) ----
         save_items = {
             # aggregated features
             f'TEST_X_{sampleRate}{classifier_name}.csv': X_data,
             f'TEST_y_{sampleRate}{classifier_name}.csv': y_data.to_frame(),   # ensure it has name 'label' on first row
             
             # raw version
             f'TEST_all_raw_data_{sampleRate}{classifier_name}.csv': all_raw_data,
             f'TEST_y_all_raw_data_{sampleRate}{classifier_name}.csv': pd.DataFrame(raw_y, columns=['label']),
             
             # norm version
             f'TEST_all_norm_data_{sampleRate}{classifier_name}.csv': all_norm_data,
             f'TEST_y_all_norm_data_{sampleRate}{classifier_name}.csv': pd.DataFrame(norm_y, columns=['label']),
         }   
        
    for fname, df in save_items.items():
        path = os.path.join(output_path_3, fname)
        
        df.to_csv(path, index=False, header=True)
        print(f'Saved {fname} with shape {df.shape}')    
    return   
