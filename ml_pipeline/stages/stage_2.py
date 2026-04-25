"""
Created on Wed Apr 15 12:04:40 2026

@author: AXILLIOS
"""
# ---------------- Data Process 2: ----------------
def stage_2(window,window_size,sampleRate,classifier_name,series_of_experiments,mode):
    print(f'\n======= Data Process: 2 Mode:{mode} for {window}s  =======\n')
    # ---------------- Import Libraries ------------------
    import os, pandas as pd, numpy as np, json, shutil 
    from utils.get_paths import get_paths
    # ----------------------------------------------------
    
    input_file_2, input_path_2, output_file_2, output_path_2, plot_path = get_paths(2, sampleRate, classifier_name, series_of_experiments, mode)
    fft_std_path = os.path.join(output_path_2, f'{series_of_experiments}fft_std{classifier_name}.csv')
    fft_index_dir = '../output/pc_classification/fft_index'
    os.makedirs(fft_index_dir, exist_ok=True)
    best_index_path = os.path.join(fft_index_dir, f'{series_of_experiments}best_fft_index{classifier_name}.json')
    
    feat_files = input_file_2
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
    # TEST MODE → load precomputed best indices (safe skip)
    # ===========================================================
    if mode == 'test':
        if not os.path.exists(best_index_path):
            print("[Stage 2 - {mode.upper()}] No best_fft_index.json found. Skipping this mode stage 2.")
            return
        with open(best_index_path, 'r') as f:
            best_idx_map = json.load(f)
        print(f"[Stage 2 - {mode.upper()}] Loaded best FFT indices from {best_index_path}")
    
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
            
            dfs.append(df_tmp)
        
            print(f"[Stage 2 - EXP] File {f}: using {n_full_windows} windows ")
        
        # If no valid FFT data collected, exit early
        if not dfs:
            print(f"[Stage 2 - {mode.upper()}] No valid training FFT data found. Skipping stage.")
            return
        
        # Stack all training portions from each file

        df_stack = np.stack([d.values for d in dfs], axis=0)
        # Compute std across all training portions
        std_values = np.std(df_stack, axis=0) 
        fft_std_df = pd.DataFrame(std_values, columns=fft_columns).round(3)
        fft_std_df.to_csv(fft_std_path, index=False)
        
        print(f"[Stage 2 - EXP] Saved training-only fft_std.csv ({fft_std_df.shape})")
    
        # Compute best index per FFT column based on std within training windows
        fft_scores = {}
        best_idx_map = {}
        for col in fft_columns:
            scores = np.zeros(window_size, dtype=int)
            for w in range(n_full_windows):
                chunk = fft_std_df[col].iloc[w*window_size:(w+1)*window_size].values
                if len(chunk) < 2:
                    continue
                arg = int(np.nanargmax(chunk[1:])) + 1
                scores[arg] += 1
            
            fft_scores[col] = scores.copy()
            best_idx_map[col] = int(np.argmax(scores[1:])) + 1
            
        for fft_name, scores in fft_scores.items():
            print(f"{fft_name}: {scores.tolist()}")
 
        with open(best_index_path, 'w') as f:
            json.dump(best_idx_map, f, indent=2)
        print(f"[Stage 2 - EXP] Saved best FFT indices to {best_index_path}")    
        scaler_dir = "../lib/scalers/"   
        os.makedirs(scaler_dir, exist_ok=True)
        header_path = os.path.join(scaler_dir , f"{series_of_experiments}best_fft_index{classifier_name}{window}.h")
        
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

