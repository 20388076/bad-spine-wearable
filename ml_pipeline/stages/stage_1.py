"""
Created on Wed Apr 15 11:57:21 2026

@author: AXILLIOS
"""
# ---------------- Data Process 1: Feature Extraction ----------------
def stage_1(window,window_size,sampleRate,classifier_name,series_of_experiments,mode):
    print(f'\n======= Data Process: 1 Mode:{mode} for {window}s  =======\n')
    
    # ---------------- Import Libraries ------------------
    import os, pandas as pd, numpy as np, multiprocessing
    from scipy.fft import fft
    from scipy.stats import iqr
    from joblib import Parallel, delayed
    from utils import UtilFunc as UF
    from utils.get_paths import get_paths
    # ----------------------------------------------------
    t1=UF.tic()
    
    input_file_1, input_path_1, output_file_1, output_path_1, plot_path = get_paths(1, sampleRate, classifier_name, series_of_experiments, mode)
    
    def process_file(file_idx, fname):  
        df = pd.read_csv(input_path_1 + input_file_1[file_idx])
        # print(f'Processing file: {input_file_1[file_idx]}')
        
        # Normalize acceleration to acceleration in g (9.80665 m/s^2)
        #--------------------------------------------------------------------
        df[['ag_x','ag_y','ag_z']] = (df[['a_x','a_y','a_z']] / 9.80665).round(3)
        # df[['ag_x','ag_y','ag_z']] = df[['a_x','a_y','a_z']] # without g normalize
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
    t2 = UF.tic()
    UF.toc(t1, t2-t1)
    return 