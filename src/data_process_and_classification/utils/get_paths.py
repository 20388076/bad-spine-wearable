"""
Created on Wed Apr 22 08:54:49 2026

@author: AXILLIOS
"""
# ----------------------------- File System Configuration ----------------------------

def get_paths(stage, sampleRate, classifier_name,series_of_experiments, mode='exp'):
    # ---------------- Import Libraries ------------------
    import os
    # ----------------------------------------------------
    
    base = f"{sampleRate}_Hz_sampling/{classifier_name}"
    if series_of_experiments == 1:
        exp_files = [f'x_1_{sampleRate}.csv', f'y_1_{sampleRate}.csv', f'z_1_{sampleRate}.csv',
                     f'x_2_{sampleRate}.csv', f'y_2_{sampleRate}.csv', f'z_2_{sampleRate}.csv',
                     f'x_3_{sampleRate}.csv', f'y_3_{sampleRate}.csv', f'z_3_{sampleRate}.csv',
                     f'x_4_{sampleRate}.csv', f'y_4_{sampleRate}.csv', f'z_4_{sampleRate}.csv',
                     f'x_5_{sampleRate}.csv', f'y_5_{sampleRate}.csv', f'z_5_{sampleRate}.csv']
        
        test_files = [f'test_x_1_{sampleRate}.csv', f'test_y_1_{sampleRate}.csv', f'test_z_1_{sampleRate}.csv',
                          f'test_x_2_{sampleRate}.csv', f'test_y_2_{sampleRate}.csv', f'test_z_2_{sampleRate}.csv',
                          f'test_x_3_{sampleRate}.csv', f'test_y_3_{sampleRate}.csv', f'test_z_3_{sampleRate}.csv',
                          f'test_x_4_{sampleRate}.csv', f'test_y_4_{sampleRate}.csv', f'test_z_4_{sampleRate}.csv',
                          f'test_x_5_{sampleRate}.csv', f'test_y_5_{sampleRate}.csv', f'test_z_5_{sampleRate}.csv']
        
        exp_path = f'./0_RAW/series_of_experiments_1/{sampleRate}_Hz_sampling/'
        
    else:

        exp_files = [f'good_1_{sampleRate}.csv', f'good_2_{sampleRate}.csv', f'good_3_{sampleRate}.csv',
                     f'mid_1_{sampleRate}.csv', f'mid_2_{sampleRate}.csv', f'mid_3_{sampleRate}.csv',
                     f'bad_1_{sampleRate}.csv', f'bad_2_{sampleRate}.csv', f'bad_3_{sampleRate}.csv']

        test_files = [f'test_good_1_{sampleRate}.csv', f'test_good_2_{sampleRate}.csv', f'test_good_3_{sampleRate}.csv',
                      f'test_mid_1_{sampleRate}.csv', f'test_mid_2_{sampleRate}.csv', f'test_mid_3_{sampleRate}.csv',
                      f'test_bad_1_{sampleRate}.csv', f'test_bad_2_{sampleRate}.csv', f'test_bad_3_{sampleRate}.csv']

        exp_path = f'./0_RAW/series_of_experiments_2/{sampleRate}_Hz_sampling/'
            
    if mode == 'exp' :
        files_to_use = exp_files
        folder = "EXP"
    else:
        files_to_use = test_files
        folder = "TEST"

    if stage == 0:
        in_files = files_to_use
        in_path = exp_path
        out_path = f'./1_CLEAN/{base}/{folder}/'
        out_files = [f"{fname.replace('.csv', f'{classifier_name}_clean.csv')}" for fname in files_to_use]
    elif stage == 1:
        in_files = [f"{fname.replace('.csv', f'{classifier_name}_clean.csv')}" for fname in files_to_use]
        in_path = f'./1_CLEAN/{base}/{folder}/'
        out_path = f'./2_FEATS_PREPROCESSSED/{base}/{folder}/'
        out_files = [f"{fname.replace('.csv', f'{classifier_name}_feat_prepr.csv')}" for fname in files_to_use]
    elif stage == 2:
        in_files = [f"{fname.replace('.csv', f'{classifier_name}_feat_prepr.csv')}" for fname in files_to_use]
        in_path = f'./2_FEATS_PREPROCESSSED/{base}/{folder}/'
        out_path = f'./3_FEATS/{base}/{folder}/'
        out_files = [f"{fname.replace('.csv', f'{classifier_name}_feat.csv')}" for fname in files_to_use]
    elif stage == 3:
        in_files = [f"{fname.replace('.csv', f'{classifier_name}_feat.csv')}" for fname in files_to_use]
        in_path = f'./3_FEATS/{base}/{folder}/'
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