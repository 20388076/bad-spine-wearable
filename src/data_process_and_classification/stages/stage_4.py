"""
Created on Wed Apr 15 12:18:22 2026

@author: AXILLIOS
"""
# ---------------- Data Process 4: ReliefF Feature Selection, Plotting and 10 best features, displaying for ESP32 use  ----------------
def stage_4(window,window_size,sampleRate,classifier_name,series_of_experiments,plot_mode,matlab):
    """
    Stage 4: ReliefF Feature Selection
    - Runs in Python by default.
    - If matlab=1, tries MATLAB Engine API, then Octave, else falls back to manual MATLAB Online.
    """
    print(f'\n======= Data Process: 4 for {window}s  =======\n')
    # ---------------- Import Libraries ------------------
    import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
    import UtilFunc as UF
    from utils.get_paths import get_paths
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
            eng.relieff_feature_selection(series_of_experiments,sampleRate,classifier_name,window,input_path_4,output_path_4,Xfname,yfname,nargout=0)
            eng.quit()

            weights_file = os.path.join(output_path_4,
                f'{series_of_experiments}Matlab_relieff_feature_indices_weights_{sampleRate}{classifier_name}{window}.csv')
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
            f'{series_of_experiments}Matlab_relieff_feature_indices_weights_{sampleRate}{classifier_name}{window}.csv'
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
                UF.sys.exit(0)
    # ---- Step 0: Set file paths and index mode ----
    input_file_4, input_path_4, time_path, output_path_4 = get_paths(4, sampleRate, classifier_name, series_of_experiments)
    Xfname = f'X_data_{sampleRate}{classifier_name}.csv'   
    X_data = pd.read_csv(os.path.join(input_path_4, Xfname))
    yfname = f'y_data_{sampleRate}{classifier_name}.csv'
    TEST_X = pd.read_csv(os.path.join(input_path_4,f'TEST_X_{sampleRate}{classifier_name}.csv'))
    
    if matlab == 1:
        # Step 0: Check if file already exists

        existing_file = check_existing_weights(output_path_4, sampleRate, classifier_name, window)
        if existing_file:
            weights_file, base_index = existing_file, 1  # MATLAB uses 1-based indexing
        else:
            t1 = UF.tic()
            weights_file, base_index = run_matlab_engine()
            t2 = UF.tic()
            UF.toc(t1, t2 - t1)

            if weights_file is None:  # manual fallback
                wait_for_key()
                weights_file = os.path.join(output_path_4, f'{series_of_experiments}Matlab_relieff_feature_indices_weights_{sampleRate}{classifier_name}{window}.csv')
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
    TEST_X_sorted = TEST_X.iloc[:, sorted_indices]
    
    # Save reordered datasets
    X_data_sorted.to_csv(os.path.join(output_path_4, f'{series_of_experiments}{names[base_index]}_X_data_weight_ordered_{sampleRate}{classifier_name}.csv'), index=False, header=True)
    TEST_X_sorted.to_csv(os.path.join(output_path_4, f'{series_of_experiments}{names[base_index]}_TEST_X_weight_ordered_{sampleRate}{classifier_name}.csv'), index=False, header=True)
    if plot_mode == 1:
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
        plt.savefig(os.path.join(plot(output_path_4), f'{series_of_experiments}{names[base_index]}_relieff_weights_plot_{sampleRate}{classifier_name}.png'), dpi=600)
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
        if plot_mode == 1:
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
            
            plot_path2 = os.path.join(plot(output_path_4), f'{series_of_experiments}{names[base_index]}_Feats_CustomScore_{sampleRate}{classifier_name}.png')
            plt.tight_layout()
            plt.savefig(plot_path2, dpi=600)
            plt.show()
            plt.close()
            print(f'Saved plot: {plot_path2}')
        
        # ---- Step 7: Save ALL features reordered by custom score ----
        custom_sorted_indices = df_sorted['Feature_Index'].to_numpy()
        
        X_data_custom_sorted = X_data.iloc[:, custom_sorted_indices]
        TEST_X_custom_sorted  = TEST_X.iloc[:, custom_sorted_indices]
        
        out_file_all_train_custom = os.path.join(output_path_4, f'{series_of_experiments}{names[base_index]}_X_data_custom_reordered_{sampleRate}{classifier_name}.csv')
        out_file_all_test_custom  = os.path.join(output_path_4, f'{series_of_experiments}{names[base_index]}_TEST_X_custom_reordered_{sampleRate}{classifier_name}.csv')
        
        X_data_custom_sorted.to_csv(out_file_all_train_custom, index=False, header=True)
        TEST_X_custom_sorted.to_csv(out_file_all_test_custom, index=False, header=True)
        
        print(f'Saved reordered X_train (custom scores): {out_file_all_train_custom}, shape {X_data_custom_sorted.shape}')
        print(f'Saved reordered TEST_X  (custom scores): {out_file_all_test_custom}, shape {TEST_X_custom_sorted.shape}')
        
    else:
        print('Warning: feats_computation_times.csv not found, skipping ESP32 plot.')
    return 
