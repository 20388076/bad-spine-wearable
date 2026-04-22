"""
Created on Wed Apr 15 11:45:28 2026

@author: AXILLIOS
"""
# **************** Stage Functions *********************
# ---------------- Data Process 0: for cleaning raw data ----------------
def stage_0(window,window_size,sampleRate,classifier_name,series_of_experiments,plot_mode,mode):
    print(f'\n ======= Data Process: 0 Mode:{mode} for {window}s =======\n')
    # ---------------- Import Libraries ------------------
    import os, pandas as pd, matplotlib.pyplot as plt, numpy as np
    from utils.get_paths import get_paths
    # ----------------------------------------------------
    
    # Define the path for saving plots
    def plot (output_path):
        plot_path = os.path.join(output_path, 'PLOTS')
        # Create output directory if it doesn't exist
        os.makedirs(plot_path, exist_ok=True)
        return plot_path
    
    input_file_0, input_path_0, output_file_0, output_path_0 = get_paths(0, sampleRate, classifier_name, series_of_experiments, mode)

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
            
            # Number of extra lines to remove at the start of each file because the ESP32 produce a header
            extra_lines = 15
            df = pd.read_csv(
                            in_file,
                            names=train_labels,
                            skiprows= extra_lines,
                            encoding="utf-8",
                            encoding_errors="ignore"
                        )
            df.reset_index(drop=True, inplace=True)
            # Find starting time and sort
            start_time = df['t (ms)'].dropna().min()
            df = df.sort_values('t (ms)').reset_index(drop=True)
            
            # Limit time range to 1 hour and 1 min so the window value does not effect the wanted data (wanted data =< 1 hour data)
            df = df[df['t (ms)'] <= 3660000 + start_time]  
            # Store for later processing
            
        if series_of_experiments == 2:
            df = pd.read_csv(in_file)
            start_time = df['t (ms)'].dropna().min()
            time_limit_ms = 1 * 61 * 1000  # 1 minute 1 sec
            print(f"[INFO] Limiting {in_name} to 1 minute and 1 sec({time_limit_ms} ms)")

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
    
            duration_ms = trimmed_df["t (ms)"].iloc[-1] - trimmed_df["t (ms)"].iloc[0]
            duration_min = duration_ms / 60000.0
    
            out_file = os.path.join(output_path_0, output_file_0[file_idx])
            trimmed_df.to_csv(out_file, index=False)
    
            print(f"[INFO] {output_file_0[file_idx]} → kept {len(trimmed_df)} rows "
                  f"≈ {duration_min:.2f} min ")

    # ================= Plotting all datasets for inspection =================
    if plot_mode == 1:
        print("[INFO] Stage 0 plotting enabled")
        datasets = {fname: pd.read_csv(os.path.join(output_path_0, fname)) for fname in output_file_0}
    
        # Axis labels for Series 1
        axis_labels = {'x': 'X Axis', 'y': 'Y Axis', 'z': 'Z Axis'}
    
        if series_of_experiments == 1:
            # ---- SERIES 1: {axis}_{number}_{rate}.csv ----
            exp_conditions = [f'_{i}_' for i in range(1, 6)]
            exp_names = [f'Exp {i}' for i in range(1, 6)]
    
            for axis, axis_name in axis_labels.items():
                fig, axs = plt.subplots(len(exp_conditions), 2, figsize=(9, 9), sharex=True)
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
                        axs[row, 0].set_ylabel('Acceleration (m/s$^2$)', fontsize=9)
                    if g_x and g_y and g_z:
                        axs[row, 1].plot(time, df[g_x], 'r', label='Gyro X')
                        axs[row, 1].plot(time, df[g_y], 'g', label='Gyro Y')
                        axs[row, 1].plot(time, df[g_z], 'b', label='Gyro Z')
                        axs[row, 1].set_ylabel('Angular Velocity (deg/s)', fontsize=9)
    
                    axs[row, 0].set_title(f"{axis_name} - {exp_names[row]} - Acc")
                    axs[row, 1].set_title(f"{axis_name} - {exp_names[row]} - Gyro")
    
                fig.suptitle(f'{axis_name} - Step Conditions', fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plot_path = os.path.join(plot(output_path_0), f'{axis}_axis_plot.png')
                plt.savefig(plot_path, dpi=600)
                #plt.show()
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
                fig, axs = plt.subplots(3, 2, figsize=(9, 9), sharex=True)
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
                    #axs[row, 0].legend(fontsize=8)
                    axs[row, 0].set_ylabel('Acceleration (m/s$^2$)', fontsize=9)
                    # Plot gyroscope data
                    for t, gx, gy, gz in all_gyro:
                        axs[row, 1].plot(t, gx, 'r', label='Gyro X', alpha=0.7)
                        axs[row, 1].plot(t, gy, 'g', label='Gyro Y', alpha=0.7)
                        axs[row, 1].plot(t, gz, 'b', label='Gyro Z', alpha=0.7)
                    axs[row, 1].set_title(f"{cond.upper()} - Gyroscope")
                    # axs[row, 1].legend(fontsize=8)
                    axs[row, 1].set_ylabel('Angular Velocity (deg/s)', fontsize=9)

                # Label bottom row
                axs[-1, 0].set_xlabel('Time (min)')
                axs[-1, 1].set_xlabel('Time (min)')
                plt.tight_layout(rect=[0, 0, 1, 0.96])

                plot_name = f'series2_experiment_{exp_tag[-1]}_conditions.png'
                plot_path = os.path.join(plot(output_path_0), plot_name)
                plt.savefig(plot_path, dpi=600)
                #plt.show()
                plt.close(fig)
            
    return 
