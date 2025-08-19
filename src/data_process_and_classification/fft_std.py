"""
Created on Tue Aug 12 15:42:34 2025

@author: AXILLIOS
"""

import os
import pandas as pd
import numpy as np

# Folder with your _feat.csv files
output_path_2 = './FEATS/'
fft_columns = [
    'FFT_acceleration x',
    'FFT_acceleration y',
    'FFT_acceleration z',
    'FFT_gyro x',
    'FFT_gyro y',
    'FFT_gyro z'
]

# 1. Find all *_feat.csv files in the folder
feat_files = [f for f in os.listdir(output_path_2) if f.endswith('_feat.csv')]
feat_files.sort()  # Ensure consistent order

if not feat_files:
    raise FileNotFoundError("No _feat.csv files found in ./FEATS/")

print(f"Found {len(feat_files)} feature files.")

# 2. Read all CSVs and keep only FFT columns
dfs = []
for file in feat_files:
    path = os.path.join(output_path_2, file)
    df = pd.read_csv(path, usecols=fft_columns)
    dfs.append(df)
    print(f"Loaded: {file} with shape {df.shape}")

# 3. Convert list of DataFrames to 3D NumPy array
#    Shape: (n_files, n_rows, n_cols)
data_stack = np.stack([df.values for df in dfs], axis=0)

# 4. Compute STD across the first axis (files)
#    Result shape: (n_rows, n_cols)
std_values = np.std(data_stack, axis=0, ddof=0).round(3)  # population std

# 5. Create final DataFrame
std_df = pd.DataFrame(std_values, columns=fft_columns)

# 6. Save result
output_file = os.path.join(output_path_2, 'fft_std.csv')
std_df.to_csv(output_file, index=False)

print(f"STD CSV saved as: {output_file}")
print(f"Shape: {std_df.shape}")
