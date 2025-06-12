
"""
Created on Wed Jun 11 16:40:06 2025

@author: AXILLIOS
"""

import pandas as pd

# Load CSV
df = pd.read_csv('y_1deg_per_min_feat.csv')

# Drop the time column (assumed to be the first column)
df = df.iloc[:, 1:]

# Your feature indices
feature_indices = [76, 32, 42, 46, 70, 28, 63, 27, 45, 62]

# Get the column names at these indices
matched_features = [df.columns[i] for i in feature_indices]

# Print result
print("Matched feature names:")
for idx, name in zip(feature_indices, matched_features):
    print(f"Index {idx}: {name}")