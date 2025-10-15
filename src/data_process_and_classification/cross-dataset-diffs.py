import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv("X_data_9.71DT.csv")
X_try = pd.read_csv("TRY_X_data.csv")

# find mean and std for both datasets, so we can compare them. Ideally we would want them to be very close
stats_train = X.describe().T[['mean', 'std']]
stats_try = X_try.describe().T[['mean', 'std']]

diff = pd.DataFrame({
    'mean_diff': (stats_train['mean'] - stats_try['mean']).abs(),
    'std_diff': (stats_train['std'] - stats_try['std']).abs()
})

# Sort by drift 
diff_sorted = diff.sort_values('mean_diff', ascending=True)

print("\n**** Mean and Std Differences Between X_data and TRY_X_data ****\n")
print(diff_sorted.head(10).round(4).to_string())