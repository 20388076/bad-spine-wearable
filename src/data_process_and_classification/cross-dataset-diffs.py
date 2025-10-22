import pandas as pd, os
import matplotlib.pyplot as plt

X = pd.read_csv(os.path.join('./4_FEATS_COMBINED/9.71_Hz_sampling/DT/',"X_data_9.71DT.csv"))
X_try = pd.read_csv(os.path.join('./4_FEATS_COMBINED/9.71_Hz_sampling/DT/','VAL_X_9.71DT.csv'))
fNames = X.columns.tolist()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X)
X_test = scaler.transform(X_try) 
X_train = pd.DataFrame(X_train)#, header=fNames)
X_test = pd.DataFrame(X_test)#, header=fNames)
# find mean and std for both datasets, so we can compare them. Ideally we would want them to be very close
stats_train = X_train.describe().T[['mean', 'std']]
stats_try = X_test.describe().T[['mean', 'std']]

diff = pd.DataFrame({
    'mean_diff': (stats_train['mean'] - stats_try['mean']).abs(),
    'std_diff': (stats_train['std'] - stats_try['std']).abs()
})

# Sort by drift 
diff_sorted = diff.sort_values('mean_diff', ascending=True)

print("\n**** Mean and Std Differences Between X_data and TRY_X_data ****\n")
print(diff_sorted.round(4).to_string())