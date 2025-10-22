import pandas as pd, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

X_train = pd.read_csv(os.path.join('./4_FEATS_COMBINED/9.71_Hz_sampling/DT/',"X_data_9.71DT.csv"))
y_train = pd.read_csv(os.path.join('./4_FEATS_COMBINED/9.71_Hz_sampling/DT/',"y_data_9.71DT.csv"))
X_test = pd.read_csv(os.path.join('./4_FEATS_COMBINED/9.71_Hz_sampling/DT/','VAL_X_9.71DT.csv'))
y_test = pd.read_csv(os.path.join('./4_FEATS_COMBINED/9.71_Hz_sampling/DT/','VAL_y_9.71DT.csv'))
#X_try = pd.read_csv("DATA2sec/TRY_X_data.csv")
#y_try = pd.read_csv("DATA2sec/TRY_y_data.csv")
'''


X_combined = pd.concat([X, X_try], ignore_index=True)
y_combined = pd.concat([y, y_try], ignore_index=True)

# standardize 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.25, random_state=42
)
'''
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train, random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)    

    
rf = RandomForestClassifier(
    n_estimators=25,       
    max_depth=15,           
    min_samples_leaf=3,     # prevents overfitting apparently?
    random_state=42,
    n_jobs=-1               
)
rf.fit(X_test, y_test)

y_pred = rf.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
cm = confusion_matrix(y_train, y_pred)
#importances = pd.Series(rf.feature_importances_, index=X_train.columns)
#top10 = importances.sort_values(ascending=False).head(10)
# Export model to c code
import os
print(f"Combined dataset (Random Forest) // Accuracy: {accuracy*100:.2f}%")
print(cm)
from micromlgen import port
model_code = port(rf)

# save_dir = paths[1]
# Windows path (use raw string or double backslashes)
save_dir = '../' 

# Ensure path exists (optional safety check)
os.makedirs(save_dir, exist_ok=True)

# Full path to save header file
save_path = os.path.join(save_dir, 'RF8sec.h')

# Write the file
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(model_code) 
'''  
import m2cgen as m2c   
# Export model to c code    
c_code = m2c.export_to_c(rf)
save_dir = '../'
# Ensure path exists (optional safety check)

os.makedirs(save_dir, exist_ok=True)

# Full path to save header file
save_path = os.path.join(save_dir, 'RF2sec.h')

# Write the file
with open(save_path , 'w') as f:
    f.write(c_code)



for feature, importance in top10.items():
    print(f"{feature:20s} : {importance:.4f}")
'''