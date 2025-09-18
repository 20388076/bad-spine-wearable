# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 10:18:03 2025

@author: AXILLIOS
"""

# -------------------------------  Working Directory -----------------------------
# Set the working directory to the script's location if running in Visual Studio Code
import os
# Change working directory for this script
os.chdir(r'C:\Users\user\OneDrive\Έγγραφα\PlatformIO\Projects\bad-spine-wearable-1\src\data_process_and_classification') # modify this path to your working directory

# ============================= Utility Functions =============================

# ----------------------------- Import Libraries ------------------------------
import time
import sys
# ----------------------------- Kernel clean ----------------------------------
def cls():
    print(chr(27) + '[2J') 
# ----------------------------- Kernel pause ----------------------------------
def pause():
    input('PRESS ENTER TO CONTINUE.')
# ----------------------------- Process time count ----------------------------
def tic():
    return float(time.time())
# ----------------------------- Process time return ---------------------------
def toc(t1, s):
    t2 = float(time.time())
    print(f'{s} time taken: {t2 - t1:.6e} seconds')
# ----------------------------- Kernel break ----------------------------------
def RETURN():
    sys.exit()
# =============================================================================

# ----------------------------- Kernel clean call -----------------------------
cls()

import pandas as pd
import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_validate
from scipy.stats import mode, randint, uniform
import m2cgen as m2c

def loadData(paths, index, input_file):
    r'''
    Load dataset X, y, feature names, and tag based on index from input_file.
    
    Parameters
    ----------
    paths : list
        [stage4_path, stage6_path]
    index : int
        Row index of input_file
    
    Returns
    -------
    X : pd.DataFrame
        Feature matrix (keeps column names)
    y : pd.Series
        Labels
    fNames : list
        Feature names
    Data_tag : str
        Tag describing the dataset
    '''

    row = input_file.iloc[index]
    X_file, y_file, Data_tag = row['X_file'], row['y_file'], row['Data_tag']

    # First 3 datasets are from Stage 4, rest from Stage 6
    base_path = paths[0] if index < 3 else paths[1]

    X_path = os.path.join(base_path, X_file)
    y_path = os.path.join(paths[0], y_file)

    # ---- Load X as DataFrame ----
    X_data = pd.read_csv(X_path, dtype=np.float32)
    fNames = X_data.columns.tolist()
    X = X_data  # keep DataFrame instead of converting to numpy

    # ---- Load y as Series ----
    y_data = pd.read_csv(y_path, header=None)

    # If first row is text (e.g., 'label'), drop it
    if isinstance(y_data.iloc[0, 0], str):
        y_data = y_data.iloc[1:]

    y = y_data.squeeze("columns").astype(np.int32)

    return X, y, fNames, Data_tag



def classifiers(cl):
    
    if cl == 0: # Decision Tree 
        from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
        return DecisionTreeClassifier(), 'DecisionTree'
        
    elif cl == 1: # Random Forest 
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(), 'RandomForest'
    
    

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# --- Step 1: Transformer to select top-k features ---
class TopKFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, k=10):
        self.k = k
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Works for both pandas DataFrame and numpy array
        if hasattr(X, "iloc"):  
            return X.iloc[:, :self.k]
        else:  
            return X[:, :self.k]


# --- Step 2: Custom refit rule factory ---
def make_refit_callable(tol=0.01, relative=True):
    """
    tol : float
        Tolerance value (0.01 = 1% if relative=True).
    relative : bool
        If True -> tolerance is relative (percentage).
        If False -> tolerance is absolute difference.
    """
    def refit_rule(cv_results):
        # cv_results is a dict of numpy arrays
        results = pd.DataFrame(cv_results)

        best_score = results['mean_test_score'].max()
        if relative:
            threshold = best_score * (1 - tol)   # within percentage
        else:
            threshold = best_score - tol         # within absolute margin

        candidates = results[
            results['mean_test_score'] >= threshold
        ]
        # pick the one with the smallest k
        best_idx = candidates['param_feat__k'].astype(int).idxmin()
        return best_idx
    return refit_rule



# --- Step 3: GridSearch with flexible refit ---
from sklearn.metrics import accuracy_score

def feature_subset_gridsearch(best_model, X_train, y_train, X_test, y_test,
                              cv, max_features=75, step = 5 , tol=0.01, relative=True):
    pipe = Pipeline([
        ('feat', TopKFeatures()), 
        ('clf', best_model)
    ])

    param_grid = {'feat__k': list(range(1, max_features+1, step))} 
    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="accuracy",
        refit=make_refit_callable(tol=tol, relative=relative),
        n_jobs=-1,
        return_train_score=True
    )
    grid.fit(X_train, y_train)

    results = pd.DataFrame(grid.cv_results_)

    # --- Compute test accuracy for each k ---
    test_scores = []
    for k in results['param_feat__k']:
        pipe.set_params(feat__k=k)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        test_scores.append(acc)
    results['test_score'] = test_scores

    best_k = grid.best_params_['feat__k']
    best_score = results.loc[results['param_feat__k'] == best_k, 'test_score'].values[0]

    # --- Plot test accuracy vs feature count ---
    plt.figure(figsize=(8, 5))
    plt.plot(results['param_feat__k'], results['test_score'], marker="o", label="Test Accuracy")
    plt.scatter([best_k], [best_score], color="red", s=100, zorder=5, label="Selected k")
    plt.xlabel("Number of Top Features")
    plt.ylabel("Test Accuracy")
    plt.title("Feature Count vs Test Performance")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Best feature count: {best_k}")
    print(f"Test Accuracy at this point: {best_score:.4f}")

    return grid, results


# ========================================================================
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, 
                             roc_curve, auc, RocCurveDisplay, make_scorer, root_mean_squared_error)
# Available Datasets
# 1) 9.71 Hz
# 2) 10 Hz
# 3) 50 Hz
sampleRate = 9.71 # Sample rate in Hz     <-- Change this value to set sample rate

folder_path = f'./3_FEATS/{sampleRate}_Hz_sampling/'
files = [
    f'x_1_0mv_{sampleRate}.csv',f'y_1_0mv_{sampleRate}.csv',f'z_1_0mv_{sampleRate}.csv',
    f'x_2_r_mv_{sampleRate}.csv',f'y_2_r_mv_{sampleRate}.csv', f'z_2_r_mv_{sampleRate}.csv',
    f'x_3_1st_p_min_{sampleRate}.csv', f'y_3_1st_p_min_{sampleRate}.csv', f'z_3_1st_p_min_{sampleRate}.csv',
    f'x_4_2st_p_min_{sampleRate}.csv', f'y_4_2st_p_min_{sampleRate}.csv', f'z_4_2st_p_min_{sampleRate}.csv',
    f'x_5_3st_p_min_w_ad_{sampleRate}.csv', f'y_5_3st_p_min_w_ad_{sampleRate}.csv',f'z_5_3st_p_min_w_ad_{sampleRate}.csv'
]

paths = [f'./4_FEATS_COMBINED/{sampleRate}_Hz_sampling/',f'./5_FEATS_SELECTION/{sampleRate}_Hz_sampling/']

input_file_train = pd.DataFrame([
    [f"all_raw_train_{sampleRate}.csv", f"y_all_raw_train_{sampleRate}.csv", "RAW_DATA "],
    [f"all_norm_train_{sampleRate}.csv", f"y_all_norm_train_{sampleRate}.csv", "G_RAW_DATA "],
    [f"X_train_{sampleRate}.csv", f"y_train_{sampleRate}.csv", "ALL_DATA "],
    [f"Matlab_X_train_weight_ordered_{sampleRate}.csv", f"y_train_{sampleRate}.csv", "WEIGHT BASED FEATURES "],
    [f"Matlab_X_train_custom_reordered_{sampleRate}.csv",f"y_train_{sampleRate}.csv", "SCORE BASED FEATURES "]
        ], columns=['X_file', 'y_file', 'Data_tag'])
    
input_file_test = pd.DataFrame([
    [f"all_raw_test_{sampleRate}.csv", f"y_all_test_{sampleRate}.csv", "RAW_DATA "],
    [f"all_norm_test_{sampleRate}.csv", f"y_all_norm_test_{sampleRate}.csv", "G_RAW_DATA "],
    [f"X_test_{sampleRate}.csv", f"y_test_{sampleRate}.csv", "ALL_DATA "],
    [f"Matlab_X_test_weight_ordered_{sampleRate}.csv", f"y_test_{sampleRate}.csv", "WEIGHT BASED FEATURES "],
    [f"Matlab_X_test_custom_reordered_{sampleRate}.csv", f"y_test_{sampleRate}.csv", "SCORE BASED FEATURES "]
        ], columns=['X_file', 'y_file', 'Data_tag'])

index = 2

for cl in range(2):
    
    X_train, y_train, fNames, Data_tag = loadData(paths, index, input_file_train)
    # X_train = X_train.iloc[:, :10].copy()
    # fNames = pd.DataFrame(row[:10] for row in fNames)
    X_test, y_test, fNs, Data_tag = loadData(paths, index, input_file_test)
    # X_test = X_test.iloc[:, :10].copy()
    
    classifier, classifier_name = classifiers(cl)
    
    
    # Parameter distributions for RandomizedSearchCV
    param_dists = {
        #KNN': {'n_neighbors': randint(3, 15),'weights': ['uniform', 'distance'],'metric': ['euclidean', 'manhattan']},
        'DecisionTree': {
            'max_depth': randint(1, 50),
            'min_samples_split': randint(2, 10)
        },
        'RandomForest': {
            'n_estimators': randint(1, 200),
            'max_depth': randint(1, 50),
            'min_samples_split': randint(2, 10)
        }
        #XGBoost': {'n_estimators': randint(50, 300),'max_depth': randint(3, 10),'learning_rate': uniform(0.01, 1)}
    }
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    ts_cv = TimeSeriesSplit(n_splits=5) 

    search = RandomizedSearchCV(classifier, 
                                param_dists[classifier_name], 
                                n_iter=1000, 
                                cv=ts_cv, 
                                n_jobs=-1, 
                                scoring='accuracy', 
                                random_state=42)
    
    # Fit and get results
    search.fit(X_train, y_train)
    best_params = search.best_params_
    best_cv_score = search.best_score_
    
    # Test set accuracy
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    
    y_pred = best_model.predict(X_train)
    train_score = accuracy_score(y_train, y_pred)
    summary = []
    
    # Store for summary
    summary.append({
        'Classifier': classifier_name,
        'CV Accuracy': round(best_cv_score, 4),
        'Train Accuracy': round(train_score, 4),
        'Test Accuracy': round(test_score, 4),
        'Best Params': best_params,
    })
    # Summary
    print('\n Summary:')
    for s in summary:
        print(f'\n {s['Classifier']}')
        print(f'   CV Accuracy  : {s['CV Accuracy']}')
        print(f'   Train : {s['Train Accuracy']}')
        print(f'   Test Accuracy: {s['Test Accuracy']}')
        print(f'   Best Params  : {s['Best Params']}')
    r'''    
    # Export model to c code    
    c_code = m2c.export_to_c(best_model)
    
    # Ensure path exists (optional safety check)
    save_dir = paths[1]
    os.makedirs(save_dir, exist_ok=True)
    
    # Full path to save header file
    save_path = os.path.join(save_dir, f'Best_{classifier_name}.h')
    
    # Write the file
    with open(save_path , 'w') as f:
        f.write(c_code)
    '''
    # Export model to c code
    from micromlgen import port
    model_code = port(best_model)
    
    # save_dir = paths[1]
    # Windows path (use raw string or double backslashes)
    save_dir = '../' 
    
    # Ensure path exists (optional safety check)
    os.makedirs(save_dir, exist_ok=True)
    
    # Full path to save header file
    save_path = os.path.join(save_dir, f'Best_{classifier_name}.h')
    
    # Write the file
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(model_code)
        
  
    def capture_output_and_plot(classifier_name, accuracy, Data_tag, 
                                classifier, X_test, y_test):
        import io
        from contextlib import redirect_stdout  
        accuracy = str(round(accuracy * 100, 2))
        buf = io.StringIO()
        with redirect_stdout(buf):
            # Plot confusion matrix image as an example 
            # Original class names from filenames
            original_class_names = [filename.replace('_feat.csv', '') 
                                    for filename in files]
            
            # Generate numeric labels for display
            class_names = [f'Class {i}' for i in range(len(original_class_names))]
            
            # Create a mapping legend
            legend_text = '\n'.join([f'{class_names[i]}: {original_class_names[i]}' 
                                     for i in range(len(class_names))])
            
            labels = np.unique(y_test)  # integers that appear in y_test
            fig, ax = plt.subplots(figsize=(10, 8))  # wider & taller
            disp = ConfusionMatrixDisplay.from_estimator(
                classifier,
                X_test,
                y_test,
                labels=labels,                                      
                display_labels=[f'Class {i}' for i in labels],       
                normalize='true',
                cmap=plt.cm.Blues,                                    # row-wise %
                xticks_rotation=90,
                values_format='.1%',  
                ax=ax            # show as percents
                )
            
            # Remove scientific notation    
            plt.gca().set_xticklabels(class_names, rotation=90)
            plt.gca().set_yticklabels(class_names)
            
            # Add the legend as a textbox
            plt.gcf().text(1.02, 
                           0.5, 
                           legend_text, 
                           fontsize=12, 
                           va='center', 
                           bbox=dict(facecolor='white', edgecolor='black')
                           )
           
            # Title
            plt.title(f'{classifier_name} Confusion Matrix \n' + Data_tag 
                       + accuracy + '%', fontsize=16)
            
            # Improve readability
            plt.tick_params(axis='x', labelsize=10)
            plt.tick_params(axis='y', labelsize=10)
            
            # Optional: Bold larger numbers or set font size
            for text in disp.ax_.texts:
                text.set_fontsize(6)  # Increase for better visibility (try 10–12 if need 
            # Save and show
            plt.savefig('temp_image.png', dpi=600, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
            plt.close()
    
        return buf.getvalue(), 'temp_image.png'
    
    classification_text, image_path = capture_output_and_plot(classifier_name,
                                                              test_score, 
                                                              Data_tag,  
                                                              best_model, 
                                                              X_test, 
                                                           y_test)
    
    r'''
# ========================================================================
    grid, results = feature_subset_gridsearch(
    best_model,
    X_train, y_train,
    X_test, y_test,   
    cv=ts_cv,
    max_features=75,
    step=1,
    tol=0.01,
    relative=True
    )
    '''