'''
Created on Tue May  6 14:39:30 2025

@author: AXILLIOS
'''
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

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import unique
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_validate
from scipy.stats import mode, randint, uniform
from micromlgen import port

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, 
                             roc_curve, auc, RocCurveDisplay, make_scorer, root_mean_squared_error)

import io
from contextlib import redirect_stdout
from docx import Document
from docx.shared import Pt, Inches

# import warnings
# warnings.filterwarnings("ignore")

def init_document(docx_filename):
    """Create a new document if it doesn't exist, otherwise load the existing one."""
    if os.path.exists(docx_filename):
        return Document(docx_filename)
    else:
        return Document()

#====================================================================
#========================== MAIN PROGRAM ============================
#====================================================================

cls() 
docx_filename = 'classification_results.docx'
doc = init_document(docx_filename)

files = ['movement_0_feat.csv',
         'x_axis_with_random_movements_feat.csv',
         'x_1step_per_min_feat.csv', 
         'x_2step_per_min_feat.csv',
         'x_anomaly_detection_3_step_per_min_feat.csv',
         'y_axis_with_random_movements_feat.csv',
         'y_1step_per_min_feat.csv', 
         'y_2step_per_min_feat.csv',
         'y_anomaly_detection_3_step_per_min_feat.csv',
         'z_axis_with_random_movements_feat.csv',
         'z_1step_per_min_feat.csv', 
         'z_2step_per_min_feat.csv',
         'z_anomaly_detection_3_step_per_min_feat.csv',
    ]
path = './3_FEATS/'
path1 = './4_FEATS_COMBINED/'
file = 'movement_0_feat.csv'
df = pd.read_csv(path + file)
fNames = df.columns[:].tolist()


for choice in range(0,2):
    for index in range(0,2):
        # X, y, fNames, Data_tag = loadData(path, files, index)
        
        # Paths for X and y (from Stage 4 outputs)
        X_path = os.path.join(path1 , "X_train.csv")
        y_path = os.path.join(path1 , "y_train.csv")
        
        # Load without headers
        X_data = pd.read_csv(X_path, header=None,dtype=np.float32)
        y_data = pd.read_csv(y_path, header=None, dtype=np.int32).squeeze("columns")  # 1D Series
        
        if index == 0:
            # Convert to numpy arrays
            X = X_data.to_numpy(dtype=np.float32)
            y = y_data.to_numpy(dtype=np.int32)
            Data_tag ='ALL FEATURES'
        
        elif index == 1:
            # Convert to numpy arrays
            X = X_data.to_numpy(dtype=np.float32)
            y = y_data.to_numpy(dtype=np.int32)
            
            #column_indices = [ 28,  40, 25, 9, 27, 77, 36, 26, 39,16 ]      
            column_indices = [66,	27,	65,	71,	29,	41,	77,	78,	74,	36]
            column_indices = [i - 1 for i in column_indices]
            X = X[:, column_indices]
            Data_tag ='RELIFF FEATURES 10 Best'
        #============================== TRAIN/TEST SPLIT ==============================
    
        def TrainTestSplit(X, y, train_size, test_size):
    
            X_train_parts = []
            X_test_parts = []
            y_train_parts = []
            y_test_parts = []
    
            for class_label in np.unique(y):
                class_data = X[y == class_label]
                X_train_class, X_test_class = train_test_split(
                    class_data, train_size=train_size, test_size=test_size, shuffle=False
                )
                X_train_parts.append(X_train_class)
                X_test_parts.append(X_test_class)
                y_train_parts.append(np.full(X_train_class.shape[0], class_label))
                y_test_parts.append(np.full(X_test_class.shape[0], class_label))
    
            X_train = np.concatenate(X_train_parts, axis=0)
            X_test = np.concatenate(X_test_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)
            y_test = np.concatenate(y_test_parts, axis=0)
    
            return X_train, X_test, y_train, y_test
        # -------------------------------------------------------------------------
        
        X_train, X_test, y_train, y_test = TrainTestSplit(X, y, train_size=0.7, test_size=0.3)
        
        #================================ CLASSIFICATION ==============================
        
        search = 1
          
        if search == 0: # Hyperparameters Tuning
        
            def classifiers(choice):
                #if choice == 0: # Nearest Centroid
                    #from sklearn.neighbors import KNeighborsClassifier
                    #return KNeighborsClassifier(), 'KNN'
                    
                if choice == 0: # Decision Tree 
                    from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
                    return DecisionTreeClassifier(), 'DecisionTree'
                    
                elif choice == 1: # Random Forest 
                    from sklearn.ensemble import RandomForestClassifier
                    return RandomForestClassifier(), 'RandomForest'     
                    
                #elif choice == 3: # eXtreme Gradient Boosting
                    #from xgboost import XGBClassifier
                    #return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), 'XGBoost'
            
            start = time.time()
            classifier, classifier_name = classifiers(choice)
            classifier = classifier.fit(X_train, y_train)
            end = time.time()
            classification_time = end-start
            y_pred = classifier.predict(X_test)
            
            # Define parameter grids
            param_grids = {
                #'KNN': {'n_neighbors': [3, 5, 7, 9],'weights': ['uniform', 'distance'],'metric': ['euclidean', 'manhattan']},
                'DecisionTree': {
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'RandomForest': {
                    'n_estimators': [5, 50, 100],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                },
                #'XGBoost': {'n_estimators': [50, 100, 200, 300],'max_depth': [3, 5, 7, 10],'learning_rate': [0.01, 0.1, 0.3, 1]}
            }
            
            # Parameter distributions for RandomizedSearchCV
            param_dists = {
                #KNN': {'n_neighbors': randint(3, 15),'weights': ['uniform', 'distance'],'metric': ['euclidean', 'manhattan']},
                'DecisionTree': {
                    'max_depth': randint(1, 20),
                    'min_samples_split': randint(2, 10)
                },
                'RandomForest': {
                    'n_estimators': randint(1, 100),
                    'max_depth': randint(1, 20),
                    'min_samples_split': randint(2, 10)
                }
                #XGBoost': {'n_estimators': randint(50, 300),'max_depth': randint(3, 10),'learning_rate': uniform(0.01, 1)}
            }
            
            ts_cv = TimeSeriesSplit(n_splits=5) 
            from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
            # GridSearchCV or RandomizedSearchCV
            search_type = 1  # 0 for grid or 1 for random
            # Store results
            summary = []
            if search_type == 0:
                search = GridSearchCV(classifier, param_grids[classifier_name], cv=ts_cv, scoring='accuracy')
            else:
                search = RandomizedSearchCV(classifier, param_dists[classifier_name], 
                                            n_iter=50, cv=ts_cv, n_jobs=-1, scoring='accuracy', random_state=42)
            # Fit and get results
            search.fit(X_train, y_train)
            best_params = search.best_params_
            best_cv_score = search.best_score_
            
            # Save parameters
            filename = f"best_params_{classifier_name}.json"
            with open(filename, "w") as f:
                json.dump(best_params, f)
        
            # Test set accuracy
            best_model = search.best_estimator_
            y_pred = best_model.predict(X_test)
            test_score = accuracy_score(y_test, y_pred)
        
            # Store for summary
            summary.append({
                'Classifier': classifier_name,
                'CV Accuracy': round(best_cv_score, 4),
                'Test Accuracy': round(test_score, 4),
                'Best Params': best_params,
                'Saved As': filename
            })
            # Summary
            print("\n Summary:")
            for s in summary:
                print(f"\n {s['Classifier']}")
                print(f"   CV Accuracy  : {s['CV Accuracy']}")
                print(f"   Test Accuracy: {s['Test Accuracy']}")
                print(f"   Best Params  : {s['Best Params']}")
                print(f"   Saved To     : {s['Saved As']}")
                
            word = 0   
            
        elif search == 1:
            
            if choice == 0:
                classifier_name = 'DecisionTree'
            elif choice == 1:  
                classifier_name = 'RandomForest'
            
            def load_classifier(choice, params):

                if choice == 0:
                    from sklearn.tree import DecisionTreeClassifier
                    return DecisionTreeClassifier(**params)
                elif choice == 1:
                    from sklearn.ensemble import RandomForestClassifier
                    return RandomForestClassifier(**params)
            
            json_file = f"best_params_{classifier_name}.json"
            try:
                with open(json_file, "r") as f:
                    params = json.load(f)
                classifier = load_classifier(choice, params)
                start = time.time()
                classifier.fit(X_train, y_train)
                end = time.time()
                classification_time = end-start
                y_pred = classifier.predict(X_test)
            except FileNotFoundError:
                print(f"  Could not find '{json_file}'")                    
                    
            # Export model to c code
            model_code = port(classifier)
            
            # Windows path (use raw string or double backslashes)
            save_dir = '../' 
            
            # Ensure path exists (optional safety check)
            os.makedirs(save_dir, exist_ok=True)
            
            # Full path to save header file
            save_path = os.path.join(save_dir, f"{classifier_name}.h")
            
            # Write the file
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(model_code)
                
            word = 1
                
        #========================== PLOTING AND WORD SAVING ==========================

        accuracy = accuracy_score(y_test, y_pred) * 100
        acc = f"Accuracy: {accuracy:.2f} % "
        #accuracy = print_confusion_matrix(y_test, y_pred)
        print('=' * 23 + ' CLASSIFICATION RESULTS ' + '=' * 23)
        print('CLASSIFIER: ', classifier_name)
        print(acc)
        print('Feature used :', Data_tag)
        print('Classification Time: ', np.asarray(classification_time).round(3),'seconds')
        print(f"Test set X__test shape: {X_test.shape}")
        
        # Helper to simulate console output and capture it
        def capture_output_and_plot(classifier_name, accuracy, Data_tag, 
                                    classification_time, classifier, X_test, y_test):
            buf = io.StringIO()
            with redirect_stdout(buf):
                print('=' * 23 + ' CLASSIFICATION RESULTS ' + '=' * 23)
                print('CLASSIFIER: ', classifier_name)
                print(acc)
                print('Feature used :', Data_tag)
                print('Classification Time: ', np.asarray(classification_time).round(3),'seconds')
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
                    display_labels=[f"Class {i}" for i in labels],       
                    normalize="true",
                    cmap=plt.cm.Blues,                                    # row-wise %
                    xticks_rotation=90,
                    values_format=".1%",  
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
                          + acc , fontsize=16)
                
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
        
        
        # Capture simulated console output and image
        
        
        if word == 0:
            classification_text, image_path = capture_output_and_plot(classifier_name,
                                                                      accuracy, Data_tag, 
                                                                      classification_time, 
                                                                      classifier, 
                                                                      X_test, 
                                                                   y_test)
        else:
            classification_text, image_path = capture_output_and_plot(classifier_name,
                                                                      accuracy, 
                                                                      Data_tag, 
                                                                      classification_time,
                                                                      classifier, 
                                                                      X_test, 
                                                                 y_test)
  
            def append_results_to_doc(doc, classification_text, image_path, 
                                      font_name='Times New Roman', font_size=11):
                paragraph = doc.add_paragraph()
                run = paragraph.add_run(classification_text)
            
                # Set font properties
                font = run.font
                font.name = font_name
                font.size = Pt(font_size)
            
                # Add image
                max_width_in_inches = 6.0  # You can change this to fit your layout
                doc.add_picture(image_path, width=Inches(max_width_in_inches))
                doc.add_paragraph('\n' + '='*70 + '\n')  # Add separation between results
                
      
            append_results_to_doc(
                doc,
                classification_text,
                image_path,
                font_name='Times New Roman',
                font_size=11
                )
            os.remove('temp_image.png') 
 
doc.save(docx_filename)
    

