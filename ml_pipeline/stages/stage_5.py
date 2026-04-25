"""
Created on Wed Apr 15 12:24:12 2026

@author: AXILLIOS
"""
# ---------------- Data Process 5: ReliefF Feature Selection Plotting and 10 best features, displaying for ESP32 use  ----------------
def stage_5(window,window_size,window_search,sampleRate,classifier_name,series_of_experiments,plot_mode,matlab,classifier,file_index,auto,model_create,scale):
    print(f'\n======= Data Process: 5 for {window}s  =======\n')

    # ---------------- Import Libraries ------------------ 
    import os
    import pandas as pd, numpy as np, matplotlib.pyplot as plt, multiprocessing
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.model_selection import  RandomizedSearchCV
    from scipy.stats import randint
    from micromlgen import port
    from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay 
    from sklearn.utils import shuffle
    # ----------------------------------------------------

    # ---------------- Helper: loadData ------------------
    def loadData(paths, index, input_file):
        """
        Load dataset X, y, feature names, and Data_tag based on index row of input_file.
        returns: X (DataFrame or np.array), y (1D np.array of ints), fNames (list), Data_tag (str)
        """
        row = input_file.iloc[index]
        X_file, y_file, Data_tag = row['X_file'], row['y_file'], row['Data_tag']

        # First 3 datasets are from Stage 4, rest from Stage 6
        base_path = paths[0] if index < 3 else paths[1]

        X_path = os.path.join(base_path, X_file)
        y_path = os.path.join(paths[0], y_file)  # y files appear stored in paths[0] per original code

        if not os.path.exists(X_path):
            raise FileNotFoundError(f"X file not found: {X_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"y file not found: {y_path}")

        # Load X as DataFrame (preserve column names)
        X_df = pd.read_csv(X_path, dtype=np.float32)
        fNames = X_df.columns.tolist()

        # Load y
        y_df = pd.read_csv(y_path, header=None)
        # If first row is textual header, drop it
        if isinstance(y_df.iloc[0, 0], str):
            y_df = y_df.iloc[1:].reset_index(drop=True)
        y = y_df.squeeze("columns").astype(np.int32).to_numpy()
        return X_df, y, fNames, Data_tag
    # ----------------------------------------------------
    
    def capture_output_and_plot(classifier_name, accuracy, cv_accuracy, Data_tag, 
                                classifier, X_test, y_test, plot_name, normalize_display=True):
        import io
        from contextlib import redirect_stdout  
        # Original class names from filenames
        
        if series_of_experiments == 1:
            original_class_names   = [f'exp_1_{sampleRate}',
                                        f'exp_2_{sampleRate}',
                                        f'exp_3_{sampleRate}', 
                                        f'exp_4_{sampleRate}', 
                                        f'exp_5_{sampleRate}', 
            ]
        else:
            original_class_names = [f'good_{sampleRate}',f'mid_{sampleRate}',f'bad_{sampleRate}']

        accuracy = str(round(accuracy * 100, 2))
        cv_accuracy = str(round(cv_accuracy  * 100, 2))
        buf = io.StringIO()
        with redirect_stdout(buf):
            # Plot confusion matrix image as an example 
            # Original class names from filenames
            
            # Generate numeric labels for display
            class_names = [f'Class {i}' for i in range(len(original_class_names))]
            
            # Create a mapping legend
            legend_text = '\n'.join([f'{class_names[i]}: {original_class_names[i]}' 
                                     for i in range(len(class_names))])
            normalize_mode = 'true' if normalize_display else None
            value_format = '.1%' if normalize_display else 'd'
            labels = np.unique(y_test)  # integers that appear in y_test
            fig, ax = plt.subplots(figsize=(10, 8))  # wider & taller
            disp = ConfusionMatrixDisplay.from_estimator(
                classifier,
                X_test,
                y_test,
                labels=labels,                                      
                display_labels=[f'Class {i}' for i in labels],       
                normalize=normalize_mode,
                cmap=plt.cm.Blues,
                xticks_rotation=90,
                values_format=value_format,  
                ax=ax            
                )
            
            # Add the legend as a textbox
            plt.gcf().text(1.02, 
                           0.5, 
                           legend_text, 
                           fontsize=12, 
                           va='center', 
                           bbox=dict(facecolor='white', edgecolor='black')
                           )
           
            # Title
            plt.title(f'Series-{series_of_experiments} {classifier_name} Window:{window}sec Confusion Matrix\n' + Data_tag 
                       + accuracy + '%' + ' with CV accuracy: ' + cv_accuracy + '%',  fontsize=16)
            
            # Improve readability
            plt.tick_params(axis='x', labelsize=10)
            plt.tick_params(axis='y', labelsize=10)
            
            # Optional: Bold larger numbers or set font size
            for text in disp.ax_.texts:
                text.set_fontsize(13)  # Increase for better visibility (try 10–14 if need 
            # Save and show
            plt.savefig(plot_name, dpi=600, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
            plt.close()
    
        return buf.getvalue(), '{series_of_experiments}{classifier_name}_image.png'
    
    # fix classifier_name1 depending on auto/window_search flags (these are globals in your environment)
    if auto == 1 and window_search == 1:
        classifier_name1 = 'ALL'
    else:
        classifier_name1 = classifier_name 

    paths = [
        f'../data/4_FEATS_COMBINED/{sampleRate}_Hz_sampling/{classifier_name1}/',
        f'../data/5_FEATS_SELECTION/{sampleRate}_Hz_sampling/{classifier_name1}/'
    ]

    input_file_train = pd.DataFrame([
        [f"X_data_{sampleRate}{classifier_name1}.csv", f"y_data_{sampleRate}{classifier_name1}.csv", "ALL_FEATURES "],
        [f"all_raw_data_{sampleRate}{classifier_name1}.csv", f"y_all_raw_data_{sampleRate}{classifier_name1}.csv", "RAW_DATA "],
        [f"all_norm_data_{sampleRate}{classifier_name1}.csv", f"y_all_norm_data_{sampleRate}{classifier_name1}.csv", "G_RAW_DATA "],
        [f"{series_of_experiments}Matlab_X_data_weight_ordered_{sampleRate}{classifier_name1}.csv", f"y_data_{sampleRate}{classifier_name1}.csv", "WEIGHT BASED FEATURES "],
        [f"{series_of_experiments}Matlab_X_data_custom_reordered_{sampleRate}{classifier_name1}.csv", f"y_data_{sampleRate}{classifier_name1}.csv", "SCORE BASED FEATURES "]
    ], columns=['X_file', 'y_file', 'Data_tag'])

    input_file_test = pd.DataFrame([
        [f"TEST_X_{sampleRate}{classifier_name1}.csv", f"TEST_y_{sampleRate}{classifier_name1}.csv", "ALL_FEATURES "],
        [f"TEST_all_raw_data_{sampleRate}{classifier_name1}.csv", f"TEST_y_all_raw_data_{sampleRate}{classifier_name1}.csv", "RAW_DATA "],
        [f"TEST_all_norm_data_{sampleRate}{classifier_name1}.csv", f"TEST_y_all_norm_data_{sampleRate}{classifier_name1}.csv", "G_RAW_DATA "],
        [f"{series_of_experiments}Matlab_TEST_X_weight_ordered_{sampleRate}{classifier_name1}.csv", f"TEST_y_{sampleRate}{classifier_name1}.csv", "WEIGHT BASED FEATURES "],
        [f"{series_of_experiments}Matlab_TEST_X_custom_reordered_{sampleRate}{classifier_name1}.csv", f"TEST_y_{sampleRate}{classifier_name1}.csv", "SCORE BASED FEATURES "]
    ], columns=['X_file', 'y_file', 'Data_tag'])
    
    X_train, y_train, fNames, Data_tag = loadData(paths, file_index, input_file_train)
    X_test, y_test, fNs, Data_tag = loadData(paths, file_index, input_file_test)
        
    if file_index in (3, 4):
        # If X_train is a DataFrame (recommended), use .iloc; otherwise handle numpy arrays. personaly i use only dataframes in this whole setup:
       
        feats  = 3 # 

        if hasattr(X_train, 'iloc'):
            X_train = X_train.iloc[:, :feats].copy()
            X_test  = X_test.iloc[:, :feats].copy()
            
        else:
            X_train = X_train[:, :feats].copy()
            X_test  = X_test[:, :feats].copy()    

        # Keep feature names as a list of the first 10 names
        fNames = fNames[:feats]
    
    if series_of_experiments == 2:
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        print('\n*** Shuffle is setted on ***\n')
    
    if scale == 1:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # Fit only on training data
        scaler.fit(X_train)
        # Fit on training data and transform both train/test
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Convert scaled arrays back to DataFrames with same column names
        X_train = pd.DataFrame(X_train_scaled, columns=fNames)
        X_test = pd.DataFrame(X_test_scaled, columns=fNames)
        print('\n*** Data are scaled with Z-score ***\n')

    # Parameter distributions for RandomizedSearchCV
    if classifier_name == 'DT':
        classifier.set_params(criterion = 'gini', random_state = 42, splitter = 'best', min_impurity_decrease = 0.01) 
        n_iter=100
        
    elif classifier_name == 'RF':
        classifier.set_params(criterion = 'gini', random_state = 42, min_impurity_decrease = 0.001)
        n_iter = 50 if series_of_experiments == 1 else 15
        
    max_depth_range = randint(1, 5) if series_of_experiments == 1 else randint(1, 4)
    
    param_dists = {
        'DT': {
            'max_depth': max_depth_range,
            'min_samples_split': randint(2, 6),
            'max_features': randint(1, 50)
            
        },
        'RF': {
            'n_estimators': randint(3, 100),
            'max_depth': max_depth_range,
            'min_samples_split': randint(2, 3),
            'max_features': randint(1, 50)
        }
    }
    
    if series_of_experiments == 1:
        cv =  TimeSeriesSplit(n_splits=5)
    else:
        cv = 5
        
    #cv = 5
    if classifier_name == 'DT':
        classifier.set_params(criterion = 'gini', random_state = 42, splitter = 'best', min_impurity_decrease = 0.01) 
        n_iter=100
        
    elif classifier_name == 'RF':
        classifier.set_params(criterion = 'gini', random_state = 42, min_impurity_decrease = 0.005)
        n_iter = 50 if series_of_experiments == 1 else 15
     
    search = RandomizedSearchCV(classifier, 
                             param_dists[classifier_name], 
                             n_iter = n_iter, 
                             cv=cv, 
                             n_jobs=max(1, multiprocessing.cpu_count() - 1), 
                             scoring='accuracy', 
                             random_state=42,
                             verbose=1)
     
    # Fit and get results
    search.fit(X_train, y_train)
    #results = pd.DataFrame(search.cv_results_)
    best_params = search.best_params_
    best_cv_score = search.best_score_
     
    # Test set accuracy
    best_model = search.best_estimator_
     
    #print(best_model.feature_names_in_)
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
        
                                                                                                                                         
    if model_create == 1:                                                                      
        import joblib
        if scale == 1:
            in_file = os.path.join('../', f"{series_of_experiments}scaler_params{classifier_name}{window}.h")
            with open(in_file, "w") as f:
                f.write("#pragma once\n\n")
                f.write(f"const int SCALER_SIZE = {len(scaler.mean_)};\n")
                f.write("const float SCALER_MEAN[] = {" + ", ".join(f"{v:.3f}" for v in scaler.mean_) + "};\n")
                f.write("const float SCALER_SCALE[] = {" + ", ".join(f"{v:.3f}" for v in scaler.scale_) + "};\n")
            sc = 'scaled'
            print("Scaler parameters exported to include/scaler_params.h")

        else:
            sc = ''
        # Save the trained model
        model_save_dir = '../output/pc_classification/models/' 
        
        # Ensure path exists
        os.makedirs(model_save_dir, exist_ok=True)
        model_path = os.path.join(model_save_dir,f"{series_of_experiments}BEST{classifier_name}W{window}F{file_index}{sc}.pkl")
        model_bundle = best_model
        joblib.dump(model_bundle, model_path)
        print(f"Saved best model to {model_path}")  
                                                                 
        # Export model to c code
        model_code = port(best_model)
    
        cl_save_dir = '../lib/classifiers/' 
        
        # Ensure path exists
        os.makedirs(cl_save_dir, exist_ok=True)
        
        # Full path to save header file
        save_path = os.path.join(cl_save_dir, f'{series_of_experiments}{classifier_name}{sampleRate}W{window}{sc}.h')
        
        # Write the file
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(model_code) 
    if plot_mode == 1:
        plot_path = f"../output/pc_classification/plots/classification_results/{classifier_name}/"
        os.makedirs(plot_path, exist_ok=True)
        classification_text, image_path = capture_output_and_plot(classifier_name, 
                                                    train_score, 
                                                    best_cv_score,
                                                    Data_tag, 
                                                    best_model, 
                                                    X_train,
                                                    y_train, 
                                                    os.path.join(plot_path,f'{series_of_experiments}{classifier_name}_train_image_{sc}.png'), 
                                                    normalize_display=False)
        classification_text, image_path = capture_output_and_plot(classifier_name, 
                                                    test_score,
                                                    best_cv_score,
                                                    Data_tag, 
                                                    best_model, 
                                                    X_test, 
                                                    y_test, 
                                                    os.path.join(plot_path,f'{series_of_experiments}{classifier_name}_test_image_{sc}.png'), 
                                                    normalize_display=False) 
    return test_score, Data_tag
