"""
Created on Wed Apr 15 17:45:10 2026

@author: AXILLIOS
"""
# -------------------------------  Working Directory -----------------------------
# Set the working directory to the script's location if running in Visual Studio Code
import os
# Change working directory for this script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# ------------------------------- Built modules ----------------------------------

from stages.stage_0 import stage_0
from stages.stage_1 import stage_1
from stages.stage_2 import stage_2
from stages.stage_3 import stage_3
from stages.stage_4 import stage_4
from stages.stage_5 import stage_5
from utils import UtilFunc as UF
from utils.copy_outputs_between_classifiers import copy_outputs_between_classifiers

# ============================= MAIN PROGRAM ==================================
UF.cls() #create a clear console for better readability of the results

series_of_experiments = 2 # we have execute 2 different experiments so it's taking 1 or 2 as input values
# ------------------------------ Auto Runner Option ---------------------------
# 0: run only one stage; 1: run all stages
auto = 0
# ------------------------------ Data Process Option --------------------------
# 0: raw -> clean;  
# 1: clean -> features preprocessed; 
# 2: features preprocessed -> features final; 
# 3: features final -> X_data, y_data; 
# 4: a) X_data_train, y_data_train -> ReliefF selected features or only 
#    b) Plotting the weight order best features and combine the ESP32 computation time.
stage = 0
# ----------------------------- Matlab Option for ReleifF ---------------------
matlab = 1 # 0: python ReleifF ; 1: Matlab ReleifF   
# ----------------------------- Window Size -----------------------------------
# Define window in sec for data trimming to fit window size and window-based features per classifier
windows = [1,1] # sec  IF window_search = 0 <-- Change this table to set time window per classifier
# ----------------------------- Window Search Value -----------------------------------
window_search = 0 # Set 1 to search for the best time window from a list of time window named candidate_windows
candidate_windows = [0.8,1,2,4,6,8,10]  # in sec 
# ----------------------------- Sample Rate Dataset ---------------------------
# Available Datasets
# 1) 9.71 Hz
# 2) 10 Hz
# 3) 50 Hz
sampleRate = 9.71 # Sample rate in Hz     <-- Change this value to set sample rate
# ----------------------------- Classifier Factory ----------------------------
def get_classifier(cl):
    if cl == 0: # Decision Tree 
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(), 'DT'
    elif cl == 1:  # Random Forest 
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(), 'RF'
    
# if you add more classifiers change this value based on the overall number of classifiers
n_classifiers = 2  # DecisionTree, RandomForest

# ============================= Auto Runner ===================================
import pandas as pd, matplotlib.pyplot as plt

def run_pipeline(window,window_size,sampleRate,classifier_name,series_of_experiments,plot_mode,matlab,modes):
    for i in range(0, len(modes)):
        stage_0(window,window_size,sampleRate,classifier_name,series_of_experiments,plot_mode,mode=modes[i])
    for i in range(0, len(modes)):
        stage_1(window,window_size,sampleRate,classifier_name,series_of_experiments,mode=modes[i])
    for i in range(0, len(modes)):
        stage_2(window,window_size,sampleRate,classifier_name,series_of_experiments,mode=modes[i])
    for i in range(0, len(modes)):
        stage_3(window,window_size,sampleRate,classifier_name,series_of_experiments,mode=modes[i])
    stage_4(window,window_size,sampleRate,classifier_name,series_of_experiments,plot_mode,matlab)
    return

modes = ['exp', 'test']
if auto == 0: 
    # ----------------------------- Plotting Option ---------------------------
    plot_mode = 1 # 0: no plots; 1: plots
    file_index = [0,0]
    model_create = 1  # enable classifier model create
    for cl in range(n_classifiers):
        window = windows[cl]
        window_size = int(round(window * sampleRate))
        print(f'Window size set to {window_size} rows for sample rate {sampleRate} Hz and window time {window} sec.')
        classifier, classifier_name = get_classifier(cl)
        if stage == 0: 
            for i in range(0,len(modes)):
                stage_0(window,window_size,sampleRate,classifier_name,series_of_experiments,plot_mode,mode = modes[i])
        elif stage == 1: 
            for i in range(0,len(modes)):
                stage_1(window,window_size,sampleRate,classifier_name,series_of_experiments,mode = modes[i])
        elif stage == 2:
            for i in range(0,len(modes)):
                stage_2(mode = modes[i])
        elif stage == 3: 
            for i in range(0,len(modes)):
                stage_3(mode = modes[i])
        elif stage == 4: stage_4()
        elif stage == 5:     
            test_score, Data_tag = stage_5(cl, file_index[cl])
            print(f"Final {Data_tag}: {classifier_name} accuracy: {test_score*100:.2f}%")


elif auto == 1 and window_search == 0:
    model_create = 1  # enable classifier model create
    file_index = 0
    plot_mode = 1 # 0: no plots; 1: plots
    
    same_window = (len(windows) >= 2 and all(w == windows[0] for w in windows[:n_classifiers]))

    if same_window and n_classifiers >= 2:
        # ---------- Run stages 0-4 only for cl=0 ----------
        cl0 = 0
        window = windows[cl0]
        window_size = int(round(window * sampleRate))
        print(f"[FAST] Same window for all classifiers: {window}s ({window_size} rows).")
        print(f'Window size set to {window_size} rows for sample rate {sampleRate} Hz and window time {window} sec.')

        classifier, classifier_name = get_classifier(cl0)

        run_pipeline(window,window_size,sampleRate,classifier_name,series_of_experiments,plot_mode,matlab,modes)

        # ---------- Copy Stage 0-4 outputs to the other classifiers ----------
        src_name = classifier_name
        for cl in range(1, n_classifiers):
            _, dst_name = get_classifier(cl)
            copy_outputs_between_classifiers(sampleRate, src_name, dst_name, modes)

        # ---------- Now runing stage 5 per classifier (training/eval/export) ----------
        for cl in range(n_classifiers):
            window = windows[cl]
            window_size = int(round(window * sampleRate))
            classifier, classifier_name = get_classifier(cl)
            test_score, Data_tag = stage_5(window,window_size,window_search,sampleRate,classifier_name,series_of_experiments,plot_mode,matlab,classifier,file_index,auto,model_create)
            print(f"Final {Data_tag}: {classifier_name} accuracy: {test_score*100:.2f}%")
    else:
        for cl in range(n_classifiers):
            window = windows[cl]
            window_size = int(round(window * sampleRate))
            print(f'Window size set to {window_size} rows for sample rate {sampleRate} Hz and window time {window} sec.')
            classifier, classifier_name = get_classifier(cl)
            run_pipeline(window,window_size,sampleRate,classifier_name,series_of_experiments,plot_mode,matlab,modes)
            #for file_index in range(0,5):
            test_score, Data_tag = stage_5(window,window_size,window_search,sampleRate,classifier_name,series_of_experiments,plot_mode,matlab,classifier,file_index,auto,model_create)
            print(f"Final {Data_tag}: {classifier_name} accuracy: {test_score*100:.2f}%")
    
elif auto == 1 and window_search == 1:
    
    model_create = 0  # set =1 if you want micromlgen export too
    plot_mode = 0  # disable plotting during search
    # results storage
    results_df = pd.DataFrame(columns=['Classifier','Window','File_Index','Accuracy'])
    # track best window (only file_index=0 in search loop)
    best_windows = {cl: {'score': -1.0, 'window': None} for cl in range(n_classifiers)}
    
    # ---------------- Main Search (file_index = 0 only) ----------------
    for w in range(len(candidate_windows)):
        window = candidate_windows[w]
        window_size = int(round(window * sampleRate))
        print(f"\n=== Running pipeline for window={window}s ({window_size} rows) ===")
        
        # run all stages once per window
        classifier_name = 'ALL'
        run_pipeline(window,window_size,sampleRate,classifier_name,series_of_experiments,plot_mode,matlab,modes)
        for cl in range(n_classifiers):
            classifier, classifier_name = get_classifier(cl)
            
            # always use file_index=0 for search
            file_index = 0
            test_score, Data_tag = stage_5(cl, file_index)            
            acc = test_score * 100
            print(f"   {Data_tag}: {classifier_name} accuracy: {acc:.2f}%")
            
            # store results
            results_df.loc[len(results_df)] = [classifier_name, window, file_index, acc]
            
            # update best window
            if acc > best_windows[cl]['score']:
                best_windows[cl]['score'] = acc
                best_windows[cl]['window'] = window

    print("\n================= Window search finished =================")
    for cl in range(n_classifiers):
        clf_name = get_classifier(cl)[1]
        best = best_windows[cl]
        print(f"Best window for {clf_name}: {best['window']}s "
              f"(accuracy={best['score']:.2f}% with file_index=0)")
    
    # ---------------- Plot search results ----------------
    plt.figure(figsize=(10,6))
    colors = plt.cm.tab10.colors

    for cl in range(n_classifiers):
        clf_name = get_classifier(cl)[1]
        subset = results_df[results_df['Classifier']==clf_name]
        grouped = subset.groupby('Window')['Accuracy'].mean().reset_index()
        color = colors[cl % len(colors)]
        
        plt.plot(grouped['Window'], grouped['Accuracy'], marker='o', color=color, label=clf_name)
        
        # mark best window (from search with file_index=0)
        best = best_windows[cl]
        plt.scatter(best['window'], best['score'], marker='*', s=200, 
                    color=color, edgecolor='black', zorder=5)

    plt.xlabel("Window (s)")
    plt.ylabel("Accuracy (%)")         
    plt.title("Classifier Accuracy vs Window Size (file_index=0)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results_window_search.png", dpi=600)
    plt.show()

    # ---------------- Re-run with best window & best file_index ----------------
    plot_mode = 0  # enable plots
    model_create = 0  # set =1 if you want micromlgen export too

    best_final_indices = {}  # store best file_index per classifier

    for cl in range(n_classifiers):
        clf_name = get_classifier(cl)[1]
        best_window = best_windows[cl]['window']
        
        best_final = {'score': -1.0, 'file_index': None}
        window = best_window
        window_size = int(round(window * sampleRate))
        
        print(f"\nTesting {clf_name} with window={window}s and file_index={file_index}")
        classifier, classifier_name = get_classifier(cl)
        
        run_pipeline(window,window_size,sampleRate,classifier_name,series_of_experiments,plot_mode,matlab,modes)
        
        for file_index in range(3,5):
            test_score, Data_tag = stage_5(cl, file_index)
            acc = round(test_score * 100, 2)
            print(f"   {Data_tag}: {clf_name} accuracy={acc:.2f}%")

            # log each re-run
            results_df.loc[len(results_df)] = [clf_name, window, str(file_index), acc]

            if acc > best_final['score']:
                best_final['score'] = acc
                best_final['file_index'] = file_index

        # final best combination
        print(f"\n>>> Final BEST for {clf_name}: "
              f"window={best_window}s, file_index={best_final['file_index']}, "
              f"accuracy={best_final['score']:.2f}%")

        # save best file_index for use in final run
        best_final_indices[cl] = best_final['file_index']

        # store final best row with actual file_index (not a string tag)
        results_df.loc[len(results_df)] = [clf_name, best_window, str(best_final['file_index']), best_final['score']]
        
    # ---------------- Final run with best (window, file_index) ----------------
    print("\n================= Final Run with Best Parameters =================")
    plot_mode = 1   # force plotting for final best run
    model_create = 1  # set to 1 if you also want micromlgen C export

    for cl in range(n_classifiers):
        clf_name = get_classifier(cl)[1]
        best_window = best_windows[cl]['window']
        file_index = best_final_indices[cl]   # directly from re-run

        print(f"\n>>> Last run for {clf_name} with window={best_window}s and file_index={file_index}")
        classifier, classifier_name = get_classifier(cl)
        
        run_pipeline(window,window_size,sampleRate,classifier_name,series_of_experiments,plot_mode,matlab,modes)
        
        test_score, Data_tag = stage_5(cl, file_index)
        acc = round(test_score * 100, 2)
        print(f"Final run {Data_tag}: {clf_name} accuracy={acc:.2f}%")

        # add explicit last run row
        results_df.loc[len(results_df)] = [clf_name, best_window, Data_tag, acc]

    # Save updated results
    results_df.to_csv("all_results_classifier_windows.csv", index=False)
    print("\nSaved updated results (with last run) to all_results_classifier_windows.csv")


