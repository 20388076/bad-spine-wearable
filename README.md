README

Wearable Posture Detection System, version: 0, author: ACHILLIOS PITTSILKAS

This repository is a project for a wearable system for analyzing and notifying sitting posture using ESP32, 1 GY-521 inertial sensor with accelerometer and MPU6050 gyroscope, using Edge AI. 
More specific, based the workflow that it must be following, each program executes these steps:

1) Data from 13 experimental procedures, located in src/data_process_and_classification/ 0_RAW

2) 2 Python codes for processing and 1 MATLAB code, data extraction, selection and training machine learning algorithms,  located in src/data_process_and_classification, named processing_raw_data_.py, ML_workflow.py and reliefF_feature_selection.m.

    -> processing_raw_data.py

    This program processes raw sensor data (accelerometer and gyroscope) from CSV files, cleans it, extracts features, and saves the results.
    It supports 4 main stages of data processing:    
    0. Data Cleaning: Removes unnecessary lines from raw CSV files and saves cleaned data and plots them.
        The results are saved in new CSV files with '_clean' suffix
    1. Data Processing only for 1 and 2 steps per min: Reads cleaned data, filters it based on time, and duplicates sensor data to fill up to the original file length.
        It saves the processed data in new CSV files with '_cln_pr' suffix.
    2. Feature Extraction: Computes various features from the cleaned data: vector magnitudes,
        cubic products, gradients, window-based statistics and features: min, mean, max, signal magnitude area, root mean square,
        mean absolute deviation, variance, standard deviation, interquartile range, fast Fourier transform, and energy.
        The features which computed over defined windows of data with 'window' variable.
        The results are saved in new CSV files with '_feat_prepr' suffix.
    3. FFT Feature Processing: Reads the preprocessed feature files, computes the standard deviation of FFT features across all datasets,
        and replaces the FFT values in each dataset with the value from the line that has the highest standard deviation.
        The results are saved in new CSV files with '_feat' suffix.
    4. Build Machine Learning - ML dataset (X_data, y_data): Combines all feature files into a single dataset, assigns class labels based on file names,
        and saves the combined features and labels into 'X_data.csv' and 'y_data.csv', splits combine data to train - test datasets for ML, with the same 
        pattern to be followed for raw and g converted data.
    5. Feature Selection using ReliefF algorithm: Applies the ReliefF feature selection algorithm to the training dataset,
        selects the top features based on their importance scores, and saves the reduced feature set and feature weights to CSV files.
        The results are saved in 'X_train_reduced_idx_py.csv' and 'Python_relieff_feature_indices_weights.csv'.

    You can skip the stage 5 using the Matlab ReliefF program named: reliefF_feature_selection.m
    by setting the 'matlab' variable to 1. After running the Matlab program, press enter to continue to stage 6.

    6. Plotting the weight order best features and combine the ESP32 computation time.
        The results are saved in 'Relieff_Feature_Weights.png' and 'ESP32_computation_time.png'.
        Also, returns the indices of the 10 best selected weight based features for Edge AI implementation.

    To run the program, set the 'data_process' variable set the variable accordingly to the desired operation:
    - 0 for Data Cleaning
    - 1 for Data Processing
    - 2 for Feature Extraction
    - 3 for FFT Feature Processing
    - 4 for Build ML dataset (X_data, y_data) 
    - 5 for Feature Selection using ReliefF algorithm
    - 6 for Plotting the weight order best features and combine the ESP32 computation time.
    To run all stages sequentially, set 'auto' to 1. To run only one stage, set 'auto' to 0.
    The 'window' variable defines the size of the window for all data processes. 



3) ESP32 test files so that anyone can reproduce the experimental procedures, as well as the wearable device,  located in src.

4) the final ESP32 sitting posture prediction file,  located in src, named main.cpp.


## 📦 Requirements

📎 You can set up the environment either with **Conda** (recommended) or with **pip**.


## ⚙️ Setup with Conda (recommended)

1. Make sure you have [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html) installed.

2. Create the environment: conda env create -f environment.yml

3. Activate the environment running on terminal: conda activate env

4. Select Python Interpreter: Press Ctrl+Shift+P then choose 👉 "Python: Select Interpreter" option and last choose your conda environment

5. Run the Python preprocessing scripts 👍


## ⚙️ Setup with pip (alternative way for non conda friends 🥲 )

1. Make sure you have Python 3.12 installed.
    (⚠️ Python 3.13 may still cause issues with multiprocessing / joblib, matlab.engine on Windows [at least to me it didn't work 😢]).

2. Create a virtual environment running on terminal: python -m venv .venv

3. Activate the environment running on terminal:
    ✒️ On Linux/Mac: source .venv/bin/activate
    ✒️ On Windows (PowerShell): .venv\Scripts\activate

4. Install dependencies: pip install -r requirements.txt

5. Run the Python preprocessing scripts 👍

    
## 📝 Notes

- For Conda users:
  After adding new packages, re-export the environment: conda env export > environment.yml

- For pip users:
  After adding new packages, update: pip freeze > requirements.txt

- On Windows + Python 3.13, you may run into _posixsubprocess / joblib errors or matlab.engine.
  Workaround: use Python 3.12 or 1 - set joblib in the last lines of the code on stage_1 locate backend and set it to "threading" instead of "loky" and 2 - use manual reliefF_feature_selection_manual.m file if you want use MATLAB's ReliefF instead of Python's. 

- (OPTIONAL) if you use a installed version of MATLAB you can use matlab.engine for quick and robust code execution
  
  # For Conda Users

    - You will need to run your Terminal or Anaconda Prompt as Administrator and run these commands:
    
    1. conda activate (Your environment name)
    2. cd "C:\Program Files\MATLAB\(your matlab version e.g R2025b)\extern\engines\python" \\ or your own directory
    3. python setup.py install

  # For Pip Users

    - You will need to run your Terminal as Administrator and run these commands:

    1. cd "C:\Program Files\MATLAB\(your matlab version e.g R2025b)\extern\engines\python" \\ or your own directory
    2. python setup.py install
 

## 🤝 Contributing

1. Fork this repo and create a new branch for your feature or bugfix.

2. Submit a pull request when ready.

3. Please update environment.yml or requirements.txt if you add new dependencies.

