README

Wearable Posture Detection System, version: 0, author: ACHILLIOS PITTSILKAS

This repository is a project for a wearable system for analyzing and notifying sitting posture using ESP32, 1 GY-521 inertial sensor with accelerometer and MPU6050 gyroscope, using Edge AI and contains:

1) Data from 13 experimental procedures, located in src/data_process_and_classification/ 0_RAW

2) 2 Python codes for processing and 1 MATLAB code, data extraction, selection and training machine learning algorithms,  located in src/data_process_and_classification, named processing_raw_data_.py, ML_workflow.py and reliefF_feature_selection.m.

3) ESP32 test files so that anyone can reproduce the experimental procedures, as well as the wearable device,  located in src.

4) the final ESP32 sitting posture prediction file,  located in src, named main.cpp.