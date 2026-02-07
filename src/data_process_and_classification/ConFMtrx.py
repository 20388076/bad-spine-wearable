# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 13:21:12 2025

@author: AXILLIOS
"""
import os
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
import matplotlib.pyplot as plt

# === CONFIGURATION ===
classifier_names = ['DT','RF']
classifier_name = classifier_names[1]
DATA_PATH = f"0_RAW/series_of_experiments_2/9.71_Hz_sampling/TESTING_{classifier_name}"  # directory with all CSVs
CLASS_MAP = {"good": 0, "mid": 1, "bad": 2}
ITERATIONS_TO_KEEP = 59  # number of iterations after the first
# === FUNCTION TO PARSE A SINGLE FILE ===
def parse_predictions(filepath):
    """
    Reads a file with lines like:
    Iteration 297 - Prediction result: 2
    Returns a DataFrame with columns: iteration, prediction
    """
    data = []
    with open(filepath, "r") as f:
        for line in f:
            if "Iteration" in line and "Prediction result:" in line:
                try:
                    parts = line.strip().split("Prediction result:")
                    iter_part = parts[0].split("Iteration")[-1].split("-")[0].strip()
                    pred_part = parts[1].strip()
                    iteration = int(iter_part)
                    prediction = int(pred_part)
                    data.append((iteration, prediction))
                except (IndexError, ValueError):
                    continue

    if not data:
        return pd.DataFrame(columns=["iteration", "prediction"])

    # Keep first iteration + next 60
    start_iter = data[0][0]
    subset = [x for x in data if start_iter <= x[0] < start_iter + ITERATIONS_TO_KEEP]
    df = pd.DataFrame(subset, columns=["iteration", "prediction"])
    return df


# === MAIN SCRIPT ===
all_data = []

for filename in os.listdir(DATA_PATH):
    if not filename.endswith(".csv"):
        continue

    # Identify class from filename
    for cname in CLASS_MAP.keys():
        if f"{classifier_name}_{cname}_" in filename:
            class_label = CLASS_MAP[cname]
            break
    else:
        continue  # skip if not good/mid/bad

    filepath = os.path.join(DATA_PATH, filename)
    df = parse_predictions(filepath)
    if df.empty:
        continue

    df["true_class"] = class_label
    df["predicted_class"] = df["prediction"]
    all_data.append(df)

# === CONCATENATE ALL DATA ===
if not all_data:
    raise ValueError("No valid datasets found. Check filenames and directory path.")
combined = pd.concat(all_data, ignore_index=True)

# === CONFUSION MATRIX & METRICS ===
cm = confusion_matrix(combined["true_class"], combined["predicted_class"])
acc = accuracy_score(combined["true_class"], combined["predicted_class"])

print("=== CONFUSION MATRIX ===")
print(cm)
print(f"\n✅ Total Accuracy: {acc * 100:.2f}%")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    combined["true_class"],
    combined["predicted_class"],
    target_names=["good (0)", "mid (1)", "bad (2)"]
))

# === VISUAL DISPLAY ===
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["good (0)", "mid (1)", "bad (2)"])
disp.plot(cmap="Blues", values_format="d")
plt.title(f"{classifier_name} Confusion Matrix (Total Accuracy: {acc * 100:.2f}%)")
plt.show()
