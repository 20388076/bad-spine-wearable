print(chr(27) + '[2J') 

import os
import glob
import csv
import re
import argparse
from collections import Counter

try:
    import numpy as np
    import pandas as pd
except Exception:
    raise SystemExit("This script requires numpy and pandas. Install them with pip install numpy pandas")

# optional sklearn usage; fallback to a manual confusion builder if not available
try:
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    _HAS_SK = True
except Exception:
    _HAS_SK = False

import matplotlib.pyplot as plt


MAX_INTERACTIONS = 1846
CLASSES = list(range(15))


def read_file_text(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
        text = fh.read()
    lines = text.splitlines()
    return text, lines


def extract_choice_num_from_csv(filepath):
    """Try to parse the file as CSV and return value at row 11 (index 10), column B (index 1).
       Return None if not found or not an int in range 0..14."""
    try:
        with open(filepath, newline='', encoding='utf-8', errors='replace') as fh:
            reader = csv.reader(fh)
            rows = list(reader)
        if len(rows) >= 11:
            row11 = rows[10]
            if len(row11) >= 2:
                val = row11[1].strip()
                if val != '':
                    m = re.search(r'(-?\d+)', val)
                    if m:
                        v = int(m.group(1))
                        if 0 <= v <= 14:
                            return v
    except Exception:
        pass
    return None


def extract_choice_num_from_text(lines, text):
    # 1) try direct CSV-like row 11 tokens
    if len(lines) >= 11:
        tokens = re.split('[,;\t]', lines[10])
        if len(tokens) >= 2:
            m = re.search(r'(-?\d+)', tokens[1])
            if m:
                v = int(m.group(1))
                if 0 <= v <= 14:
                    return v
    # 2) search for a labelled occurrence like "choice_num: 3" or "Choice Num = 3"
    m = re.search(r'choice[_ ]?num\s*[:=]?\s*(-?\d+)', text, flags=re.IGNORECASE)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 14:
            return v
    # 3) fallback: look for the first small integer (0..14) that appears near the top of the file (first 30 lines)
    top = '\n'.join(lines[:30])
    all_nums = re.findall(r'(-?\d+)', top)
    for n in all_nums:
        v = int(n)
        if 0 <= v <= 14:
            return v
    return None


def extract_predictions_from_text(text, lines):
    preds = []
    # Primary regex: single-line pattern
    # e.g. "Iteration 0/1846 - Prediction result: 10"
    pattern = re.compile(r'Iteration\s*(\d+)\s*/\s*' + str(MAX_INTERACTIONS) + r"\s*[-–—]*\s*Prediction\s*result\s*[:=]?\s*(\d+)", flags=re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        for it, p in matches:
            p_int = int(p)
            if 0 <= p_int <= 14:
                preds.append(p_int)
            if len(preds) >= MAX_INTERACTIONS:
                break
        return preds

    # Secondary strategy: line-by-line
    for i, line in enumerate(lines):
        if 'Iteration' in line and 'Prediction' in line:
            # Typically both iteration and prediction are on same line
            m = re.search(r'Prediction\s*result\s*[:=]?\s*(\d+)', line, flags=re.IGNORECASE)
            if m:
                p_int = int(m.group(1))
                if 0 <= p_int <= 14:
                    preds.append(p_int)
            else:
                # maybe prediction is on the next non-empty line
                j = i + 1
                while j < len(lines) and lines[j].strip() == '':
                    j += 1
                if j < len(lines):
                    m2 = re.search(r'Prediction\s*result\s*[:=]?\s*(\d+)', lines[j], flags=re.IGNORECASE)
                    if m2:
                        p_int = int(m2.group(1))
                        if 0 <= p_int <= 14:
                            preds.append(p_int)
        else:
            # some files may have a single line like: "Iteration 0/1846 - Prediction result: 10"
            m3 = re.search(r'Iteration\s*(\d+)\s*/\s*' + str(MAX_INTERACTIONS) + r".*Prediction.*?(\d+)", line, flags=re.IGNORECASE)
            if m3:
                p_int = int(m3.group(2))
                if 0 <= p_int <= 14:
                    preds.append(p_int)
        if len(preds) >= MAX_INTERACTIONS:
            break

    # Tertiary: a very generic search for lines containing "Iteration" and then any small integer after ':'
    if not preds:
        for line in lines:
            if 'Iteration' in line:
                m = re.search(r'Prediction\s*[:=]?\s*(\d+)', line, flags=re.IGNORECASE)
                if m:
                    p_int = int(m.group(1))
                    if 0 <= p_int <= 14:
                        preds.append(p_int)
            if len(preds) >= MAX_INTERACTIONS:
                break

    return preds


def build_confusion_matrix(pairs, class_labels, classifier_name, project_root, sampleRate, n_classes=15):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --- Build raw confusion matrix ---
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in pairs:
        if 0 <= int(t) < n_classes and 0 <= int(p) < n_classes:
            cm[int(t), int(p)] += 1

    total = cm.sum()
    correct = np.trace(cm)
    acc = correct / total if total > 0 else float("nan")

    print(f"\nTotal predictions = {total}")
    print(f"Correct predictions = {correct}")
    print(f"Overall Accuracy = {acc*100:.2f}%")

    # --- Save raw counts to CSV ---
    df_cm = pd.DataFrame(cm, index=range(n_classes), columns=range(n_classes))
    out_cm_csv = os.path.join(project_root, f"confusion_matrix_counts{sampleRate}.csv")
    df_cm.to_csv(out_cm_csv)
    print("Saved raw confusion matrix to", out_cm_csv)

    # --- Normalize for percentages ---
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # avoid NaNs

    # --- Plot styled heatmap ---
    plt.figure(figsize=(11, 8))
    sns.heatmap(cm_normalized,
            annot=True, fmt=".1%", cmap="Blues",
            xticklabels=[f"Class {i}" for i in range(n_classes)],
            yticklabels=[f"Class {i}" for i in range(n_classes)],
            annot_kws={"size": 8})   # set text size here

    plt.title(f"ESP32 {classifier_name} Confusion Matrix\nALL_DATA: {acc*100:.2f}%")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # Legend with mapping
    legend_text = "\n".join([f"Class {i}: {class_labels[i]}" for i in range(n_classes)])
    plt.gcf().text(0.80, 0.5, legend_text, fontsize=10, va='center',bbox=dict(facecolor='white', edgecolor='black'))

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    out_png = os.path.join(project_root, f"confusion_matrix_{sampleRate}.png")
    plt.savefig(out_png, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved styled confusion matrix image to", out_png)

    return cm, total, acc


def main(sampleRate, classifier_name, project_root=None):
    if project_root is None:
        # assume script lives inside data_process_and_classification
        script_dir = os.path.abspath(os.path.dirname(__file__))
        project_root = os.path.join(script_dir)

    esp_folder = os.path.join(project_root,'0_RAW', '9.71_Hz_sampling', 'esp_classification_results')
    if not os.path.isdir(esp_folder):
        print('ESP results folder not found at:', esp_folder)
        print('Please run this script from inside the data_process_and_classification folder or use --project-root')
        return
    class_labels = {
    0: f"x_1_0mv_{sampleRate}.csv",
    1: f"y_1_0mv_{sampleRate}.csv",
    2: f"z_1_0mv_{sampleRate}.csv",
    3: f"x_2_r_mv_{sampleRate}.csv",
    4: f"y_2_r_mv_{sampleRate}.csv",
    5: f"z_2_r_mv_{sampleRate}.csv",
    6: f"x_3_1st_p_min_{sampleRate}.csv",
    7: f"y_3_1st_p_min_{sampleRate}.csv",
    8: f"z_3_1st_p_min_{sampleRate}.csv",
    9: f"x_4_2st_p_min_{sampleRate}.csv",
    10: f"y_4_2st_p_min_{sampleRate}.csv",
    11: f"z_4_2st_p_min_{sampleRate}.csv",
    12: f"x_5_3st_p_min_w_ad_{sampleRate}.csv",
    13: f"y_5_3st_p_min_w_ad_{sampleRate}.csv",
    14: f"z_5_3st_p_min_w_ad_{sampleRate}.csv",
}
    patterns = [os.path.join(esp_folder, '*.csv'), os.path.join(esp_folder, '*.txt'), os.path.join(esp_folder, '*.log')]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(files)

    if not files:
        print('No result files found in', esp_folder)
        return

    print('Found', len(files), 'files. Parsing...')

    all_pairs = []
    summary = []
    for f in files:
        text, lines = read_file_text(f)
        choice = extract_choice_num_from_csv(f)
        if choice is None:
            choice = extract_choice_num_from_text(lines, text)
        preds = extract_predictions_from_text(text, lines)

        if choice is None:
            print(f'  SKIP {os.path.basename(f)}: could not determine choice_num (B11).')
            continue
        if not preds:
            print(f'  SKIP {os.path.basename(f)}: no "Iteration ... Prediction result" lines found.')
            continue

        # limit to MAX_INTERACTIONS
        preds = preds[:MAX_INTERACTIONS]
        for p in preds:
            all_pairs.append((choice, int(p)))
        summary.append((os.path.basename(f), choice, len(preds)))
        print(f'  OK   {os.path.basename(f)}  choice={choice}  extracted={len(preds)}')

    print('\nFiles processed (basename, choice_num, n_preds):')
    for s in summary:
        print('  ', s)

    if not all_pairs:
        print('\nNo (true,pred) pairs were extracted from any file. Stopping.')
        return

    cm, total, acc = build_confusion_matrix(all_pairs, class_labels, classifier_name, esp_folder, sampleRate, n_classes=len(CLASSES))

    # Save confusion matrix
    df_cm = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)

    # Per-class counts and accuracy
    row_sums = df_cm.sum(axis=1)
    correct = np.diag(df_cm.values)
    accuracy = []
    for i in range(len(CLASSES)):
        total = int(row_sums.iloc[i])
        corr = int(correct[i])
        acc = float(corr) / total if total > 0 else float('nan')
        accuracy.append({'class': i, 'total': total, 'correct': corr, 'accuracy': f"{acc*100:.2f}" if total > 0 else "nan"})
    df_acc = pd.DataFrame(accuracy)
    out_acc_csv = os.path.join(esp_folder, f'{sampleRate}_per_class_accuracy.csv')
    df_acc.to_csv(out_acc_csv, index=False)
    print('\nSaved per-class accuracy to', out_acc_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=str, default=None,
                        help='Path to data_process_and_classification folder (default: script folder)')
    args = parser.parse_args()
    sampleRate = 9.71 # Sample rate in Hz     <-- Change this value to set sample rate
    
    classifier_name = ['DecisionTree',
                       'RandomForest'
        ]
    
    main(sampleRate, classifier_name[0], args.project_root)

