# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 12:08:02 2025

@author: AXILLIOS
"""
# -------------------------------  Working Directory -----------------------------
# Set the working directory to the script's location if running in Visual Studio Code
import os
# Change working directory for this script
working_directory = r'C:\Users\user\OneDrive - MSFT\PlatformIO\PlatformIO\Projects\bad-spine-wearable-1\src\data_process_and_classification' 
os.chdir(working_directory) # modify this path to your working directory

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
# A wrapper that:
# 1. Loads the uploaded processing_raw_data.py
# 2. For each window size from 1s to 120s (inclusive), it will modify the script to set auto=0 and window={w}
# 3. Exec the modified code inside an isolated namespace, call stage_0..stage_5 if available (capture stdout)
# 4. Parse the printed Test Accuracy from stage_5 output (if present) and store results
# 5. After looping, identify the window with max accuracy and re-run the pipeline for that window (saving outputs)
# The environment may not contain the required CSV data files; in that case the attempts will fail and be skipped.
# Results will be printed and the modified script will be saved to /mnt/data/processing_raw_data_multiwindow.py

import io, os, re, traceback, textwrap, sys
from contextlib import redirect_stdout
import pandas as pd

INPUT = './processing_raw_data.py'
OUTPUT = './processing_raw_data_multiwindow.py'

# Read original script
with open(INPUT, 'r', encoding='utf-8') as f:
    orig = f.read()

# Basic safety modifications to avoid the original auto-run behavior and to make the script re-usable.
if "auto = 1" in orig:
    orig_safe = orig.replace("auto = 1", "auto = 0")
else:
    orig_safe = orig

# We'll save a reusable modified script template (with the auto=0 change)
with open(OUTPUT, 'w', encoding='utf-8') as f:
    f.write(orig_safe)

# Function to run the pipeline for given window seconds
def run_for_window(window_sec):
    results = {'window_sec': window_sec, 'success': False, 'error': None, 'test_accuracy': None, 'stdout': ''}
    try:
        # Prepare modified code: set window = <window_sec> and recompute window_size where appropriate.
        code = orig_safe
        # Replace the explicit "window = ..." assignment near top if it exists.
        code = re.sub(r'window\s*=\s*\d+(\.\d+)?', f'window = {window_sec}', code, count=1)
        # Ensure window_size line is present and recalculated after window assignment - we'll force it.
        if 'window_size = int(round(window * sampleRate))' not in code:
            # append calculation
            code = code + '\nwindow_size = int(round(window * sampleRate))\nprint(f"[INFO] window_size computed: {window_size}")\n'
        else:
            # leave as-is (it will compute at exec time with the new window)
            pass
        
        # Exec in an isolated namespace
        ns = {}
        # capture stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(compile(code, '<string>', 'exec'), ns, ns)
            # After import, call stage functions if present
            # prefer to run stages 0..5 if defined
            for st in range(0,6):
                fn = ns.get(f'stage_{st}', None)
                if callable(fn):
                    try:
                        print(f"[RUN] Calling stage_{st} for window {window_sec}s ...")
                        fn()  # run the stage
                    except SystemExit:
                        # Stage can call sys.exit(); capture and continue
                        print(f"[WARN] Stage_{st} called SystemExit; continuing to next window.")
                        raise
                    except Exception as e:
                        print(f"[ERROR] Exception within stage_{st} for window {window_sec}s:\n{traceback.format_exc()}")
                        # do not re-raise; record error and break
                        results['error'] = f"Exception in stage_{st}: {e}"
                        break
                else:
                    print(f"[SKIP] stage_{st} not found in namespace.")
        out = buf.getvalue()
        results['stdout'] = out
        # Try to parse Test Accuracy reported in stage_5 prints like 'Test Accuracy: 0.95' or percentages.
        # Look for lines containing 'Test Accuracy' and extract a float
        m = re.search(r'Test Accuracy[:\s]+([0-9]*\.?[0-9]+)', out)
        if m:
            results['test_accuracy'] = float(m.group(1))
            results['success'] = True
        else:
            # Also try percent form like 'Test Accuracy: 95.00%'
            m2 = re.search(r'Test Accuracy[:\s]+([0-9]*\.?[0-9]+)\s*%', out)
            if m2:
                results['test_accuracy'] = float(m2.group(1)) / 100.0
                results['success'] = True
            else:
                # no accuracy found, but maybe stage_5 printed summary block. Try to find 'Test Accuracy' with text surroundings.
                if 'Test Accuracy' in out:
                    results['success'] = True
                else:
                    results['success'] = False
        return results
    except Exception as e:
        results['error'] = traceback.format_exc()
        return results

# Define window list: from 1s to 120s inclusive (step 1s). To limit runtime, we'll try first a smaller subset then expand.
windows = list(range(1, 121))  # 1..120 seconds

summary = []
# Run loop (this may fail quickly if files are missing; that's captured)
for w in windows:
    print(f'=== Running pipeline for window = {w} s ===')
    res = run_for_window(w)
    summary.append(res)
    # If run succeeded and found an accuracy, log it.
    if res['success'] and res['test_accuracy'] is not None:
        print(f"-> Window {w}s: test_accuracy = {res['test_accuracy']:.4f}")
    else:
        print(f"-> Window {w}s: success={res['success']}, error={'yes' if res['error'] else 'no'}")

# Build a DataFrame summary and save
df_summary = pd.DataFrame(summary)
df_summary.to_csv('./window_search_summary.csv', index=False)

# Find best window by test_accuracy (skip None)
valid = df_summary[df_summary['test_accuracy'].notnull()]
if not valid.empty:
    best_row = valid.loc[valid['test_accuracy'].idxmax()]
    best_window = int(best_row['window_sec'])
    best_accuracy = float(best_row['test_accuracy'])
    best_info = best_row.to_dict()
    # Re-run pipeline for best window and save its stdout
    print(f'\n*** Best window found: {best_window}s with test_accuracy = {best_accuracy:.4f}. Re-running pipeline for that window. ***\n')
    final_res = run_for_window(best_window)
    with open('./best_window_run_stdout.txt', 'w', encoding='utf-8') as f:
        f.write(final_res['stdout'] or '')
    # Save final run result
    final_res['rerun_for_best'] = True
else:
    best_window = None
    best_accuracy = None
    final_res = None
    print('\n*** No valid test_accuracy values were found in any run. See /mnt/data/window_search_summary.csv for details. ***\n')

# Save the modified script template for user's inspection
print(f'Modified script template saved to: {OUTPUT}')
print('Summary table saved to window_search_summary.csv')
if best_window is not None:
    print('Best run stdout saved to best_window_run_stdout.txt')

# Display a small summary DataFrame (first 20 rows)
display_df = df_summary.head(20)
import caas_jupyter_tools as tools
tools.display_dataframe_to_user('Window Search Summary (first 20 rows)', display_df)

# Also provide full summary path
print('Done.')
