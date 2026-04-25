"""
Created on Wed Apr 15 12:08:34 2026

@author: AXILLIOS
"""
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
    elapsed = t2 - t1
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    milliseconds = int((elapsed * 1000) % 1000)
    print(f" time taken: {minutes:02d} min:{seconds:02d} s:{milliseconds:03d} millis") 
# ----------------------------- Kernel break ----------------------------------
def RETURN():
    sys.exit()
# =============================================================================
